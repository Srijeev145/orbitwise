"""
OrbitWise: flood to payout, and the satellites behind it. One system, one command:

    pip install -r requirements.txt
    python orbitwise.py

It runs on the small pieces of real data bundled in data_sample/ (no downloads), in five stages:

  TRACK A  1. Flood AI       : train on expert-labelled radar pixels (Sen1Floods11); score vs 2 baselines
           2. Flood map      : Kerala 2018 before/after Sentinel-1 radar -> new water, flooded farmland
           3. Payout         : parametric trigger per 1 km cell -> hectares to pay, ₹ if sum insured is set
  TRACK B  4. Collision risk : predict the final risk 2 days early on real ESA warnings; vs "do nothing"
           5. Debris screen  : propagate today's orbits 72 h; close approaches to Sentinel-1 and Cartosat

Everything it finds goes to results/report.md (plus results/summary.json and figures).
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_sample"
RESULTS = ROOT / "results"

CONFIG = {
    # --- Track A: payout rule (a design choice; tune on past floods to limit basis risk)
    "trigger_flooded_share": 0.30,   # pay a cell when >= 30% of its farmland is newly flooded
    "cell_px": 18,                   # 18 x 55 m pixels ~ 1 km cell, a stand-in for a village
    "min_cropland_share": 0.10,      # ignore cells with almost no farmland
    "sum_insured_per_ha": None,      # ₹/ha, PMFBY scale of finance for paddy (Alappuzha/Kottayam). FILL IN.
    "farmer_premium_rate": 0.02,     # PMFBY cap, kharif food crops (2% of sum insured)
    # --- Track B
    "high_risk_log10": -6,           # ESA Kelvins high-risk line: collision probability >= 1e-6
    "screen_hours": 72,
    "screen_step_s": 60,
    "refine_within_km": 500,         # coarse-step distance that triggers a 1-second recheck (objects move ~900 km per step)
    "report_within_km": 10.0,        # public orbit data (TLEs) are only good to ~1 km or worse
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# =============================================================================================
# TRACK A · 1. FLOOD AI
# =============================================================================================
FLOOD_FEATURES = ["VV", "VH", "VV_minus_VH", "VV_mean3", "VH_mean3", "VV_mean9", "VH_mean9", "VH_std9"]


def flood_features(vv, vh):
    """8 numbers per pixel from the two radar bands (dB). Water is dark (very negative) and smooth,
    so neighbourhood averages and texture separate a flooded field from one dark noisy pixel."""
    from scipy.ndimage import uniform_filter
    vv = np.nan_to_num(vv, nan=-25.0)
    vh = np.nan_to_num(vh, nan=-30.0)
    m9 = uniform_filter(vh, 9)
    f = [vv, vh, vv - vh, uniform_filter(vv, 3), uniform_filter(vh, 3), uniform_filter(vv, 9), m9,
         np.sqrt(np.maximum(uniform_filter(vh ** 2, 9) - m9 ** 2, 0))]
    return np.stack(f, -1).reshape(-1, len(FLOOD_FEATURES))


def water_scores(y, p):
    tp, fp, fn = ((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()
    return {"IoU_water": tp / (tp + fp + fn), "F1_water": 2 * tp / (2 * tp + fp + fn), "accuracy": float((p == y).mean())}


def stage1_flood_ai():
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = np.load(DATA / "flood_pixels.npz")
    X, y = d["X_train"].astype("float32"), d["y_train"]
    log(f"[1] training flood AI on {len(y):,} expert-labelled pixels ({y.mean() * 100:.1f}% water)")
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_depth=8, random_state=0).fit(X, y)
    pv = model.predict_proba(d["X_valid"].astype("float32"))[:, 1]
    cut = max(np.arange(0.2, 0.81, 0.05), key=lambda t: water_scores(d["y_valid"], (pv >= t).astype(int))["IoU_water"])
    rows = []
    for split, label in (("test", "Test floods"), ("bolivia", "Unseen flood (Bolivia)")):
        Xs, ys, otsu, ind = d[f"X_{split}"].astype("float32"), d[f"y_{split}"], d[f"otsu_{split}"], d[f"india_{split}"]
        preds = {"Fixed threshold (VH < -20 dB)": (Xs[:, 1] < -20).astype(int),
                 "Otsu threshold": otsu.astype(int),
                 "AI (Gradient Boosting)": (model.predict_proba(Xs)[:, 1] >= cut).astype(int)}
        for name, p in preds.items():
            rows.append({"evaluated_on": label, "method": name, **water_scores(ys, p)})
            if ind.any():
                rows.append({"evaluated_on": "Indian floods (test)", "method": name, **water_scores(ys[ind], p[ind])})
    # Hybrid: blend the AI's probability with Otsu's per-image vote. Weight 0.8 and cut-off 0.35 were chosen by
    # holding out one whole flood event at a time (tools/flood_experiments.py, experiment F), so they are tuned
    # for floods the model has never seen. The hybrid model is trained on train + validation pixels.
    W_AI, CUT_H = 0.8, 0.35
    hyb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_depth=8, random_state=0).fit(
        np.vstack([X, d["X_valid"].astype("float32")]), np.concatenate([y, d["y_valid"]]))
    for split, label in (("test", "Test floods"), ("bolivia", "Unseen flood (Bolivia)")):
        Xs, ys, otsu, ind = d[f"X_{split}"].astype("float32"), d[f"y_{split}"], d[f"otsu_{split}"], d[f"india_{split}"]
        p = ((W_AI * hyb.predict_proba(Xs)[:, 1] + (1 - W_AI) * otsu) >= CUT_H).astype(int)
        rows.append({"evaluated_on": label, "method": "Hybrid (AI + Otsu)", **water_scores(ys, p)})
        if ind.any():
            rows.append({"evaluated_on": "Indian floods (test)", "method": "Hybrid (AI + Otsu)", **water_scores(ys[ind], p[ind])})
    table = pd.DataFrame(rows)
    log(f"[1] done. cut-off {cut:.2f}\n" + table.round(3).to_string(index=False))
    return model, float(cut), table


# =============================================================================================
# TRACK A · 2. FLOOD MAP (Kerala 2018)
# =============================================================================================
def read_radar(path):
    import rasterio
    with rasterio.open(path) as f:
        a = f.read().astype("float32")
        a[a == -32768] = np.nan
        return a / 100, f.transform, f.shape


def worldcover_on_grid(transform, shape):
    """ESA WorldCover land-use classes on the radar grid (used only by tools/make_samples.py)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif"
    out = np.zeros(shape, "uint8")
    W, N = transform.c, transform.f
    E, S = W + transform.a * shape[1], N + transform.e * shape[0]
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    if Path("/root/.ccr/ca-bundle.crt").exists():          # proxy certificate in the cloud sandbox only
        env["GDAL_CURL_CA_BUNDLE"] = "/root/.ccr/ca-bundle.crt"
    with rasterio.Env(**env):
        for la in range(int(S // 3 * 3), int(np.ceil(N / 3) * 3), 3):
            for lo in range(int(W // 3 * 3), int(np.ceil(E / 3) * 3), 3):
                with rasterio.open("/vsicurl/" + url.format(t=f"N{la:02d}E{lo:03d}")) as src, \
                     WarpedVRT(src, crs="EPSG:4326", transform=transform, width=shape[1], height=shape[0],
                               resampling=Resampling.mode, nodata=0) as v:
                    a = v.read(1)
                out = np.where(out == 0, a, out)
    return out


def drop_specks(mask, min_px=8):
    from scipy.ndimage import label
    lab, n = label(mask)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_px
    keep[0] = False
    return keep[lab]


def stage2_flood_map(model, cut):
    from scipy.ndimage import median_filter
    before, tr, shape = read_radar(DATA / "KL2018_t165_before.tif")
    after, _, _ = read_radar(DATA / "KL2018_t165_after_0821.tif")
    wc = np.load(DATA / "KL2018_worldcover.npz")["worldcover"]
    log("[2] mapping Kerala flood: before = Apr-May 2018 median, after = 21 Aug 2018 (Sentinel-1, track 165)")

    valid = np.isfinite(before[1]) & np.isfinite(after[1]) & ~np.isin(wc, [0, 80])  # drop sea + permanent water
    b = median_filter(np.nan_to_num(before[1], nan=0), 3)
    a = median_filter(np.nan_to_num(after[1], nan=0), 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline = drop_specks(valid & (a / np.where(b == 0, np.nan, b) > 1.25))  # UN-SPIDER change detection

    def water(img):
        return (model.predict_proba(flood_features(img[0], img[1]))[:, 1] >= cut).reshape(shape)
    ai = drop_specks(valid & water(after) & ~water(before))                     # new water only

    lat_mid = tr.f + tr.e * shape[0] / 2
    px_km2 = abs(tr.a * tr.e) * 110.57 * 111.32 * np.cos(np.radians(lat_mid))
    rows = np.arange(shape[0]) * tr.e + tr.f
    cols = np.arange(shape[1]) * tr.a + tr.c
    box = (rows[:, None] >= 9.05) & (rows[:, None] <= 9.95) & (cols[None, :] >= 76.25) & (cols[None, :] <= 76.8)
    stats = {}
    for name, m in (("baseline", baseline), ("ai", ai)):
        stats[name] = {"flooded_km2": float(m.sum() * px_km2),
                       "cropland_km2": float((m & (wc == 40)).sum() * px_km2),
                       "builtup_km2": float((m & (wc == 50)).sum() * px_km2),
                       "alappuzha_kottayam_km2": float((m & box).sum() * px_km2)}
    log(f"[2] new water: baseline {stats['baseline']['flooded_km2']:.0f} km², AI {stats['ai']['flooded_km2']:.0f} km² | "
        f"farmland: {stats['baseline']['cropland_km2']:.0f} / {stats['ai']['cropland_km2']:.0f} km²")
    return {"before": before, "after": after, "baseline": baseline, "ai": ai, "wc": wc,
            "transform": tr, "shape": shape, "px_km2": px_km2, "stats": stats}


# =============================================================================================
# TRACK A · 3. PAYOUT
# =============================================================================================
def stage3_payout(fm):
    c = CONFIG["cell_px"]
    H, W = (fm["shape"][0] // c) * c, (fm["shape"][1] // c) * c
    crop = (fm["wc"] == 40)[:H, :W]
    out = {}
    for method in ("baseline", "ai"):
        fl = fm[method][:H, :W] & crop
        crop_cells = crop.reshape(H // c, c, W // c, c).sum(axis=(1, 3))
        fl_cells = fl.reshape(H // c, c, W // c, c).sum(axis=(1, 3))
        share = np.where(crop_cells > 0, fl_cells / np.maximum(crop_cells, 1), 0)
        eligible = crop_cells >= CONFIG["min_cropland_share"] * c * c
        trig = eligible & (share >= CONFIG["trigger_flooded_share"])
        ha = float(fl_cells[trig].sum() * fm["px_km2"] * 100)
        si = CONFIG["sum_insured_per_ha"]
        out[method] = {"cells_with_farmland": int(eligible.sum()), "cells_triggered": int(trig.sum()),
                       "hectares_to_pay": ha,
                       "payout_crore": None if si is None else ha * si / 1e7,
                       "farmer_premium_per_ha": None if si is None else si * CONFIG["farmer_premium_rate"],
                       "trigger_grid": trig}
    msg = " | ".join(f"{m}: {o['cells_triggered']} of {o['cells_with_farmland']} cells, {o['hectares_to_pay']:,.0f} ha" for m, o in out.items())
    log(f"[3] payout trigger (>= {CONFIG['trigger_flooded_share']:.0%} farmland flooded per ~1 km cell): {msg}")
    if CONFIG["sum_insured_per_ha"] is None:
        log("[3] set CONFIG['sum_insured_per_ha'] (₹/ha) to turn hectares into a ₹ payout")
    return out


# =============================================================================================
# TRACK B · 4. COLLISION RISK, 2 DAYS EARLY
# =============================================================================================
RISK_FEATURES = ["time_to_tca", "risk", "miss_distance", "relative_speed", "max_risk_estimate", "mahalanobis_distance"]


def stage4_collision_risk():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import fbeta_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    ev = pd.read_csv(DATA / "kelvins_events_train.csv")
    ev = ev[ev.final_days < 1].copy()                      # events with a final answer
    ev["high_risk"] = (ev.final_risk >= CONFIG["high_risk_log10"]).astype(int)
    X, y = ev[RISK_FEATURES], ev.high_risk
    log(f"[4] {len(ev):,} real ESA events, {y.sum()} end high-risk; input = latest warning >= 2 days out")
    X_tr, X_o, y_tr, y_o = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_v, X_c, y_v, y_c = train_test_split(X_o, y_o, test_size=0.5, stratify=y_o, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=300, max_depth=3, learning_rate=0.05,
                                           class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    pv = model.predict_proba(X_v)[:, 1]
    cut = max([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], key=lambda t: fbeta_score(y_v, pv >= t, beta=2, zero_division=0))
    res = {}
    for name, alert in (("Do nothing (trust 2-day-old risk)", X_c.risk >= CONFIG["high_risk_log10"]),
                        ("AI (Gradient Boosting)", model.predict_proba(X_c)[:, 1] >= cut)):
        res[name] = {"caught": int((alert & (y_c == 1)).sum()), "dangerous": int(y_c.sum()),
                     "precision": precision_score(y_c, alert, zero_division=0),
                     "recall": recall_score(y_c, alert, zero_division=0),
                     "F2": fbeta_score(y_c, alert, beta=2, zero_division=0), "alerts": int(alert.sum())}
    model.fit(X, y)                                         # final model on all labelled events
    test = pd.read_csv(DATA / "kelvins_events_test.csv")
    test["ai_score"] = model.predict_proba(test[RISK_FEATURES])[:, 1]
    test["alert"] = np.where(test.ai_score >= cut, "HIGH", np.where(test.ai_score >= cut / 2, "MEDIUM", "LOW"))
    table = pd.DataFrame(res).T
    log("[4] check set:\n" + table.round(3).to_string())
    return {"table": table, "cutoff": cut, "test_alerts": test.alert.value_counts().to_dict(),
            "top_test": test.sort_values("ai_score", ascending=False).head(5)[["event_id", "time_to_tca", "risk", "ai_score", "alert"]]}


# =============================================================================================
# TRACK B · 5. LIVE DEBRIS SCREEN
# =============================================================================================
def read_tles(path):
    lines = [l.rstrip("\r\n") for l in open(path, encoding="utf-8", errors="ignore") if l.strip()]
    out = []
    for i in range(len(lines) - 2):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
    return out


def stage5_debris_screen():
    from sgp4.api import Satrec, SatrecArray
    targets, objects = [], []
    for name, l1, l2 in read_tles(DATA / "tle_sentinel1.txt"):
        (objects if name.strip() == "SENTINEL-1B" else targets).append((name, l1, l2))   # 1B is dead: an object
    targets += read_tles(DATA / "tle_cartosat.txt")
    for f in sorted(DATA.glob("tle_*debris*.txt")):
        objects += read_tles(f)
    T = [Satrec.twoline2rv(l1, l2) for _, l1, l2 in targets]
    O = [Satrec.twoline2rv(l1, l2) for _, l1, l2 in objects]
    n_debris = len(objects)
    # active satellites: keep only those whose orbit crosses the targets' altitude band
    active_file = DATA / "tle_active.txt"
    n_active = 0
    if active_file.exists():
        def alt_range(sat):
            a = (398600.4418 / (sat.no_kozai / 60.0) ** 2) ** (1 / 3)           # semi-major axis, km
            return a * (1 - sat.ecco) - 6378.137, a * (1 + sat.ecco) - 6378.137  # perigee, apogee altitude
        lo = min(alt_range(t)[0] for t in T) - 50
        hi = max(alt_range(t)[1] for t in T) + 50
        skip = {t.satnum for t in T} | {o.satnum for o in O}
        for name, l1, l2 in read_tles(active_file):
            sat = Satrec.twoline2rv(l1, l2)
            per, apo = alt_range(sat)
            if sat.satnum not in skip and per < hi and apo > lo:
                objects.append((name, l1, l2)); O.append(sat); skip.add(sat.satnum); n_active += 1
        log(f"[5] + {n_active:,} active satellites whose orbits cross {lo:.0f}-{hi:.0f} km altitude")
    start_jd = max(s.jdsatepoch + s.jdsatepochF for s in T)
    step, n = CONFIG["screen_step_s"], int(CONFIG["screen_hours"] * 3600 / CONFIG["screen_step_s"])
    log(f"[5] screening {len(targets)} satellites vs {len(objects):,} objects for {CONFIG['screen_hours']} h "
        f"({n:,} time steps of {step} s)")
    ta, oa = SatrecArray(T), SatrecArray(O)
    cand = set()                                    # (target, object, coarse time) to recheck
    chunk = 480
    for k in range(0, n, chunk):
        t_days = (np.arange(k, min(k + chunk, n)) * step) / 86400.0
        jd, fr = np.full_like(t_days, np.floor(start_jd)), (start_jd - np.floor(start_jd)) + t_days
        et, rt, _ = ta.sgp4(jd, fr)
        eo, ro, _ = oa.sgp4(jd, fr)
        rt[et != 0] = np.nan
        ro[eo != 0] = np.nan
        for i in range(len(T)):
            d = np.linalg.norm(ro - rt[i][None, :, :], axis=2)              # (objects, times) km
            d = np.where(np.isnan(d), np.inf, d)
            # every local minimum under the threshold is a separate meeting (pairs can meet every orbit)
            pad = np.pad(d, ((0, 0), (1, 1)), constant_values=np.inf)
            local = (d <= pad[:, :-2]) & (d < pad[:, 2:]) & (d < CONFIG["refine_within_km"])
            for o, tk in zip(*np.nonzero(local)):
                cand.add((i, int(o), float(t_days[tk])))
    # recheck every pair that came within refine_within_km at 1-second steps around the coarse minimum
    log(f"[5] rechecking {len(cand):,} coarse close passes at 1-second steps")
    best = {}
    for i, j, t0 in sorted(cand):
        tt = t0 + np.arange(-step, step + 1) / 86400.0
        jd, fr = np.full_like(tt, np.floor(start_jd)), (start_jd - np.floor(start_jd)) + tt
        _, r1, v1 = T[i].sgp4_array(jd, fr)
        _, r2, v2 = O[j].sgp4_array(jd, fr)
        d = np.linalg.norm(r1 - r2, axis=1)
        m = int(np.nanargmin(d))
        when = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=float(jd[m] + fr[m] - 2451545.0))
        row = {"satellite": targets[i][0], "object": objects[j][0], "closest_km": float(d[m]),
               "relative_speed_km_s": float(np.linalg.norm(v1[m] - v2[m])),
               "time_utc": when.strftime("%Y-%m-%d %H:%M:%S")}
        if (i, j) not in best or row["closest_km"] < best[(i, j)]["closest_km"]:
            best[(i, j)] = row
    rows = list(best.values())
    close = pd.DataFrame(rows, columns=["satellite", "object", "closest_km", "relative_speed_km_s", "time_utc"])
    close = close.sort_values("closest_km").reset_index(drop=True)
    start = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=start_jd - 2451545.0)
    within = close[close.closest_km <= CONFIG["report_within_km"]]
    log(f"[5] {len(within)} approaches within {CONFIG['report_within_km']:.0f} km (window from {start:%Y-%m-%d %H:%M} UTC)")
    if len(close):
        log("[5] closest:\n" + close.head(8).round(2).to_string(index=False))
    close["object_type"] = ["debris" if ("DEB" in o or o.strip() == "SENTINEL-1B") else "active satellite" for o in close.object]
    return {"close": close, "start": start, "n_targets": len(targets), "n_objects": len(objects),
            "n_debris": n_debris, "n_active": n_active}


# =============================================================================================
# REPORT
# =============================================================================================
def figure(fm, pay, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr, shape = fm["transform"], fm["shape"]
    ext = [tr.c, tr.c + tr.a * shape[1], tr.f + tr.e * shape[0], tr.f]
    fig, ax = plt.subplots(1, 4, figsize=(20, 7), constrained_layout=True)
    ax[0].imshow(fm["before"][1], cmap="gray", vmin=-25, vmax=-5, extent=ext); ax[0].set_title("Before: Apr-May 2018 (VH)")
    ax[1].imshow(fm["after"][1], cmap="gray", vmin=-25, vmax=-5, extent=ext); ax[1].set_title("After: 21 Aug 2018 (VH)")
    ax[2].imshow(fm["after"][1], cmap="gray", vmin=-25, vmax=-5, extent=ext)
    ax[2].imshow(np.ma.masked_where(~fm["baseline"], fm["baseline"]), cmap="cool", alpha=0.7, extent=ext, interpolation="nearest")
    ax[2].imshow(np.ma.masked_where(~fm["ai"], fm["ai"]), cmap="winter", alpha=0.9, extent=ext, interpolation="nearest")
    ax[2].set_title("New water: baseline (light blue) + AI (dark blue)")
    c = CONFIG["cell_px"]
    g = pay["baseline"]["trigger_grid"]
    ax[3].imshow(fm["after"][1], cmap="gray", vmin=-25, vmax=-5, extent=ext)
    ax[3].imshow(np.ma.masked_where(~g, g), cmap="autumn", alpha=0.8, interpolation="nearest",
                 extent=[tr.c, tr.c + tr.a * g.shape[1] * c, tr.f + tr.e * g.shape[0] * c, tr.f])
    ax[3].set_title(f"Payout triggered: {g.sum()} cells (~1 km)")
    for a in ax:
        a.set_xlabel("Longitude °E"); a.set_ylabel("Latitude °N")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_report(flood_table, fm, pay, risk, screen):
    RESULTS.mkdir(exist_ok=True)
    figure(fm, pay, RESULTS / "kerala_flood_and_payout.png")
    s = fm["stats"]
    piv = flood_table.pivot_table(index="method", columns="evaluated_on", values="IoU_water").round(3)
    close = screen["close"]
    lines = [
        "# OrbitWise run report", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. All numbers below come from this run.", "",
        "## Track A · 1. Flood AI (water IoU: overlap ÷ union with expert labels; 1.0 = perfect)", "",
        piv.to_markdown(), "",
        "## Track A · 2. Kerala flood, 21 Aug 2018", "",
        "| | Baseline (UN-SPIDER) | AI |", "|---|---|---|",
        f"| New water (km²) | {s['baseline']['flooded_km2']:.0f} | {s['ai']['flooded_km2']:.0f} |",
        f"| Of which farmland (km²) | {s['baseline']['cropland_km2']:.0f} | {s['ai']['cropland_km2']:.0f} |",
        f"| Alappuzha + Kottayam box (km²) | {s['baseline']['alappuzha_kottayam_km2']:.0f} | {s['ai']['alappuzha_kottayam_km2']:.0f} |",
        "", "IWMI / UN-SPIDER measured 407 km² in Alappuzha + Kottayam on 14 Aug 2018 (flood peak; water receded by 21 Aug).", "",
        "![Kerala flood and payout](kerala_flood_and_payout.png)", "",
        f"## Track A · 3. Payout (trigger: ≥ {CONFIG['trigger_flooded_share']:.0%} of a ~1 km cell's farmland newly flooded)", "",
        "| | Baseline | AI |", "|---|---|---|",
        f"| Cells with farmland | {pay['baseline']['cells_with_farmland']} | {pay['ai']['cells_with_farmland']} |",
        f"| Cells triggered | {pay['baseline']['cells_triggered']} | {pay['ai']['cells_triggered']} |",
        f"| Hectares to pay | {pay['baseline']['hectares_to_pay']:,.0f} | {pay['ai']['hectares_to_pay']:,.0f} |",
        "| Payout (₹ crore) | " + " | ".join("set sum_insured_per_ha" if pay[m]["payout_crore"] is None else f"{pay[m]['payout_crore']:,.1f}" for m in ("baseline", "ai")) + " |",
        "", "PMFBY: claims are stipulated to be paid within two months of crop-cutting experiments / harvest "
        "(PIB, Ministry of Agriculture & Farmers Welfare, 19 Jul 2022). This trigger is computed minutes after the satellite pass.", "",
        "## Track B · 4. Collision risk, 2 days before closest approach (real ESA Kelvins data)", "",
        risk["table"].round(3).to_markdown(), "",
        f"Alert cut-off {risk['cutoff']}. Alerts on the {sum(risk['test_alerts'].values()):,} ESA test events: {risk['test_alerts']}.", "",
        f"## Track B · 5. Debris screen: {screen['n_targets']} satellites vs {screen['n_objects']:,} objects, "
        f"{CONFIG['screen_hours']} h from {screen['start']:%Y-%m-%d %H:%M} UTC", "",
        (close.head(10).round(2).to_markdown(index=False) if len(close) else "No approaches under 50 km."), "",
        "Public orbit data (TLEs) carry errors of a kilometre or more: these are screening candidates, "
        "the step before an operator requests precise data.", "",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"flood_ai": flood_table.to_dict("records"), "kerala": s,
               "payout": {m: {k: v for k, v in o.items() if k != "trigger_grid"} for m, o in pay.items()},
               "collision_risk": risk["table"].to_dict("index"), "test_alerts": risk["test_alerts"],
               "debris_closest": close.head(10).to_dict("records"), "config": CONFIG}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(f"report written: {RESULTS / 'report.md'}")


# =============================================================================================
# CHARTS (for slides): one idea per chart, value labels on every bar, recessive axes
# =============================================================================================
AI_BLUE, BASE_ORANGE, THIRD_AQUA = "#257de0", "#f47230", "#00b894"   # validated categorical slots 1-3
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
PMFBY_CLAIMS_CRORE = {"2018-19": 28464, "2019-20": 26413, "2020-21": 17932, "2021-22\n(kharif only)": 7558}  # PIB, 19 Jul 2022


def _style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=17, color=INK, pad=14, fontweight="bold")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors=INK2, labelsize=13, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=13)


def _bar_labels(ax, bars, fmt):
    for b in bars:
        h = b.get_height() if b.get_height() == b.get_height() else 0
        ax.annotate(fmt(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=13, color=INK)


def make_charts(flood_table, fm, pay, risk, screen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = RESULTS / "charts"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

    # 1. flood AI vs baselines
    piv = flood_table.pivot_table(index="evaluated_on", columns="method", values="IoU_water")
    groups = ["Test floods", "Indian floods (test)", "Unseen flood (Bolivia)"]
    methods = [("Fixed threshold (VH < -20 dB)", THIRD_AQUA, "Fixed threshold"),
               ("Otsu threshold", BASE_ORANGE, "Otsu (standard)"), ("AI (Gradient Boosting)", AI_BLUE, "Our AI"),
               ("Hybrid (AI + Otsu)", "#6a4ce2", "Hybrid (AI + Otsu)")]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x, w = np.arange(len(groups)), 0.2
    for k, (m, c, lab) in enumerate(methods):
        bars = ax.bar(x + (k - 1.5) * (w + 0.015), [piv.loc[g, m] for g in groups], w, color=c, label=lab)
        _bar_labels(ax, bars, lambda v: f"{v:.2f}")
    ax.set_xticks(x, ["Test floods", "Indian floods", "Unseen flood (Bolivia)"])
    ax.set_ylim(0, 0.92)
    _style(ax, "Flood detection accuracy (water IoU, higher is better)")
    ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=4)
    fig.tight_layout(); fig.savefig(out / "flood_ai_iou.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # 2. Kerala: baseline vs AI, all in km2
    s = fm["stats"]
    cats = ["New water", "Flooded farmland", "Farmland in payout cells"]
    vals = {m: [s[m]["flooded_km2"], s[m]["cropland_km2"], pay[m]["hectares_to_pay"] / 100] for m in ("baseline", "ai")}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x, w = np.arange(3), 0.36
    for k, (m, c, lab) in enumerate((("baseline", BASE_ORANGE, "Change detection (UN-SPIDER)"), ("ai", AI_BLUE, "Our AI"))):
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals[m], w, color=c, label=lab)
        _bar_labels(ax, bars, lambda v: f"{v:.0f}")
    ax.set_xticks(x, cats)
    _style(ax, "Kerala flood, 21 Aug 2018 (km²)")
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "kerala_km2.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # 3. collision risk: two small charts, same orientation, no dual axis
    t = risk["table"]
    names = ["Do nothing", "Our AI"]
    rows = ["Do nothing (trust 2-day-old risk)", "AI (Gradient Boosting)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, col, title, fmt in ((axes[0], "caught", f"Dangerous events caught (of {int(t.iloc[0].dangerous)})", lambda v: f"{v:.0f}"),
                                (axes[1], "alerts", "Alerts raised", lambda v: f"{v:.0f}")):
        bars = ax.bar(names, [t.loc[r, col] for r in rows], 0.55, color=[BASE_ORANGE, AI_BLUE])
        _bar_labels(ax, bars, fmt)
        _style(ax, title)
        ax.set_ylim(0, max(t[col]) * 1.25)
    fig.suptitle("Collision risk, 2 days before closest approach (real ESA data)", x=0.02, ha="left", fontsize=15, color=INK2)
    fig.tight_layout(); fig.savefig(out / "collision_risk.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # 4. debris screen: closest approaches to Sentinel-1 (our flood satellites), horizontal bars
    allc = screen["close"]
    c = allc[allc.satellite.str.contains("SENTINEL")].head(10).iloc[::-1]
    if len(c):
        def short(n):
            return n.replace(" DEB", " debris").title()
        labels = [f"{short(a)}  ↔  {short(b)}" for a, b in zip(c.satellite, c.object)]
        colors = [BASE_ORANGE if t == "debris" else THIRD_AQUA for t in c.object_type]
        fig, ax = plt.subplots(figsize=(11, 6.4))
        y = np.arange(len(c))                       # one row per approach (a pair can appear twice)
        bars = ax.barh(y, c.closest_km, 0.6, color=colors)
        ax.set_yticks(y, labels)
        for b, v, when in zip(bars, c.closest_km, c.time_utc):
            ax.annotate(f"{v:.1f} km · {when[5:16]} UTC", (v, b.get_y() + b.get_height() / 2), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=12, color=INK)
        ax.set_xlim(0, CONFIG["report_within_km"] * 1.7)
        _style(ax, "Closest approaches to Sentinel-1 in the next 72 h (km)")
        ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=BASE_ORANGE, label="Debris"), Patch(color=THIRD_AQUA, label="Active satellite")],
                  frameon=False, fontsize=12, loc="upper right")
        fig.tight_layout(); fig.savefig(out / "debris_closest.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # 5. PMFBY claims paid (market size), single series
    fig, ax = plt.subplots(figsize=(11, 5.6))
    yrs = list(PMFBY_CLAIMS_CRORE)
    bars = ax.bar(yrs, list(PMFBY_CLAIMS_CRORE.values()), 0.55, color=AI_BLUE)
    bars[-1].set_alpha(0.45)
    _bar_labels(ax, bars, lambda v: f"₹{v:,.0f} cr")
    ax.set_ylim(0, 33000)
    ax.set_yticks([0, 10000, 20000, 30000], ["0", "10k", "20k", "30k"])
    _style(ax, "PMFBY crop-insurance claims paid (₹ crore)")
    fig.text(0.01, 0.01, "Source: PIB, Ministry of Agriculture & Farmers Welfare, 19 Jul 2022 (data as of 30 Jun 2022)",
             fontsize=11, color=INK2)
    fig.tight_layout(); fig.savefig(out / "pmfby_claims.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    log(f"charts written: {out}")


def main():
    t0 = time.time()
    model, cut, flood_table = stage1_flood_ai()
    fm = stage2_flood_map(model, cut)
    pay = stage3_payout(fm)
    risk = stage4_collision_risk()
    screen = stage5_debris_screen()
    write_report(flood_table, fm, pay, risk, screen)
    make_charts(flood_table, fm, pay, risk, screen)
    log(f"all stages done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
