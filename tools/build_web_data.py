"""
Exports the real outputs of orbitwise.py for the interactive website (demo/):
  data.json            numbers, per-cell farmland grid, collision events + timelines, debris approaches
  radar_before.png     Kerala VH radar, Apr-May 2018 (grey)
  radar_after.png      Kerala VH radar, 21 Aug 2018 (grey)
  flood_base.png       change-detection new water (transparent overlay)
  flood_ai.png         AI new water (transparent overlay)

    python tools/build_web_data.py [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import orbitwise as ow  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo"
OUT.mkdir(parents=True, exist_ok=True)
DS = 2  # downsample radar for the web (55 m -> 110 m pixels)


def grey_png(db, path):
    v = np.clip((np.nan_to_num(db, nan=-30) + 25) / 20, 0, 1)
    Image.fromarray((v[::DS, ::DS] * 255).astype("uint8"), "L").save(path, optimize=True)


def mask_png(mask, rgb, path):
    m = mask[::DS, ::DS]
    a = np.zeros(m.shape + (4,), "uint8")
    a[m] = (*rgb, 235)
    Image.fromarray(a, "RGBA").save(path, optimize=True)


def collision_events():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import fbeta_score
    from sklearn.model_selection import train_test_split
    ev = pd.read_csv(ow.DATA / "kelvins_events_train.csv")
    ev = ev[ev.final_days < 1].copy()
    ev["high_risk"] = (ev.final_risk >= ow.CONFIG["high_risk_log10"]).astype(int)
    X, y = ev[ow.RISK_FEATURES], ev.high_risk
    X_tr, X_o, y_tr, y_o = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_v, X_c, y_v, y_c = train_test_split(X_o, y_o, test_size=0.5, stratify=y_o, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=300, max_depth=3, learning_rate=0.05,
                                           class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    pv = model.predict_proba(X_v)[:, 1]
    cut = max([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], key=lambda t: fbeta_score(y_v, pv >= t, beta=2, zero_division=0))
    check = ev.loc[X_c.index].copy()
    check["ai_score"] = model.predict_proba(X_c)[:, 1]
    # full warning timelines for every check event (time to closest approach, log10 risk)
    raw = pd.read_csv(ROOT / "data_full/kelvins/train_data.csv",
                      usecols=["event_id", "time_to_tca", "risk", "miss_distance", "relative_speed", "c_object_type"])
    raw = raw[raw.event_id.isin(check.event_id)].sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    tl = {int(k): [[round(float(a), 3), round(float(b), 3)] for a, b in zip(g.time_to_tca, g.risk)]
          for k, g in raw.groupby("event_id")}
    otype = raw.groupby("event_id").c_object_type.last().to_dict()
    events = [{"id": int(r.event_id), "d2": round(float(r.time_to_tca), 2), "r2": round(float(r.risk), 3),
               "fin": round(float(r.final_risk), 3), "ai": round(float(r.ai_score), 4), "y": int(r.high_risk),
               "miss": round(float(r.miss_distance)), "spd": round(float(r.relative_speed)),
               "obj": str(otype.get(r.event_id, "")), "tl": tl.get(int(r.event_id), [])}
              for r in check.itertuples()]
    return {"cutoff": cut, "n_events_total": int(len(ev)), "n_high_total": int(y.sum()), "events": events}


def main():
    print("stage 1-3 (flood)")
    model, cut, flood_table = ow.stage1_flood_ai()
    fm = ow.stage2_flood_map(model, cut)
    grey_png(fm["before"][1], OUT / "radar_before.png")
    grey_png(fm["after"][1], OUT / "radar_after.png")
    mask_png(fm["baseline"], (90, 200, 250), OUT / "flood_base.png")
    mask_png(fm["ai"], (37, 99, 235), OUT / "flood_ai.png")
    c = ow.CONFIG["cell_px"]
    H, W = (fm["shape"][0] // c) * c, (fm["shape"][1] // c) * c
    crop = (fm["wc"] == 40)[:H, :W]
    blk = lambda m: m.reshape(H // c, c, W // c, c).sum(axis=(1, 3)).astype(int)
    tr = fm["transform"]
    flood = {
        "bounds": {"west": tr.c, "north": tr.f, "east": tr.c + tr.a * fm["shape"][1], "south": tr.f + tr.e * fm["shape"][0]},
        "img_w": int(fm["shape"][1] // DS), "img_h": int(fm["shape"][0] // DS),
        "cell_px": c, "cell_rows": H // c, "cell_cols": W // c, "px_km2": fm["px_km2"],
        "grid_frac_w": W / fm["shape"][1], "grid_frac_h": H / fm["shape"][0],
        "crop": blk(crop).ravel().tolist(),
        "fl_base": blk(fm["baseline"][:H, :W] & crop).ravel().tolist(),
        "fl_ai": blk(fm["ai"][:H, :W] & crop).ravel().tolist(),
        "stats": fm["stats"],
        "accuracy": flood_table.round(4).to_dict("records"),
        "ai_cutoff": cut,
    }
    print("stage 4 (collision risk)")
    coll = collision_events()
    rf = pd.read_csv(ROOT / "archive/step_by_step_scripts/trackB_results/model_comparison.csv")
    rf = rf[(rf.features.str.startswith("B")) | (rf.model.str.startswith("Baseline"))]
    coll["models"] = rf[["model", "accuracy", "high_recall", "high_precision"]].round(4).to_dict("records")
    print("stage 5 (debris screen)")
    scr = ow.stage5_debris_screen()
    close = scr["close"]
    close = close[close.closest_km <= ow.CONFIG["report_within_km"]]
    debris = {"start_utc": scr["start"].strftime("%Y-%m-%d %H:%M"), "hours": ow.CONFIG["screen_hours"],
              "n_targets": scr["n_targets"], "n_objects": scr["n_objects"], "n_debris": scr["n_debris"],
              "n_active": scr["n_active"],
              "rows": [{"sat": r.satellite.strip(), "obj": r.object.strip(), "km": round(r.closest_km, 2),
                        "v": round(r.relative_speed_km_s, 2), "t": r.time_utc, "type": r.object_type}
                       for r in close.itertuples()]}
    data = {"flood": flood, "collision": coll, "debris": debris,
            "config": {k: v for k, v in ow.CONFIG.items() if k != "sum_insured_per_ha"}}
    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":"), default=float))
    print("written", OUT, round((OUT / "data.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
