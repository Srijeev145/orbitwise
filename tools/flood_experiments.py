"""
Experiments to close the flood AI's gap with Otsu on Indian / unseen floods.
Same protocol for every variant: train on the train split (1,000 labelled pixels per chip),
choose the cut-off (and any blend weight) on the validation split only, score water IoU on
test floods, Indian test chips and Bolivia (held-out event).

  A  current AI           : 8 radar features
  B  + permanent water    : A + JRC permanent-water map (pixel and 9x9 share)
  C  scene-normalised     : radar features relative to each image's own median brightness
  D  AI + Otsu blend      : weighted average of A's probability and Otsu's 0/1 vote
  E  B + Otsu blend       : weighted average of B's probability and Otsu's vote

    python tools/flood_experiments.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orbitwise import flood_features, water_scores  # noqa: E402

D = ROOT / "data_full" / "sen1floods11"
rng = np.random.default_rng(0)


def chip_features(vv, vh, jrc):
    base = flood_features(vv, vh)                                   # A
    perm = np.stack([jrc.astype("float32"), uniform_filter(jrc.astype("float32"), 9)], -1).reshape(-1, 2)
    vvn, vhn = np.nan_to_num(vv, nan=-25.0), np.nan_to_num(vh, nan=-30.0)
    norm = flood_features(vvn - np.median(vvn), vhn - np.median(vhn))  # C (same 8 features, scene-relative)
    return base, perm, norm


def load(split, n):
    rows = pd.read_csv(D / "flood_handlabeled" / f"flood_{split}_data.csv", header=None)
    out = {k: [] for k in ("A", "P", "C", "y", "otsu", "india")}
    for s1, lab in rows.itertuples(index=False):
        with rasterio.open(D / "S1Hand" / s1) as f:
            vv, vh = f.read().astype("float32")
        y = rasterio.open(D / "LabelHand" / lab).read(1).ravel()
        otsu = rasterio.open(D / "S1OtsuLabelHand" / s1.replace("S1Hand", "S1OtsuLabelHand")).read(1).ravel()
        jrc = rasterio.open(D / "JRCWaterHand" / s1.replace("S1Hand", "JRCWaterHand")).read(1)
        A, P, C = chip_features(vv, vh, jrc)
        ok = np.flatnonzero(y >= 0)
        take = rng.choice(ok, min(len(ok), n), replace=False) if len(ok) else ok
        for k, v in (("A", A), ("P", P), ("C", C)):
            out[k].append(v[take])
        out["y"].append(y[take]); out["otsu"].append(otsu[take]); out["india"].append(np.full(len(take), s1.startswith("India")))
    return {k: np.concatenate(v) for k, v in out.items()}


def fit(X, y):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_depth=8, random_state=0).fit(X, y)


def best_cut(p, y, grid=np.arange(0.1, 0.91, 0.05)):
    return max(grid, key=lambda t: water_scores(y, (p >= t).astype(int))["IoU_water"])


if __name__ == "__main__":
    tr, va = load("train", 1000), load("valid", 1000)
    te, bo = load("test", 3000), load("bolivia", 3000)
    feats = {"A": lambda d: d["A"], "B": lambda d: np.hstack([d["A"], d["P"]]), "C": lambda d: d["C"]}
    models = {k: fit(f(tr), tr["y"]) for k, f in feats.items()}
    prob = {k: {s: models[k].predict_proba(feats[k](d))[:, 1] for s, d in (("va", va), ("te", te), ("bo", bo))} for k in feats}

    variants = {}
    for k, name in (("A", "A · current AI"), ("B", "B · + permanent water"), ("C", "C · scene-normalised")):
        variants[name] = {s: prob[k][s] for s in ("va", "te", "bo")}
    for k, name in (("A", "D · AI + Otsu blend"), ("B", "E · B + Otsu blend")):
        # blend weight w on the AI probability, tuned with the cut-off on validation only
        best = max(((w, best_cut(w * prob[k]["va"] + (1 - w) * va["otsu"], va["y"])) for w in np.arange(0.2, 0.81, 0.1)),
                   key=lambda wc: water_scores(va["y"], ((wc[0] * prob[k]["va"] + (1 - wc[0]) * va["otsu"]) >= wc[1]).astype(int))["IoU_water"])
        w = best[0]
        variants[name] = {s: w * prob[k][s] + (1 - w) * d["otsu"] for s, d in (("va", va), ("te", te), ("bo", bo))}
        variants[name]["w"] = round(float(w), 2)

    rows = []
    def add(method, te_pred, bo_pred):
        rows.append({"method": method,
                     "Test floods": water_scores(te["y"], te_pred)["IoU_water"],
                     "Indian floods": water_scores(te["y"][te["india"]], te_pred[te["india"]])["IoU_water"],
                     "Unseen (Bolivia)": water_scores(bo["y"], bo_pred)["IoU_water"]})
    add("Fixed threshold (VH < -20 dB)", (te["A"][:, 1] < -20).astype(int), (bo["A"][:, 1] < -20).astype(int))
    add("Otsu threshold", te["otsu"].astype(int), bo["otsu"].astype(int))
    for name, v in variants.items():
        cut = best_cut(v["va"], va["y"])
        label = name + (f" (AI weight {v['w']})" if "w" in v else "")
        add(label, (v["te"] >= cut).astype(int), (v["bo"] >= cut).astype(int))
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "flood_experiments.csv"
    res.to_csv(out, index=False)
    print(res.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------
# F · blend tuned by leave-one-EVENT-out (closer to how the system meets a brand-new flood)
# ---------------------------------------------------------------------------------------------
def load_events(n=1000):
    out = []
    for split in ("train", "valid"):
        rows = pd.read_csv(D / "flood_handlabeled" / f"flood_{split}_data.csv", header=None)
        for s1, lab in rows.itertuples(index=False):
            with rasterio.open(D / "S1Hand" / s1) as f:
                vv, vh = f.read().astype("float32")
            y = rasterio.open(D / "LabelHand" / lab).read(1).ravel()
            otsu = rasterio.open(D / "S1OtsuLabelHand" / s1.replace("S1Hand", "S1OtsuLabelHand")).read(1).ravel()
            X = flood_features(vv, vh)
            ok = np.flatnonzero(y >= 0)
            take = rng.choice(ok, min(len(ok), n), replace=False) if len(ok) else ok
            out.append((s1.split("_")[0], X[take], y[take], otsu[take]))
    return out


def run_F():
    chips = load_events()
    events = sorted({c[0] for c in chips})
    grid_w, grid_t = np.arange(0.0, 1.01, 0.1), np.arange(0.1, 0.91, 0.05)
    score = np.zeros((len(grid_w), len(grid_t)))
    for ev in events:  # hold out one whole event
        trn = [c for c in chips if c[0] != ev]; hold = [c for c in chips if c[0] == ev]
        m = fit(np.vstack([c[1] for c in trn]), np.concatenate([c[2] for c in trn]))
        Xh, yh, oh = np.vstack([c[1] for c in hold]), np.concatenate([c[2] for c in hold]), np.concatenate([c[3] for c in hold])
        ph = m.predict_proba(Xh)[:, 1]
        for i, w in enumerate(grid_w):
            for j, t in enumerate(grid_t):
                score[i, j] += water_scores(yh, ((w * ph + (1 - w) * oh) >= t).astype(int))["IoU_water"] / len(events)
    i, j = np.unravel_index(score.argmax(), score.shape)
    w, t = grid_w[i], grid_t[j]
    print(f"F: leave-one-event-out over {len(events)} events picks AI weight {w:.1f}, cut-off {t:.2f} "
          f"(mean held-out IoU {score[i, j]:.3f}; pure AI best {score[-1].max():.3f}; pure Otsu {score[0].max():.3f})")
    m = fit(np.vstack([c[1] for c in chips]), np.concatenate([c[2] for c in chips]))
    te, bo = load("test", 3000), load("bolivia", 3000)
    pt, pb = w * m.predict_proba(te["A"])[:, 1] + (1 - w) * te["otsu"], w * m.predict_proba(bo["A"])[:, 1] + (1 - w) * bo["otsu"]
    r = {"method": f"F · blend tuned on held-out events (AI weight {w:.1f})",
         "Test floods": water_scores(te["y"], (pt >= t).astype(int))["IoU_water"],
         "Indian floods": water_scores(te["y"][te["india"]], (pt[te["india"]] >= t).astype(int))["IoU_water"],
         "Unseen (Bolivia)": water_scores(bo["y"], (pb >= t).astype(int))["IoU_water"]}
    print(pd.DataFrame([r]).round(3).to_string(index=False))
    res = pd.read_csv(ROOT / "results" / "flood_experiments.csv")
    pd.concat([res, pd.DataFrame([r])]).to_csv(ROOT / "results" / "flood_experiments.csv", index=False)
