"""
AI flood mapping on Sen1Floods11 (hand-labelled Sentinel-1 flood chips, 11 events, 6 continents).

Compares, on pixels experts labelled:
  1. Fixed threshold   : water if VH < -20 dB            (simplest rule)
  2. Otsu threshold    : per-chip automatic threshold    (dataset's own baseline labels)
  3. Gradient Boosting : learns from VV, VH, their ratio and neighbourhood texture

Scored on the official test split AND on Bolivia, a flood event never seen in training.
Metric: IoU of water (overlap / union of predicted and true water), plus F1 and accuracy.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier

D = Path(__file__).parent / "data" / "sen1floods11"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
FEATURES = ["VV", "VH", "VV_minus_VH", "VV_mean3", "VH_mean3", "VV_mean9", "VH_mean9", "VH_std9"]


def otsu(x):
    """Automatic threshold separating dark (water) from bright (land) pixels in one image."""
    h, e = np.histogram(x[np.isfinite(x)], bins=256)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def features(vv, vh):
    """Per-pixel features from two radar bands in dB (works for any image, not just these chips)."""
    vv = np.nan_to_num(vv, nan=-25.0); vh = np.nan_to_num(vh, nan=-30.0)
    m9 = uniform_filter(vh, 9)
    f = [vv, vh, vv - vh, uniform_filter(vv, 3), uniform_filter(vh, 3), uniform_filter(vv, 9), m9,
         np.sqrt(np.maximum(uniform_filter(vh ** 2, 9) - m9 ** 2, 0))]
    return np.stack(f, -1).reshape(-1, len(FEATURES))


def load(split):
    rows = pd.read_csv(D / "flood_handlabeled" / f"flood_{split}_data.csv", header=None)
    for s1name, lab in rows.itertuples(index=False):
        with rasterio.open(D / "S1Hand" / s1name) as d:
            vv, vh = d.read().astype("float32")
        with rasterio.open(D / "LabelHand" / lab) as d:
            y = d.read(1).ravel()
        with rasterio.open(D / "S1OtsuLabelHand" / s1name.replace("S1Hand", "S1OtsuLabelHand")) as d:
            otsu_lab = d.read(1).ravel()
        yield s1name, features(vv, vh), y, otsu_lab


def scores(y, p):
    tp, fp, fn = ((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()
    return {"IoU_water": tp / (tp + fp + fn), "F1_water": 2 * tp / (2 * tp + fp + fn), "accuracy": (p == y).mean()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    def sample(split, n=4000):
        X, Y = [], []
        for _, f, y, _ in load(split):
            ok = np.flatnonzero(y >= 0)  # -1 = not labelled (cloud / unsure)
            take = rng.choice(ok, min(len(ok), n), replace=False) if len(ok) else ok
            X.append(f[take]); Y.append(y[take])
        return np.concatenate(X), np.concatenate(Y)

    X, Y = sample("train")
    Xv, Yv = sample("valid", 20000)
    print(f"training pixels: {len(Y):,} ({Y.mean() * 100:.1f}% water)")
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_depth=8,
                                           random_state=0).fit(X, Y)
    # choose the water/not-water cutoff that maximises IoU on the validation events only
    pv = model.predict_proba(Xv)[:, 1]
    CUT = max(np.arange(0.2, 0.81, 0.05), key=lambda t: scores(Yv, (pv >= t).astype(int))["IoU_water"])
    print(f"cutoff chosen on validation: {CUT:.2f}")
    joblib.dump({"model": model, "cutoff": CUT, "features": FEATURES}, OUT / "flood_model.joblib")

    rows = []
    for split, label in (("test", "Test split (seen events)"), ("bolivia", "Bolivia (unseen event)")):
        ys, preds = [], {"Fixed threshold (VH < -20 dB)": [], "Otsu threshold": [], "Gradient Boosting (AI)": []}
        india = {k: [] for k in preds}; yi = []
        for name, f, y, otsu_lab in load(split):
            ok = y >= 0
            p = {"Fixed threshold (VH < -20 dB)": (f[:, 1] < -20).astype(int),
                 "Otsu threshold": otsu_lab, "Gradient Boosting (AI)": (model.predict_proba(f)[:, 1] >= CUT).astype(int)}
            ys.append(y[ok])
            for k in preds:
                preds[k].append(p[k][ok])
            if name.startswith("India"):
                yi.append(y[ok]); [india[k].append(p[k][ok]) for k in preds]
        y = np.concatenate(ys)
        for k, v in preds.items():
            rows.append({"evaluated_on": label, "method": k, **scores(y, np.concatenate(v))})
        if yi:
            y = np.concatenate(yi)
            for k, v in india.items():
                rows.append({"evaluated_on": "India chips in test split", "method": k, **scores(y, np.concatenate(v))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "flood_model_comparison.csv", index=False)
    print(res.round(3).to_string(index=False))
