"""
Builds data_sample/: small pieces of the real datasets that orbitwise.py runs on.
Only needed if you re-download the full data (see tools/). The pieces are already bundled.

    python tools/make_samples.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orbitwise import flood_features, worldcover_on_grid  # noqa: E402

FULL = ROOT / "data_full"
OUT = ROOT / "data_sample"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def sen1floods11_pixels():
    d = FULL / "sen1floods11"
    out = {}
    for split, n in (("train", 1000), ("valid", 1000), ("test", 3000), ("bolivia", 3000)):
        rows = pd.read_csv(d / "flood_handlabeled" / f"flood_{split}_data.csv", header=None)
        X, Y, O, IND = [], [], [], []
        for s1name, lab in rows.itertuples(index=False):
            with rasterio.open(d / "S1Hand" / s1name) as f:
                vv, vh = f.read().astype("float32")
            with rasterio.open(d / "LabelHand" / lab) as f:
                y = f.read(1).ravel()
            with rasterio.open(d / "S1OtsuLabelHand" / s1name.replace("S1Hand", "S1OtsuLabelHand")) as f:
                otsu = f.read(1).ravel()
            feats = flood_features(vv, vh)
            ok = np.flatnonzero(y >= 0)
            take = rng.choice(ok, min(len(ok), n), replace=False) if len(ok) else ok
            X.append(feats[take]); Y.append(y[take]); O.append(otsu[take])
            IND.append(np.full(len(take), s1name.startswith("India")))
        out[f"X_{split}"] = np.concatenate(X).astype("float16")
        out[f"y_{split}"] = np.concatenate(Y).astype("int8")
        out[f"otsu_{split}"] = np.concatenate(O).astype("int8")
        out[f"india_{split}"] = np.concatenate(IND)
        print(f"  {split}: {len(out[f'y_{split}']):,} pixels")
    np.savez_compressed(OUT / "flood_pixels.npz", **out)


def kerala():
    src = FULL / "s1"
    for name in ("KL2018_t165_before.tif", "KL2018_t165_after_0821.tif"):
        shutil.copy(src / name, OUT / name)
    with rasterio.open(OUT / "KL2018_t165_before.tif") as f:
        wc = worldcover_on_grid(f.transform, f.shape)
    np.savez_compressed(OUT / "KL2018_worldcover.npz", worldcover=wc)
    print("  Kerala radar + land use")


def kelvins():
    k = FULL / "kelvins"
    cols = ["event_id", "time_to_tca", "risk", "miss_distance", "relative_speed",
            "max_risk_estimate", "mahalanobis_distance", "c_object_type"]
    for name, has_final in (("train_data.csv", True), ("test_data.csv", False)):
        df = pd.read_csv(k / name, usecols=cols)
        early = df[df.time_to_tca >= 2]
        early = early.loc[early.groupby("event_id")["time_to_tca"].idxmin()]
        if has_final:
            final = df.loc[df.groupby("event_id")["time_to_tca"].idxmin(), ["event_id", "risk", "time_to_tca"]]
            early = early.merge(final.rename(columns={"risk": "final_risk", "time_to_tca": "final_days"}), on="event_id")
        out = OUT / ("kelvins_events_train.csv" if has_final else "kelvins_events_test.csv")
        early.to_csv(out, index=False)
        print(f"  {out.name}: {len(early):,} events")


def tles():
    for f in (FULL / "tle").glob("*.txt"):
        shutil.copy(f, OUT / f"tle_{f.name}")
    print("  orbit files")


if __name__ == "__main__":
    sen1floods11_pixels(); kerala(); kelvins(); tles()
