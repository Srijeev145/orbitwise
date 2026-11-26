"""
Turn before/after Sentinel-1 images into flood maps, flooded-cropland numbers and slide images.

Two methods, side by side:
  Baseline (UN-SPIDER change detection): flood where after/before VH ratio > 1.25 (pixel ~25% darker in dB)
  AI (Gradient Boosting from flood_model.py): water probability on the after AND before image;
      flood = water after, but not water before (removes rivers, lakes, permanent paddy water)
Both: remove patches smaller than 8 pixels. Using the same orbit track before/after means terrain shadow
appears in both images and cancels out.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.ndimage import label, median_filter

from flood_model import features

D = Path(__file__).parent / "data"
S1 = D / "s1"
OUT = Path(__file__).parent / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EVENTS = {
    "Kerala 21 Aug 2018": ("KL2018_t165_before.tif", "KL2018_t165_after_0821.tif"),
    "Karnataka 3 Aug 2019": ("KA2019_before.tif", "KA2019_after_0803.tif"),
    "Karnataka 21 Aug 2019": ("KA2019_before.tif", "KA2019_after_0821.tif"),
}
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")


def read(path):
    with rasterio.open(path) as d:
        a = d.read().astype("float32")
        a[a == -32768] = np.nan
        return a / 100, d.transform, d.crs, d.shape


def worldcover_on_grid(transform, shape):
    """ESA WorldCover land-use classes resampled onto the radar grid (mode of 10 m pixels)."""
    out = np.zeros(shape, "uint8")
    W, N = transform.c, transform.f
    E, S = W + transform.a * shape[1], N + transform.e * shape[0]
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    if Path("/root/.ccr/ca-bundle.crt").exists():
        env["GDAL_CURL_CA_BUNDLE"] = "/root/.ccr/ca-bundle.crt"
    with rasterio.Env(**env):
        for la in range(int(S // 3 * 3), int(np.ceil(N / 3) * 3), 3):
            for lo in range(int(W // 3 * 3), int(np.ceil(E / 3) * 3), 3):
                try:
                    with rasterio.open("/vsicurl/" + WC_URL.format(tile=f"N{la:02d}E{lo:03d}")) as src, \
                         WarpedVRT(src, crs="EPSG:4326", transform=transform, width=shape[1], height=shape[0],
                                   resampling=Resampling.mode, nodata=0) as v:
                        a = v.read(1)
                    out = np.where(out == 0, a, out)
                except Exception as e:
                    print("  worldcover tile skipped", la, lo, type(e).__name__)
    return out


def drop_specks(mask, min_px=8):
    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def km2_per_px(transform, shape):
    lat = transform.f + transform.e * shape[0] / 2
    return abs(transform.a * transform.e) * 110.57 * 111.32 * np.cos(np.radians(lat))


def ai_water(img, bundle):
    f = features(img[0], img[1])
    p = bundle["model"].predict_proba(f)[:, 1].reshape(img.shape[1:])
    return p >= bundle["cutoff"]


if __name__ == "__main__":
    bundle = joblib.load(OUT / "flood_model.joblib")
    stats, wc_cache = {}, {}
    for name, (bf, af) in EVENTS.items():
        before, tr, crs, shape = read(S1 / bf)
        after, *_ = read(S1 / af)
        valid = np.isfinite(before[1]) & np.isfinite(after[1])
        b_vh, a_vh = median_filter(np.nan_to_num(before[1], nan=0), 3), median_filter(np.nan_to_num(after[1], nan=0), 3)

        key = bf
        if key not in wc_cache:
            wc_cache[key] = worldcover_on_grid(tr, shape)
        wc = wc_cache[key]
        valid &= ~np.isin(wc, [0, 80])  # drop sea and permanent water bodies (ESA WorldCover)

        base = valid & (a_vh / np.where(b_vh == 0, np.nan, b_vh) > 1.25)
        base = drop_specks(base)
        ai = valid & ai_water(after, bundle) & ~ai_water(before, bundle)
        ai = drop_specks(ai)

        px = km2_per_px(tr, shape)
        s = {}
        for m, mask in (("baseline", base), ("ai", ai)):
            s[m] = {"flooded_km2": float(mask.sum() * px),
                    "cropland_km2": float((mask & (wc == 40)).sum() * px),
                    "builtup_km2": float((mask & (wc == 50)).sum() * px),
                    "wetland_or_grass_km2": float((mask & np.isin(wc, [30, 90])).sum() * px)}
        s["coverage_pct"] = float(valid.mean() * 100)
        # Alappuzha + Kottayam lowlands box, to compare with IWMI's 14 Aug 2018 numbers (241.94 + 164.82 km2)
        if name.startswith("Kerala"):
            rows = np.arange(shape[0]) * tr.e + tr.f
            cols = np.arange(shape[1]) * tr.a + tr.c
            box = ((rows[:, None] >= 9.05) & (rows[:, None] <= 9.95) & (cols[None, :] >= 76.25) & (cols[None, :] <= 76.8))
            s["alappuzha_kottayam_box_km2"] = {m: float((mk & box).sum() * px) for m, mk in (("baseline", base), ("ai", ai))}
        stats[name] = s
        print(name, json.dumps(s, indent=None))

        fig, ax = plt.subplots(1, 3, figsize=(15, 6.5), constrained_layout=True)
        ext = [tr.c, tr.c + tr.a * shape[1], tr.f + tr.e * shape[0], tr.f]
        for a_, img, t in ((ax[0], before[1], "Before (VH, dB)"), (ax[1], after[1], "After (VH, dB)")):
            a_.imshow(img, cmap="gray", vmin=-25, vmax=-5, extent=ext); a_.set_title(t)
        ax[2].imshow(after[1], cmap="gray", vmin=-25, vmax=-5, extent=ext)
        ax[2].imshow(np.ma.masked_where(~ai, ai), cmap="winter", alpha=0.85, extent=ext, interpolation="nearest")
        ax[2].set_title(f"AI flood map: {s['ai']['flooded_km2']:.0f} km² new water")
        for a_ in ax:
            a_.set_xlabel("Longitude °E"); a_.set_ylabel("Latitude °N")
        fig.suptitle(f"{name}: Sentinel-1 radar", fontsize=14)
        fig.savefig(FIG / f"{name.replace(' ', '_')}.png", dpi=110)
        plt.close(fig)
    json.dump(stats, open(OUT / "flood_stats.json", "w"), indent=2)
