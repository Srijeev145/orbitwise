"""
Download Sentinel-1 RTC (radiometrically terrain-corrected) radar images from Microsoft Planetary Computer
for our two flood events, clipped to the study areas and resampled to ~55 m.

Output GeoTIFFs: 2 bands (VV, VH) in dB, EPSG:4326, in data/s1/.
"Before" images are the median of several pre-flood passes (removes radar speckle);
"after" images are single passes during the flood. Before/after always share the same orbit track,
so brightness differences come from the ground, not from viewing geometry.
"""
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

OUT = Path(__file__).resolve().parents[1] / "data_full" / "s1"
OUT.mkdir(parents=True, exist_ok=True)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
RES = 0.0005  # degrees (~55 m)

EVENTS = {
    # name: (bbox [W, S, E, N], track, {output: [dates]})
    "KA2019": ([74.5, 15.5, 76.0, 17.0], 63, {
        "before": ["2019-07-10", "2019-07-16", "2019-07-22"],
        "after_0803": ["2019-08-03"],
        "after_0821": ["2019-08-21"],
    }),
    "KL2018_t63": ([76.0, 9.0, 76.8, 10.5], 63, {
        "before": ["2018-04-28", "2018-05-10", "2018-05-22"],
        "after_0814": ["2018-08-14"],
    }),
    "KL2018_t165": ([76.0, 9.0, 76.8, 10.5], 165, {
        "before": ["2018-04-23", "2018-05-05", "2018-05-17"],
        "after_0821": ["2018-08-21"],
    }),
}


def items_for(bbox, track, date):
    r = requests.post(STAC, json={"collections": ["sentinel-1-rtc"], "bbox": bbox,
                                  "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
                                  "query": {"sat:relative_orbit": {"eq": track}}, "limit": 20}, timeout=60)
    r.raise_for_status()
    return r.json()["features"]


def fresh_token():
    """SAS tokens expire after ~15 min, so fetch a new one for every image."""
    return requests.get(TOKEN_URL, timeout=30).json()["token"]


def read_date(bbox, track, date, token=None):
    """Mosaic all frames of one pass onto our grid; returns (2, H, W) linear backscatter, NaN = no data.
    Reads only the study-area window, at reduced resolution (uses the COG overviews), then reprojects."""
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds as win_from_bounds
    W, S, E, N = bbox
    w, h = round((E - W) / RES), round((N - S) / RES)
    tr = from_bounds(W, S, E, N, w, h)
    out = np.full((2, h, w), np.nan, dtype="float32")
    feats = items_for(bbox, track, date)
    if not feats:
        raise RuntimeError(f"no scenes for {date} track {track}")
    for f in feats:
        token = fresh_token()
        for b, pol in enumerate(("vv", "vh")):
            with rasterio.open(f["assets"][pol]["href"] + "?" + token) as src:
                l, bt, r, t = transform_bounds("EPSG:4326", src.crs, W, S, E, N)
                win = win_from_bounds(l, bt, r, t, src.transform).intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height))
                k = 5  # 10 m -> 50 m
                shape = (max(1, int(win.height // k)), max(1, int(win.width // k)))
                a = src.read(1, window=win, out_shape=shape, resampling=Resampling.average).astype("float32")
                wt = src.window_transform(win) * rasterio.Affine.scale(win.width / shape[1], win.height / shape[0])
                a[a <= 0] = np.nan
                dst = np.full((h, w), np.nan, dtype="float32")
                reproject(a, dst, src_transform=wt, src_crs=src.crs, src_nodata=np.nan,
                          dst_transform=tr, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.bilinear)
            out[b] = np.where(np.isnan(out[b]), dst, out[b])
    return out, tr


def save(arr_db, tr, path):
    data = np.where(np.isnan(arr_db), -32768, np.round(arr_db * 100)).astype("int16")  # dB x 100
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=2,
                       dtype="int16", crs="EPSG:4326", transform=tr, nodata=-32768, compress="deflate") as d:
        d.write(data)
        d.set_band_description(1, "VV_dB_x100")
        d.set_band_description(2, "VH_dB_x100")


if __name__ == "__main__":
    token = None
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2"}
    if Path("/root/.ccr/ca-bundle.crt").exists():
        env["GDAL_CURL_CA_BUNDLE"] = "/root/.ccr/ca-bundle.crt"
    with rasterio.Env(**env):
        for name, (bbox, track, outputs) in EVENTS.items():
            for oname, dates in outputs.items():
                path = OUT / f"{name}_{oname}.tif"
                if path.exists():
                    print("exists", path.name); continue
                stack = []
                for d in dates:
                    lin, tr = read_date(bbox, track, d, token)
                    stack.append(10 * np.log10(lin))
                    print(f"  {name} {oname} {d}: {np.isfinite(lin[0]).mean() * 100:.0f}% of area covered")
                db = np.nanmedian(np.stack(stack), axis=0) if len(stack) > 1 else stack[0]
                save(db, tr, path)
                print("saved", path.name, f"| VH median {np.nanmedian(db[1]):.1f} dB")
