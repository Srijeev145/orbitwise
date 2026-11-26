"""
OrbitWise / Track A - data acquisition.

Pulls three satellite-derived datasets for a study region and caches them in data/:
  1. ESA WorldCover 2021 (10 m, from Sentinel-1 + Sentinel-2)   -> land use / exclusions
  2. NASA POWER (CERES + MERRA-2 satellite/reanalysis)          -> solar irradiance, wind, temperature
  3. SRTM 90 m via OpenTopoData (Space Shuttle radar)           -> elevation / slope

Usage:
    python fetch.py landcover
    python fetch.py climate
    python fetch.py terrain      # needs landcover first (skips sea cells)
    python fetch.py all
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

from region import REGION, cell_centers

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
CA = "/root/.ccr/ca-bundle.crt" if Path("/root/.ccr/ca-bundle.crt").exists() else None


# --------------------------------------------------------------------------- WorldCover
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
LC_RES = 0.001  # degrees per pixel in our mosaic (~110 m)


def fetch_landcover():
    import rasterio
    from rasterio.enums import Resampling

    s, n, w, e = REGION["south"], REGION["north"], REGION["west"], REGION["east"]
    H, W = round((n - s) / LC_RES), round((e - w) / LC_RES)
    mosaic = np.zeros((H, W), dtype=np.uint8)  # 0 = no data
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    if CA:
        env["GDAL_CURL_CA_BUNDLE"] = CA

    lat0s = range(int(np.floor(s / 3) * 3), int(np.ceil(n / 3) * 3), 3)
    lon0s = range(int(np.floor(w / 3) * 3), int(np.ceil(e / 3) * 3), 3)
    with rasterio.Env(**env):
        for la in lat0s:
            for lo in lon0s:
                tile = f"N{la:02d}E{lo:03d}"
                try:
                    with rasterio.open("/vsicurl/" + WC_URL.format(tile=tile)) as d:
                        px = round(3 / LC_RES)
                        a = d.read(1, out_shape=(px, px), resampling=Resampling.mode)
                except Exception as ex:  # ocean-only tiles do not exist
                    print(f"  {tile}: skipped ({type(ex).__name__})")
                    continue
                # tile covers lat [la, la+3], lon [lo, lo+3]; row 0 = north
                r0 = round((n - (la + 3)) / LC_RES)
                c0 = round((lo - w) / LC_RES)
                rs, cs = max(r0, 0), max(c0, 0)
                re_, ce = min(r0 + px, H), min(c0 + px, W)
                if rs >= re_ or cs >= ce:
                    continue
                mosaic[rs:re_, cs:ce] = a[rs - r0:re_ - r0, cs - c0:ce - c0]
                print(f"  {tile}: ok")
    np.save(DATA / "landcover_mosaic.npy", mosaic)
    print("saved landcover_mosaic.npy", mosaic.shape)


# --------------------------------------------------------------------------- NASA POWER
POWER = "https://power.larc.nasa.gov/api/temporal"


def _power_regional(kind, params, extra=""):
    url = (f"{POWER}/{kind}/regional?parameters={','.join(params)}&community=RE"
           f"&latitude-min={REGION['south']}&latitude-max={REGION['north']}"
           f"&longitude-min={REGION['west']}&longitude-max={REGION['east']}&format=JSON{extra}")
    for attempt in range(4):
        r = requests.get(url, timeout=300, verify=CA or True)
        if r.ok:
            return r.json()
        print("  POWER retry", r.status_code, r.text[:200])
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def fetch_climate():
    # the regional endpoint accepts one parameter per request
    clim = {}
    for p in ["ALLSKY_SFC_SW_DWN", "WS50M", "WS10M", "T2M"]:
        clim[p] = _power_regional("climatology", [p])
        print(f"  {p}: {len(clim[p]['features'])} nodes")
    json.dump(clim, open(DATA / "power_climatology.json", "w"))
    print("saved power_climatology.json")
    # year-by-year irradiance -> interannual variability (P50 / P90 bankability)
    yearly = _power_regional("monthly", ["ALLSKY_SFC_SW_DWN"], "&start=2005&end=2024")
    json.dump(yearly, open(DATA / "power_yearly.json", "w"))
    print("saved power_yearly.json")


# --------------------------------------------------------------------------- SRTM terrain
TOPO = "https://api.opentopodata.org/v1/srtm90m"
D = 0.01  # ~1.1 km offset for slope estimation


def fetch_terrain():
    lats, lons = cell_centers()
    land = np.ones(lats.shape, bool)
    lc_path = DATA / "landcover_mosaic.npy"
    if lc_path.exists():
        from features import cell_class_fractions
        fr = cell_class_fractions(np.load(lc_path))
        land = (fr[80] + fr[0]) < 0.95  # skip open water / no-data cells

    pts = []
    for la, lo in zip(lats[land], lons[land]):
        pts += [(la, lo), (la + D, lo), (la - D, lo), (la, lo + D), (la, lo - D)]
    out_path = DATA / "terrain_points.json"
    done = json.load(open(out_path)) if out_path.exists() else {}
    todo = [p for p in pts if f"{p[0]:.4f},{p[1]:.4f}" not in done]
    print(f"terrain: {len(pts)} points, {len(todo)} to fetch")
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        locs = "|".join(f"{a:.4f},{b:.4f}" for a, b in chunk)
        for attempt in range(5):
            r = requests.post(TOPO, data={"locations": locs}, timeout=60, verify=CA or True)
            if r.ok:
                break
            time.sleep(2 + 2 * attempt)
        r.raise_for_status()
        for p, res in zip(chunk, r.json()["results"]):
            done[f"{p[0]:.4f},{p[1]:.4f}"] = res["elevation"]
        if (i // 100) % 10 == 0:
            json.dump(done, open(out_path, "w"))
            print(f"  {i + len(chunk)}/{len(todo)}")
        time.sleep(1.05)  # public API limit: 1 request / second
    json.dump(done, open(out_path, "w"))
    print("saved terrain_points.json")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("landcover", "all"):
        fetch_landcover()
    if what in ("climate", "all"):
        fetch_climate()
    if what in ("terrain", "all"):
        fetch_terrain()
