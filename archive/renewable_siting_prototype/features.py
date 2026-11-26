"""Turn the raw satellite layers into one feature table: one row per 0.1-degree candidate cell."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

from region import REGION, CELL, grid_shape, cell_centers

DATA = Path(__file__).parent / "data"
LC_RES = 0.001
CLASSES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
LC_NAMES = {0: "nodata", 10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
            60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def cell_class_fractions(mosaic):
    """dict class -> 2-D array of the fraction of each analysis cell covered by that class."""
    k = round(CELL / LC_RES)
    nr, nc = grid_shape()
    blocks = mosaic[:nr * k, :nc * k].reshape(nr, k, nc, k)
    return {c: (blocks == c).mean(axis=(1, 3)) for c in CLASSES}


def town_distance_km(mosaic):
    """Distance from each cell centre to the nearest built-up cluster (>=25% built in a 1 km block).
    Proxy for road access, grid substations and electricity demand."""
    b = 10  # 10 px of 0.001 deg = ~1 km
    H, W = mosaic.shape
    built = (mosaic[:H // b * b, :W // b * b] == 50).reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    towns = built >= 0.25
    mid_lat = (REGION["south"] + REGION["north"]) / 2
    d = distance_transform_edt(~towns, sampling=(0.01 * 110.6, 0.01 * 111.3 * np.cos(np.radians(mid_lat))))
    k = round(CELL / 0.01)
    nr, nc = grid_shape()
    return d[k // 2::k, k // 2::k][:nr, :nc]


def _interp_power(feats, key="ANN"):
    pts, vals = [], []
    for f in feats:
        lon, lat = f["geometry"]["coordinates"][:2]
        v = list(f["properties"]["parameter"].values())[0]
        v = v[key] if isinstance(v, dict) else v
        if v is not None and v > -900:
            pts.append((lat, lon)); vals.append(v)
    lats, lons = cell_centers()
    pts, vals = np.array(pts), np.array(vals)
    lin = griddata(pts, vals, (lats, lons), method="linear")
    near = griddata(pts, vals, (lats, lons), method="nearest")
    return np.where(np.isnan(lin), near, lin)


def climate_layers():
    clim = json.load(open(DATA / "power_climatology.json"))
    out = {
        "ghi": _interp_power(clim["ALLSKY_SFC_SW_DWN"]["features"]),   # kWh/m2/day
        "ws50": _interp_power(clim["WS50M"]["features"]),              # m/s
        "ws10": _interp_power(clim["WS10M"]["features"]),
        "t2m": _interp_power(clim["T2M"]["features"]),                 # deg C
    }
    # interannual variability of irradiance -> P90 yield
    yearly = json.load(open(DATA / "power_yearly.json"))["features"]
    years = sorted({k[:4] for f in yearly for k in f["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
                    if k.endswith("13")})
    stack = np.stack([_interp_power(yearly, key=f"{y}13") for y in years])
    out["ghi_cv"] = stack.std(0) / stack.mean(0)
    out["ghi_years"] = len(years)
    return out


def terrain_layers():
    lats, lons = cell_centers()
    path = DATA / "terrain_points.json"
    slope = np.full(lats.shape, np.nan)
    elev = np.full(lats.shape, np.nan)
    if not path.exists():
        return elev, slope
    t = json.load(open(path))
    D = 0.01
    g = lambda a, b: t.get(f"{a:.4f},{b:.4f}")
    for idx in np.ndindex(lats.shape):
        la, lo = lats[idx], lons[idx]
        c, n, s, e, w = g(la, lo), g(la + D, lo), g(la - D, lo), g(la, lo + D), g(la, lo - D)
        if None in (c, n, s, e, w):
            continue
        dy = (n - s) / (2 * D * 110600)
        dx = (e - w) / (2 * D * 111300 * np.cos(np.radians(la)))
        slope[idx] = np.degrees(np.arctan(np.hypot(dx, dy)))
        elev[idx] = c
    return elev, slope


def build_table():
    mosaic = np.load(DATA / "landcover_mosaic.npy")
    fr = cell_class_fractions(mosaic)
    lats, lons = cell_centers()
    clim = climate_layers()
    elev, slope = terrain_layers()
    df = pd.DataFrame({
        "row": np.repeat(np.arange(lats.shape[0]), lats.shape[1]),
        "col": np.tile(np.arange(lats.shape[1]), lats.shape[0]),
        "lat": lats.ravel().round(3), "lon": lons.ravel().round(3),
        "ghi": clim["ghi"].ravel(), "ghi_cv": clim["ghi_cv"].ravel(),
        "ws50": clim["ws50"].ravel(), "ws10": clim["ws10"].ravel(), "t2m": clim["t2m"].ravel(),
        "elev_m": elev.ravel(), "slope_deg": slope.ravel(),
        "town_km": town_distance_km(mosaic).ravel(),
    })
    for c in CLASSES:
        if c not in (0, 70, 100):
            df[f"lc_{LC_NAMES[c]}"] = fr[c].ravel()
    df["lc_nodata"] = fr[0].ravel()
    df["grid_km"], df.attrs["grid_source"] = df.town_km, "distance to built-up cluster (proxy)"
    sub = substation_distance_km(df)
    if sub is not None:
        df["grid_km"], df.attrs["grid_source"] = sub, "distance to >=66 kV substation (OpenStreetMap)"
    df.attrs["ghi_years"] = clim["ghi_years"]
    return df


def substation_distance_km(df):
    path = DATA / "osm_substations.json"
    if not path.exists():
        return None
    from scipy.spatial import cKDTree
    pts = []
    for el in json.load(open(path))["elements"]:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat and lon:
            pts.append((lat * 110.6, lon * 111.3 * np.cos(np.radians(15))))
    if len(pts) < 5:
        return None
    d, _ = cKDTree(pts).query(np.c_[df.lat * 110.6, df.lon * 111.3 * np.cos(np.radians(15))])
    return d


if __name__ == "__main__":
    df = build_table()
    df.to_csv(DATA / "features.csv", index=False)
    print(df.describe().T.round(3).to_string())
