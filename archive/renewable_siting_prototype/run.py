"""One command: satellite layers -> features -> economics -> AI -> dashboard data.
    python run.py
"""
import json
from pathlib import Path

import numpy as np

import ml
from features import build_table
from model import ASSUMPTIONS, evaluate, explain
from region import REGION, CELL, grid_shape

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent.parent / "dashboard"
OUT.mkdir(exist_ok=True)

df = build_table()
res = evaluate(df)
res, ml_meta = ml.apply(res)
res.to_csv(DATA / "results.csv", index=False)

med = res.median(numeric_only=True)
med["years"] = df.attrs["ghi_years"]

# compact payload for the dashboard
cols = {
    "lat": 2, "lon": 2, "ghi": 2, "ws50": 2, "slope_deg": 1, "grid_km": 1, "elev_m": 0,
    "lc_tree": 2, "lc_crop": 2, "lc_built": 2, "lc_shrub": 2, "lc_grass": 2, "lc_bare": 2, "lc_water": 2,
    "dev_ha": 0,
    "solar_cuf": 3, "solar_cuf_p90": 3, "solar_lcoe": 2, "solar_npv_p90_cr": 1, "solar_npv_cr": 1,
    "solar_irr": 3, "solar_payback_yr": 1, "solar_capex_total_cr": 0, "solar_final": 1,
    "wind_v_hub": 2, "wind_cf": 3, "wind_lcoe": 2, "wind_npv_p90_cr": 1, "wind_npv_cr": 1,
    "wind_irr": 3, "wind_payback_yr": 1, "wind_capex_total_cr": 0, "wind_final": 1,
    "solar_likeness": 3, "wind_likeness": 3,
}
payload = {k: [None if (isinstance(v, float) and np.isnan(v)) else v
               for v in res[k].round(r).tolist()] for k, r in cols.items()}
for k in ("solar_excluded", "wind_excluded"):
    payload[k] = res[k].tolist()
for k in ("solar_gem", "wind_gem"):
    payload[k] = res[k].astype(bool).astype(int).tolist()

top = {}
for tech in ("solar", "wind"):
    ok = res[res[f"{tech}_excluded"] == ""].sort_values([f"{tech}_final", f"{tech}_npv_p90_cr"], ascending=False).head(10)
    top[tech] = [{"i": int(i), "why": explain(r, tech, med)} for i, r in ok.iterrows()]

meta = {
    "region": REGION, "cell": CELL, "shape": grid_shape(), "assumptions": ASSUMPTIONS,
    "ghi_years": int(df.attrs["ghi_years"]), "grid_source": df.attrs["grid_source"], "ml": ml_meta,
    "counts": {t: int((res[f"{t}_excluded"] == "").sum()) for t in ("solar", "wind")},
}
js = json.dumps({"meta": meta, "cells": payload, "top": top}, separators=(",", ":"), default=float)
(OUT / "data.js").write_text("window.ORBITWISE=" + js + ";")
print(f"dashboard/data.js written ({len(js) / 1e6:.2f} MB)")
for tech in ("solar", "wind"):
    print(f"\nTop {tech}:")
    for t in top[tech][:5]:
        r = res.loc[t["i"]]
        print(f"  {r.lat:.2f}N {r.lon:.2f}E  score {r[f'{tech}_final']:.0f}  LCOE Rs{r[f'{tech}_lcoe']:.2f}  "
              f"P90-NPV Rs{r[f'{tech}_npv_p90_cr']:.1f} cr")
