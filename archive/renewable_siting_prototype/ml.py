"""
OrbitWise / Track A - the AI layer: "learn where developers actually build".

Physics + finance (model.py) tells us what *should* pay. But real projects also depend on things
no formula captures well: land-lease willingness, local permitting, evacuation capacity.
Existing plants reveal those preferences. We train a Random Forest (same family as our
debris-risk classifier) as a positive-unlabelled learner:

    positives  = cells that already host a solar / wind plant (OpenStreetMap)
    unlabelled = all other eligible cells

Validation uses SPATIAL block cross-validation (1-degree blocks) so the model cannot cheat by
memorising neighbouring cells. The output "developer-likeness" is blended with the economic
score, and cells that score high on both but have no plant yet are flagged as "hidden gems".
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from region import REGION, CELL

DATA = Path(__file__).parent / "data"
FEATURES = ["ghi", "ghi_cv", "t2m", "ws50", "slope_deg", "elev_m", "town_km",
            "lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water"]


def load_plants():
    path = DATA / "osm_plants.json"
    if not path.exists():
        return None
    rows = []
    for el in json.load(open(path))["elements"]:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        t = el.get("tags", {})
        src = t.get("plant:source") or t.get("generator:source") or ""
        tech = "solar" if "solar" in src else "wind" if "wind" in src else None
        if lat and lon and tech:
            rows.append((lat, lon, tech, t.get("name", "")))
    return pd.DataFrame(rows, columns=["lat", "lon", "tech", "name"])


def label_cells(res, plants, tech):
    p = plants[plants.tech == tech]
    r = ((REGION["north"] - p.lat) / CELL).astype(int)
    c = ((p.lon - REGION["west"]) / CELL).astype(int)
    hit = set(zip(r, c))
    return np.array([(a, b) in hit for a, b in zip(res.row, res.col)])


def train(res, tech):
    plants = load_plants()
    if plants is None or (plants.tech == tech).sum() < 8:
        print(f"[ml] {tech}: no / too few OSM plants - run fetch_osm.py on a laptop first")
        return None
    elig = res[f"{tech}_excluded"] == ""
    X = res.loc[elig, FEATURES].fillna(res[FEATURES].median())
    y = label_cells(res, plants, tech)[elig.to_numpy()]
    groups = (np.floor(res.loc[elig, "lat"]).astype(int) * 1000 + np.floor(res.loc[elig, "lon"]).astype(int))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                   random_state=0, n_jobs=-1).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(y, oof) if 0 < y.sum() < len(y) else float("nan")
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                               random_state=0, n_jobs=-1).fit(X, y)
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"[ml] {tech}: {int(y.sum())} plant cells / {len(y)} eligible, spatial-CV ROC-AUC = {auc:.3f}")
    print("     top drivers:", ", ".join(f"{k} {v:.2f}" for k, v in imp.head(5).items()))
    like = pd.Series(0.0, index=res.index)
    like[elig] = m.predict_proba(X)[:, 1]
    return {"likeness": like, "has_plant": pd.Series(label_cells(res, plants, tech), index=res.index),
            "auc": auc, "importance": imp.round(3).to_dict(), "n_pos": int(y.sum())}


def apply(res):
    meta = {}
    for tech in ("solar", "wind"):
        out = train(res, tech)
        if out is None:
            res[f"{tech}_likeness"] = np.nan
            res[f"{tech}_final"] = res[f"{tech}_score"]
            res[f"{tech}_gem"] = False
            continue
        like_pct = out["likeness"].rank(pct=True) * 100
        res[f"{tech}_likeness"] = out["likeness"].round(3)
        res[f"{tech}_final"] = np.where(res[f"{tech}_excluded"] == "",
                                        (0.65 * res[f"{tech}_score"] + 0.35 * like_pct).round(1), 0)
        res[f"{tech}_has_plant"] = out["has_plant"]
        res[f"{tech}_gem"] = (res[f"{tech}_final"] >= 85) & ~out["has_plant"]
        meta[tech] = {k: out[k] for k in ("auc", "importance", "n_pos")}
    return res, meta
