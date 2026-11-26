"""
Collision-risk prediction on REAL ESA conjunction data (Kelvins Collision Avoidance Challenge).

Question: using only the warnings (CDMs) available >= 2 days before closest approach,
can we predict the FINAL risk category (Low / Medium / High)?

Compares: naive "Day-2 estimate" baseline, Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting - on two feature sets:
  A) risk only at Day 4, Day 3, Day 2   (the Phase II-style setup)
  B) A + geometry / uncertainty / object features
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).parent / "data" / "kelvins"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

HIGH, MED = -6.0, -8.0          # log10 Pc: >= 1e-6 High (ESA Kelvins threshold), >= 1e-8 Medium
LABELS = ["Low", "Medium", "High"]


def category(r):
    return np.where(r >= HIGH, 2, np.where(r >= MED, 1, 0))


def build_events(df):
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    rows = []
    for eid, g in df.groupby("event_id", sort=False):
        final = g[g.time_to_tca < 2]
        early = g[g.time_to_tca >= 2]
        if final.empty or early.empty:
            continue
        r = {"event_id": eid, "final_risk": final.iloc[-1].risk}
        for d in (4, 3, 2):  # latest risk known at day d before closest approach
            known = g[g.time_to_tca >= d]
            r[f"risk_day{d}"] = known.iloc[-1].risk if len(known) else np.nan
        last = early.iloc[-1]
        r.update({
            "risk_trend": last.risk - early.iloc[0].risk,
            "n_cdms": len(early),
            "miss_distance": last.miss_distance,
            "relative_speed": last.relative_speed,
            "mahalanobis": last.mahalanobis_distance,
            "t_sigma_r": last.t_sigma_r, "t_sigma_t": last.t_sigma_t, "t_sigma_n": last.t_sigma_n,
            "c_sigma_r": last.c_sigma_r, "c_sigma_t": last.c_sigma_t, "c_sigma_n": last.c_sigma_n,
            "max_risk_estimate": last.max_risk_estimate,
            "c_debris": int(last.c_object_type == "DEBRIS"),
            "c_payload": int(last.c_object_type == "PAYLOAD"),
            "c_rocket": int(last.c_object_type == "ROCKET BODY"),
            "F10": last.F10, "AP": last.AP,
        })
        rows.append(r)
    ev = pd.DataFrame(rows)
    ev["y"] = category(ev.final_risk)
    return ev


RISK_ONLY = ["risk_day4", "risk_day3", "risk_day2"]
RICH = RISK_ONLY + ["risk_trend", "n_cdms", "miss_distance", "relative_speed", "mahalanobis",
                    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
                    "max_risk_estimate", "c_debris", "c_payload", "c_rocket", "F10", "AP"]


def models():
    return {
        "Logistic Regression": make_pipeline(SimpleImputer(), StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision Tree": make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                                              random_state=42)),
        "Random Forest": make_pipeline(SimpleImputer(), RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=3, learning_rate=0.05, class_weight="balanced", random_state=42),
    }


def scores(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "macro_F1": f1_score(y, p, average="macro"),
        "high_recall": recall_score(y, p, labels=[2], average="macro", zero_division=0),
        "high_precision": precision_score(y, p, labels=[2], average="macro", zero_division=0),
    }


if __name__ == "__main__":
    cols = ["event_id", "time_to_tca", "risk", "miss_distance", "relative_speed", "mahalanobis_distance",
            "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
            "max_risk_estimate", "c_object_type", "F10", "AP"]
    ev = build_events(pd.read_csv(DATA / "train_data.csv", usecols=cols))
    ev.to_csv(OUT / "events.csv", index=False)
    print(f"{len(ev)} labelled events | class counts:",
          dict(zip(LABELS, np.bincount(ev.y, minlength=3).tolist())))

    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(ev, ev.y))
    print("5-fold stratified cross-validation, split by event; scores are averaged over folds\n")

    rows, cms = [], {}

    def run(name, feats, make, key=None):
        acc, cm = [], np.zeros((3, 3), int)
        for tr_i, te_i in folds:
            tr, te = ev.iloc[tr_i], ev.iloc[te_i]
            p = category(te.risk_day2.fillna(-30)) if make is None else make().fit(tr[feats], tr.y).predict(te[feats])
            acc.append(scores(te.y, p)); cm += confusion_matrix(te.y, p, labels=[0, 1, 2])
        if key:
            cms[key] = cm
        return pd.DataFrame(acc).mean().to_dict()

    rows.append({"features": "-", "model": "Baseline: Day-2 estimate", **run("b", None, None, "Baseline")})
    for fname, feats in (("A: Day 4/3/2 risk", RISK_ONLY), ("B: + geometry/uncertainty", RICH)):
        for mname in models():
            key = mname if fname.startswith("B") and mname in ("Random Forest", "Gradient Boosting") else None
            rows.append({"features": fname, "model": mname,
                         **run(mname, feats, lambda m=mname: models()[m], key)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_comparison.csv", index=False)
    print(res.round(3).to_string(index=False))

    # training-size comparison: RF vs GB, feature set B, fixed test fold
    print("\nTraining-size comparison (feature set B, fold-1 test set):")
    tr_i, te_i = folds[0]
    tr, te = ev.iloc[tr_i], ev.iloc[te_i]
    lc = []
    for n in (160, 400, 800, 2000, len(tr)):
        sub = tr if n >= len(tr) else train_test_split(tr, train_size=n, stratify=tr.y, random_state=42)[0]
        for mname in ("Random Forest", "Gradient Boosting"):
            m = models()[mname].fit(sub[RICH], sub.y)
            lc.append({"train_events": len(sub), "model": mname, **scores(te.y, m.predict(te[RICH]))})
    lc = pd.DataFrame(lc)
    lc.to_csv(OUT / "learning_curve.csv", index=False)
    print(lc.round(3).to_string(index=False))

    print("\nConfusion matrices summed over 5 folds (rows = true Low/Med/High, cols = predicted):")
    for k, v in cms.items():
        print(k); print(v)
