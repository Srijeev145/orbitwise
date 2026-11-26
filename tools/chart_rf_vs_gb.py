"""Chart for the Random Forest vs Gradient Boosting question, from the 5-model comparison
(archive/step_by_step_scripts/compare_models.py, 5-fold cross-validation on 8,892 real ESA events)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "archive/step_by_step_scripts/trackB_results/model_comparison.csv")
df = df[(df.features.str.startswith("B")) | (df.model.str.startswith("Baseline"))]
order = ["Baseline: Day-2 estimate", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
df = df.set_index("model").loc[order]
labels = ["Do nothing", "Logistic\nRegression", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"]
metrics = [("accuracy", "#257de0", "Accuracy"), ("high_recall", "#f47230", "High-risk events caught (recall)"),
           ("high_precision", "#00b894", "Alerts that were real (precision)")]
fig, ax = plt.subplots(figsize=(12, 5.8))
x, w = np.arange(len(order)), 0.26
for k, (m, c, lab) in enumerate(metrics):
    bars = ax.bar(x + (k - 1) * (w + 0.02), df[m] * 100, w, color=c, label=lab)
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=12, color="#191919")
ax.set_xticks(x, labels)
ax.set_ylim(0, 115)
ax.set_yticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#bbbbbb")
ax.tick_params(colors="#666666", labelsize=13, length=0)
ax.yaxis.grid(True, color="#e6e6e6"); ax.set_axisbelow(True)
ax.set_title("Five models on real ESA collision data (5-fold cross-validation)", loc="left", fontsize=17,
             color="#191919", fontweight="bold", pad=14)
ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=3)
fig.tight_layout()
out = ROOT / "results/charts"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "rf_vs_gb.png", dpi=200, bbox_inches="tight")
print("saved", out / "rf_vs_gb.png")
