"""Step 3c - Consolidated model comparison table + plots (from saved step-3 results).

Outputs (results/models/): model_comparison.csv, cv_macro_f1_comparison.png,
student_test_comparison.png, log_loss_comparison.png, efficiency.png,
confusion_matrices.png

Run:  python scripts/03c_compare_models.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from pathsnap import config, metrics, training
from pathsnap.models import MODEL_SPECS

OUT = config.RESULTS_DIR / "models"


def main():
    names = [n for n in MODEL_SPECS if (OUT / f"summary_{n}.json").exists()]
    summ = pd.DataFrame([json.load(open(OUT / f"summary_{n}.json")) for n in names])
    folds = pd.concat([pd.read_csv(OUT / f"cv_folds_{n}.csv") for n in names])
    base = json.load(open(OUT / "baseline_nearest_segment.json"))
    summ.to_csv(OUT / "model_comparison.csv", index=False)
    cols = ["model", "family", "cv_macro_f1_mean", "cv_macro_f1_std", "train_macro_f1", "test_top1",
            "test_macro_f1", "test_log_loss", "test_route_continuity", "fit_seconds", "inference_ms_per_1k_obs"]
    print(summ[cols].round(4).to_string(index=False))
    print("nearest-segment baseline:", {k: round(v, 4) for k, v in base.items()})

    order = summ.sort_values("cv_macro_f1_mean").model.tolist()
    s = summ.set_index("model").loc[order]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].barh(order, s.cv_macro_f1_mean, xerr=s.cv_macro_f1_std, capsize=3, color="C0")
    lo = s.cv_macro_f1_mean.min() - 0.03
    ax[0].set_xlim(lo, 1.0)
    for i, v in enumerate(s.cv_macro_f1_mean):
        ax[0].text(0.998, i, f"{v:.4f}", va="center", ha="right", fontsize=8)
    ax[0].set(title="5-fold GroupKFold CV Macro-F1 (mean +/- std)", xlabel="Macro-F1")
    data = [folds[folds.model == n].macro_f1.values for n in order]
    try:                                   # matplotlib >= 3.10
        ax[1].boxplot(data, orientation="horizontal")
    except TypeError:                      # older matplotlib (e.g. on Kaggle)
        ax[1].boxplot(data, vert=False)
    ax[1].set_yticks(range(1, len(order) + 1), order)
    ax[1].set(title="CV Macro-F1 per fold (stability)", xlabel="Macro-F1")
    fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1_comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(order))
    ax.bar(x - .2, s.test_top1, .4, label="Top-1 accuracy")
    ax.bar(x + .2, s.test_macro_f1, .4, label="Macro-F1")
    ax.axhline(base["top1_acc"], ls="--", c="gray", label=f"nearest-segment baseline top-1 ({base['top1_acc']:.3f})")
    ax.set_xticks(x, order, rotation=20); ax.set_ylim(0.5, 1.0)
    ax.set(title="Student-testing partition (10 trips): classifier-only", ylabel="score"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "student_test_comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(order, s.test_log_loss, color="C3")
    ax.set_yscale("log"); ax.set(title="Student-test log loss (lower is better, log scale)", ylabel="log loss")
    for i, v in enumerate(s.test_log_loss):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(OUT / "log_loss_comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(order, s.fit_seconds); ax[0].set(title="Training time (final fit)", ylabel="seconds")
    ax[1].bar(order, s.inference_ms_per_1k_obs); ax[1].set(title="Inference time", ylabel="ms per 1000 GPS obs")
    ax[1].set_yscale("log")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(OUT / "efficiency.png", dpi=120); plt.close(fig)

    te, _ = training.load_student_test()
    fig, axes = plt.subplots(2, int(np.ceil(len(names) / 2)), figsize=(4 * int(np.ceil(len(names) / 2)), 7))
    for a, n in zip(axes.ravel(), names):
        p = np.load(OUT / f"student_test_proba_{n}.npy")
        key = te.trip + "_" + te.obs.astype(str)
        mask = metrics.top1_from_scores(key.values, p)
        cm = confusion_matrix(te.y.values, mask.astype(int), labels=[0, 1])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks([0, 1], ["not chosen", "chosen (top-1)"]); a.set_yticks([0, 1], ["wrong arc", "true arc"])
        a.set_title(n, fontsize=10)
    for a in axes.ravel()[len(names):]:
        a.axis("off")
    fig.suptitle("Candidate-level confusion matrices (student test)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
