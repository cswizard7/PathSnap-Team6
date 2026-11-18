"""Step 3b - Epoch/iteration-wise training histories (only for models that have them).

* HistGradientBoosting: log loss after every boosting iteration (staged predictions)
* MLP: log loss after every epoch (partial_fit loop)

Training-partition trips are split by trip into 80 % fit / 20 % validation
(GroupShuffleSplit), so the validation curve is on unseen trips.
Other models (LR, NB, KNN, DT, RF) do not produce epoch-wise histories, so no
curves are drawn for them.

Run:  python scripts/03b_learning_curves.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap import config, training
from pathsnap.features import FEATURES

OUT = config.RESULTS_DIR / "models"


def main():
    tr = training.load_training_rows()
    X, y, g = tr[FEATURES].values, tr.y.values, tr.trip.values
    i_fit, i_val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED).split(X, y, g))
    Xf, yf, Xv, yv = X[i_fit], y[i_fit], X[i_val], y[i_val]
    rng = np.random.default_rng(config.RANDOM_SEED)
    sub = rng.choice(len(Xf), min(200_000, len(Xf)), replace=False)   # train-loss estimate sample

    # ---- HistGradientBoosting ----
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                         early_stopping=False, random_state=config.RANDOM_SEED).fit(Xf, yf)
    tr_l = [log_loss(yf[sub], p[:, 1], labels=[0, 1]) for p in hgb.staged_predict_proba(Xf[sub])]
    va_l = [log_loss(yv, p[:, 1], labels=[0, 1]) for p in hgb.staged_predict_proba(Xv)]
    h1 = pd.DataFrame({"iteration": np.arange(1, len(tr_l) + 1), "train_log_loss": tr_l, "val_log_loss": va_l})
    h1.to_csv(OUT / "history_HistGradientBoosting.csv", index=False)

    # ---- MLP ----
    sc = StandardScaler().fit(Xf)
    Xfs, Xvs = sc.transform(Xf), sc.transform(Xv)
    msub = rng.choice(len(Xfs), min(300_000, len(Xfs)), replace=False)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                        learning_rate_init=1e-3, random_state=config.RANDOM_SEED)
    rows = []
    for ep in range(1, 31):
        idx = rng.permutation(msub)
        mlp.partial_fit(Xfs[idx], yf[idx], classes=[0, 1])
        rows.append({"epoch": ep,
                     "train_log_loss": log_loss(yf[sub], mlp.predict_proba(Xfs[sub])[:, 1], labels=[0, 1]),
                     "val_log_loss": log_loss(yv, mlp.predict_proba(Xvs)[:, 1], labels=[0, 1])})
        print(rows[-1], flush=True)
    h2 = pd.DataFrame(rows)
    h2.to_csv(OUT / "history_MLP.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h1.iteration, h1.train_log_loss, label="train (fit trips)")
    ax[0].plot(h1.iteration, h1.val_log_loss, label="validation (held-out trips)")
    ax[0].set(title="HistGradientBoosting: log loss per boosting iteration", xlabel="iteration", ylabel="log loss")
    ax[1].plot(h2.epoch, h2.train_log_loss, label="train (fit trips)")
    ax[1].plot(h2.epoch, h2.val_log_loss, label="validation (held-out trips)")
    ax[1].set(title="MLP (64,32): log loss per epoch", xlabel="epoch", ylabel="log loss")
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "training_histories.png", dpi=120)


if __name__ == "__main__":
    main()
