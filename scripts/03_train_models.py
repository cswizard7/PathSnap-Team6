"""Step 3 - Train and compare candidate classifiers.

Protocol
  * Fitting / CV: 80 training trips (minus label-unreliable trips found in EDA),
    5-fold GroupKFold by trip (no trip is split across folds).
  * Development comparison: 10 student-testing trips (every observation counts).
  * Evaluator-testing data is never touched.

Outputs (results/models/):
  cv_folds.csv, model_comparison.csv, confusion_<model>.png, comparison plots,
  student_test_proba_<model>.npy (for the HMM experiments)

Run:  python scripts/03_train_models.py                 (all models)
      python scripts/03_train_models.py --models MLP KNN (subset)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config, training
from pathsnap.features import FEATURES
from pathsnap.models import MODEL_SPECS

OUT = config.RESULTS_DIR / "models"
OUT.mkdir(parents=True, exist_ok=True)


def run_model(name, tr, te, te_obs, arcs_te):
    spec = MODEL_SPECS[name]
    X, y, groups = tr[FEATURES].values, tr.y.values, tr.trip.values
    folds = []
    for k, (i_tr, i_va) in enumerate(GroupKFold(n_splits=config.N_FOLDS).split(X, y, groups)):
        Xs, ys = training.subsample(X[i_tr], y[i_tr], spec["max_rows"])
        m = spec["factory"]()
        t0 = time.perf_counter()
        m.fit(Xs, ys)
        fit_s = time.perf_counter() - t0
        p, _ = training.timed_predict(m, X[i_va])
        r = training.evaluate_rows(tr.iloc[i_va], p)
        r.update(model=name, fold=k, fit_seconds=fit_s, n_train_rows=len(Xs))
        folds.append(r)
        print(f"  {name} fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f}", flush=True)
    # final fit on all training rows
    Xs, ys = training.subsample(X, y, spec["max_rows"])
    m = spec["factory"]()
    t0 = time.perf_counter()
    m.fit(Xs, ys)
    fit_s = time.perf_counter() - t0
    # training-set (in-sample) metrics, for over/under-fitting diagnosis
    p_in, _ = training.timed_predict(m, X)
    r_in = training.evaluate_rows(tr, p_in)
    p, inf_s = training.timed_predict(m, te[FEATURES].values)
    res, preds = training.evaluate_partition(te, te_obs, p, arcs_te)
    np.save(OUT / f"student_test_proba_{name}.npy", p.astype(np.float32))
    cv = pd.DataFrame(folds)
    summary = {
        "model": name, "family": spec["family"], "config": spec["config"],
        "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
        "cv_top1_mean": cv.top1_acc.mean(), "cv_log_loss_mean": cv.log_loss.mean(),
        "train_macro_f1": r_in["macro_f1"], "train_top1": r_in["top1_acc"],
        "test_top1": res["top1_acc"], "test_macro_f1": res["macro_f1"],
        "test_precision": res["precision"], "test_recall": res["recall"], "test_f1": res["f1"],
        "test_log_loss": res["log_loss"], "test_route_continuity": res["route_continuity"],
        "fit_seconds": fit_s, "n_train_rows": len(Xs),
        "inference_ms_per_1k_obs": 1000 * inf_s / te_obs.shape[0] * 1000,
    }
    return summary, cv, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    args = ap.parse_args()
    tr = training.load_training_rows()
    te, te_obs = training.load_student_test()
    arcs_te = training.arcs_tables(config.STUDENT_TEST_DIR, sorted(te_obs.trip.unique()))
    print(f"training rows={len(tr):,} obs={tr.groupby(['trip','obs']).ngroups:,} trips={tr.trip.nunique()}; "
          f"student-test rows={len(te):,} obs={len(te_obs):,}")

    # non-learned reference: nearest segment
    base, _ = training.evaluate_partition(te, te_obs, -te.dist.values, arcs_te)
    json.dump(base, open(OUT / "baseline_nearest_segment.json", "w"), indent=2)
    print("nearest-segment baseline (student test):", base)

    for name in args.models:
        print(f"== {name}", flush=True)
        summary, cv, _ = run_model(name, tr, te, te_obs, arcs_te)
        cv.to_csv(OUT / f"cv_folds_{name}.csv", index=False)
        pd.Series(summary).to_json(OUT / f"summary_{name}.json", indent=2)
        print(pd.Series(summary).to_string(), flush=True)


if __name__ == "__main__":
    main()
