"""Step 4 - Bayesian hyperparameter optimisation of the selected model.

Selected model: HistGradientBoostingClassifier (see results/models/model_comparison.csv
and docs/ for the reasoning).

Method   : Optuna TPE sampler (Tree-structured Parzen Estimator = sequential
           model-based / Bayesian optimisation), seed 42.
Objective: maximise mean group-aware CV Macro-F1 (3-fold GroupKFold by trip on
           the training partition). Student-test and evaluator data are NOT used.
Search space (justification in docs/):
    learning_rate      log-uniform [0.02, 0.3]
    max_iter           int [100, 600]
    max_leaf_nodes     int [15, 127]
    min_samples_leaf   log-int [20, 1000]
    l2_regularization  log-uniform [1e-4, 10]
    max_features       uniform [0.5, 1.0]   (feature subsampling per split)

Outputs (results/bayes_opt/): trials.csv (full search record), best_params.json,
tuned_vs_untuned.csv, optimisation_history.png, param_importance.png
and the final model in models/.

Run:  python scripts/04_bayes_opt.py --trials 15
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap import config, training
from pathsnap.features import FEATURES

OUT = config.RESULTS_DIR / "bayes_opt"
OUT.mkdir(parents=True, exist_ok=True)
config.MODELS_DIR.mkdir(exist_ok=True)

UNTUNED = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False,
               random_state=config.RANDOM_SEED)


def cv_score(params, tr, n_folds):
    X, y, g = tr[FEATURES].values, tr.y.values, tr.trip.values
    scores, tops, lls = [], [], []
    for i_tr, i_va in GroupKFold(n_splits=n_folds).split(X, y, g):
        m = HistGradientBoostingClassifier(**params).fit(X[i_tr], y[i_tr])
        r = training.evaluate_rows(tr.iloc[i_va], m.predict_proba(X[i_va])[:, 1])
        scores.append(r["macro_f1"]); tops.append(r["top1_acc"]); lls.append(r["log_loss"])
    return float(np.mean(scores)), float(np.mean(tops)), float(np.mean(lls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    tr = training.load_training_rows()
    te, te_obs = training.load_student_test()
    arcs_te = training.arcs_tables(config.STUDENT_TEST_DIR, sorted(te_obs.trip.unique()))

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 1000, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
            early_stopping=False, random_state=config.RANDOM_SEED)
        t0 = time.time()
        f1, top1, ll = cv_score(params, tr, args.folds)
        trial.set_user_attr("cv_top1", top1)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: macroF1={f1:.5f} top1={top1:.4f} ({time.time()-t0:.0f}s) {trial.params}", flush=True)
        return f1

    storage = f"sqlite:///{(OUT / 'optuna_study.db').as_posix()}"
    if (OUT / "optuna_study.db").exists():
        (OUT / "optuna_study.db").unlink()
    study = optuna.create_study(direction="maximize", study_name="pathsnap_hgb", storage=storage,
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED, n_startup_trials=5))
    # trial 0 = the untuned configuration, so the search starts from the baseline
    study.enqueue_trial(dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
                             l2_regularization=1e-4, max_features=1.0))
    t_search = time.time()
    study.optimize(objective, n_trials=args.trials)
    search_s = time.time() - t_search
    df = study.trials_dataframe()
    df.to_csv(OUT / "trials.csv", index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv_macro_f1": study.best_value, "n_trials": len(study.trials),
               "cv_folds": args.folds, "search_seconds": search_s, "sampler": "TPE (seed 42)"},
              open(OUT / "best_params.json", "w"), indent=2)

    # ---- tuned vs untuned, same protocol (5-fold CV + student test) ----
    rows = []
    for name, params in [("untuned", UNTUNED),
                         ("tuned", dict(best, early_stopping=False, random_state=config.RANDOM_SEED))]:
        f1, top1, ll = cv_score(params, tr, config.N_FOLDS)
        t0 = time.time()
        m = HistGradientBoostingClassifier(**params).fit(tr[FEATURES].values, tr.y.values)
        fit_s = time.time() - t0
        p, inf_s = training.timed_predict(m, te[FEATURES].values)
        res, _ = training.evaluate_partition(te, te_obs, p, arcs_te)
        np.save(OUT / f"student_test_proba_{name}.npy", p.astype(np.float32))
        rows.append({"model": f"HGB_{name}", "cv_macro_f1_5fold": f1, "cv_top1_5fold": top1, "cv_log_loss": ll,
                     "test_top1": res["top1_acc"], "test_macro_f1": res["macro_f1"],
                     "test_log_loss": res["log_loss"], "test_route_continuity": res["route_continuity"],
                     "fit_seconds": fit_s, "inference_ms_per_1k_obs": 1e6 * inf_s / len(te_obs)})
        if name == "tuned":
            joblib.dump(m, config.MODELS_DIR / "final_model.joblib", compress=3)
            json.dump({"model": "HistGradientBoostingClassifier", "params": params, "features": FEATURES,
                       "candidate_radius_m": config.CANDIDATE_RADIUS_M, "max_candidates": config.MAX_CANDIDATES},
                      open(config.MODELS_DIR / "model_config.json", "w"), indent=2)
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.number, df.value, "o", label="trial CV Macro-F1")
    ax.plot(df.number, df.value.cummax(), "-", label="best so far")
    ax.set(title=f"Bayesian optimisation (TPE) history, {len(df)} trials", xlabel="trial", ylabel="3-fold CV Macro-F1")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "optimisation_history.png", dpi=120); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        ax.set(title="Hyperparameter importance (fANOVA)", xlabel="importance")
        fig.tight_layout(); fig.savefig(OUT / "param_importance.png", dpi=120); plt.close(fig)
        pd.Series(imp).to_csv(OUT / "param_importance.csv", header=["importance"])
    except Exception as e:  # importance needs >1 distinct value per param
        print("importance skipped:", e)


if __name__ == "__main__":
    main()
