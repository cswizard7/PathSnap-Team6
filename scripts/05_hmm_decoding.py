"""Step 5 - HMM/Viterbi decoding experiments.

1. Decoder-parameter tuning (beta, emission_weight) on TRAINING trips using
   out-of-fold classifier probabilities (5-fold GroupKFold, the held-out trip's
   probabilities come from a model that never saw it). Grid search; objective =
   observation-level Top-1 accuracy after decoding.
2. Student-test comparison: classifier-only vs classifier + HMM/Viterbi, for
   the final tuned model AND for every model from step 3 (decoder compatibility).
3. Per-trip analysis (where HMM helped / did not help).

Outputs (results/hmm/) + models/hmm_config.json

Run:  python scripts/05_hmm_decoding.py
"""
import itertools
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
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap import config, data_io, hmm, metrics, network, training
from pathsnap.features import FEATURES

OUT = config.RESULTS_DIR / "hmm"
OUT.mkdir(parents=True, exist_ok=True)
N_TUNE_TRIPS = 30


def decode_eval(part_dir, cands, proba, obs_table, params, trips, cache):
    rows = []
    for tid in trips:
        m = (cands.trip == tid).values
        df, p = cands[m], proba[m]
        if tid not in cache:
            trip = data_io.load_trip(part_dir, tid)
            cache[tid] = (trip, network.RoadNetwork(trip))
        trip, net = cache[tid]
        ref = obs_table[obs_table.trip == tid].ref_arc.values
        clf = np.full(len(ref), -1)
        if len(df):
            mask = metrics.top1_from_scores(df.obs.values, p)
            clf[df.obs.values[mask]] = df.arc.values[mask]
        t0 = time.perf_counter()
        r = hmm.viterbi(trip, net, df, p, params)
        route = hmm.reconstruct_route(net, r["arcs"])
        dt = time.perf_counter() - t0
        ov_h = metrics.route_overlap(route, trip.route, net.length)
        ov_c = metrics.route_overlap(hmm.reconstruct_route(net, clf), trip.route, net.length)
        rows.append({"trip": tid, "n_obs": len(ref),
                     "clf_top1": metrics.observation_accuracy(clf, ref),
                     "hmm_top1": metrics.observation_accuracy(r["arcs"], ref),
                     "clf_continuity": metrics.route_continuity(clf, trip.arcs),
                     "hmm_continuity": metrics.route_continuity(r["arcs"], trip.arcs),
                     "clf_route_f1": ov_c["route_f1"], "hmm_route_f1": ov_h["route_f1"],
                     "hmm_route_precision": ov_h["route_precision"], "hmm_route_recall": ov_h["route_recall"],
                     "n_breaks": r["n_breaks"], "n_segments": r["n_segments"], "decode_seconds": dt})
    return pd.DataFrame(rows)


def weighted(df, col):
    return float(np.average(df[col], weights=df.n_obs))


def main():
    final_cfg = json.load(open(config.MODELS_DIR / "model_config.json"))
    params = final_cfg["params"]

    # ---------- 1. out-of-fold probabilities on training trips ----------
    all_tr = pd.read_parquet(config.FEATURES_DIR / "train_candidates.parquet")
    tr_obs = pd.read_csv(config.FEATURES_DIR / "train_observations.csv", dtype={"trip": str})
    ok = sorted(training.label_ok_trips())
    all_tr = all_tr[all_tr.trip.isin(ok)].reset_index(drop=True)
    fit_rows = training.load_training_rows()
    oof = np.zeros(len(all_tr))
    trips_arr = np.array(ok)
    for k, (i_a, i_b) in enumerate(GroupKFold(n_splits=config.N_FOLDS).split(trips_arr, groups=trips_arr)):
        fit_trips, held = set(trips_arr[i_a]), set(trips_arr[i_b])
        f = fit_rows[fit_rows.trip.isin(fit_trips)]
        m = HistGradientBoostingClassifier(**params).fit(f[FEATURES].values, f.y.values)
        sel = all_tr.trip.isin(held).values
        oof[sel] = m.predict_proba(all_tr.loc[sel, FEATURES].values)[:, 1]
        print("oof fold", k, flush=True)

    rng = np.random.default_rng(config.RANDOM_SEED)
    tune_trips = sorted(rng.choice(ok, size=min(N_TUNE_TRIPS, len(ok)), replace=False).tolist())
    cache = {}
    grid = []
    for beta, w in itertools.product([2.0, 5.0, 10.0, 20.0, 40.0], [0.5, 1.0, 2.0]):
        hp = hmm.HMMParams(beta=beta, emission_weight=w)
        res = decode_eval(config.TRAIN_DIR, all_tr, oof, tr_obs, hp, tune_trips, cache)
        grid.append({"beta": beta, "emission_weight": w, "hmm_top1": weighted(res, "hmm_top1"),
                     "clf_top1": weighted(res, "clf_top1"), "hmm_continuity": weighted(res, "hmm_continuity"),
                     "hmm_route_f1": res.hmm_route_f1.mean()})
        print(grid[-1], flush=True)
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "decoder_param_search_train_oof.csv", index=False)
    best = grid.loc[grid.hmm_top1.idxmax()]
    best_params = {"beta": float(best.beta), "emission_weight": float(best.emission_weight)}
    json.dump({"params": best_params, "tuned_on": f"{len(tune_trips)} training trips, out-of-fold probabilities",
               "objective": "observation-level top-1 accuracy", "gap_seconds": hmm.GAP_SECONDS,
               "back_slack_m": hmm.BACK_SLACK_M},
              open(config.MODELS_DIR / "hmm_config.json", "w"), indent=2)
    print("best decoder params:", best_params)

    # heat-map of the search
    piv = grid.pivot(index="beta", columns="emission_weight", values="hmm_top1")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set(xlabel="emission weight", ylabel="beta (m)", title="HMM top-1 (train OOF) vs decoder params")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "decoder_param_search.png", dpi=120); plt.close(fig)

    # ---------- 2. student test ----------
    te, te_obs = training.load_student_test()
    hp = hmm.HMMParams(**best_params)
    st_trips = sorted(te_obs.trip.unique())
    cache = {}
    final = joblib.load(config.MODELS_DIR / "final_model.joblib")
    p_final = final.predict_proba(te[FEATURES].values)[:, 1]
    per_trip = decode_eval(config.STUDENT_TEST_DIR, te, p_final, te_obs, hp, st_trips, cache)
    per_trip.to_csv(OUT / "student_test_per_trip_final_model.csv", index=False)
    summ = [{"model": "HGB_tuned (final)", **{c: weighted(per_trip, c) for c in
                                              ["clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity"]},
             "clf_route_f1": per_trip.clf_route_f1.mean(), "hmm_route_f1": per_trip.hmm_route_f1.mean(),
             "decode_seconds_total": per_trip.decode_seconds.sum()}]
    for f in sorted((config.RESULTS_DIR / "models").glob("student_test_proba_*.npy")):
        name = f.stem.replace("student_test_proba_", "")
        r = decode_eval(config.STUDENT_TEST_DIR, te, np.load(f).astype(float), te_obs, hp, st_trips, cache)
        summ.append({"model": name, **{c: weighted(r, c) for c in
                                       ["clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity"]},
                     "clf_route_f1": r.clf_route_f1.mean(), "hmm_route_f1": r.hmm_route_f1.mean(),
                     "decode_seconds_total": r.decode_seconds.sum()})
        print(summ[-1], flush=True)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "classifier_vs_hmm_student_test.csv", index=False)

    # ---------- 3. plots ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(summ))
    ax[0].bar(x - .2, summ.clf_top1, .4, label="classifier only")
    ax[0].bar(x + .2, summ.hmm_top1, .4, label="classifier + HMM/Viterbi")
    ax[0].set_xticks(x, summ.model, rotation=30, ha="right"); ax[0].set_ylim(0.5, 1)
    ax[0].set(title="Student-test observation Top-1 accuracy", ylabel="accuracy"); ax[0].legend()
    ax[1].bar(x - .2, summ.clf_continuity, .4, label="classifier only")
    ax[1].bar(x + .2, summ.hmm_continuity, .4, label="classifier + HMM/Viterbi")
    ax[1].set_xticks(x, summ.model, rotation=30, ha="right"); ax[1].set_ylim(0.5, 1)
    ax[1].set(title="Student-test route continuity", ylabel="fraction of valid transitions"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "classifier_vs_hmm.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(per_trip))
    ax.bar(x - .2, per_trip.clf_top1, .4, label="classifier only")
    ax.bar(x + .2, per_trip.hmm_top1, .4, label="+ HMM/Viterbi")
    ax.set_xticks(x, per_trip.trip, rotation=45); ax.set_ylim(0.4, 1)
    ax.set(title="Per student-test trip Top-1 accuracy (final model)", ylabel="accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "per_trip_student_test.png", dpi=120); plt.close(fig)
    print(summ.to_string())
    print(per_trip.to_string())


if __name__ == "__main__":
    main()
