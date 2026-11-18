"""Shared helpers for loading the feature tables and evaluating a fitted model."""
import time

import numpy as np
import pandas as pd

from . import config, data_io, metrics
from .features import FEATURES


def label_ok_trips(max_median_m=30.0):
    """Training trips whose reference route actually follows the GPS (EDA finding)."""
    st = pd.read_csv(config.FEATURES_DIR / "candidate_stats.csv", dtype={"trip": str})
    st = st[st.partition == "train"]
    return set(st.loc[st.label_dist_median_m <= max_median_m, "trip"])


def load_training_rows():
    """Training rows: label-reliable trips, observations whose reference arc is a candidate."""
    df = pd.read_parquet(config.FEATURES_DIR / "train_candidates.parquet")
    df = df[df.trip.isin(label_ok_trips())]
    pos = df.loc[df.y == 1, ["trip", "obs"]].drop_duplicates()
    df = df.merge(pos, on=["trip", "obs"], how="inner")
    return df.reset_index(drop=True)


def load_student_test():
    df = pd.read_parquet(config.FEATURES_DIR / "student_test_candidates.parquet")
    obs = pd.read_csv(config.FEATURES_DIR / "student_test_observations.csv", dtype={"trip": str})
    return df, obs


def subsample(X, y, max_rows, seed=config.RANDOM_SEED):
    if max_rows is None or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max_rows, replace=False)
    return X[idx], y[idx]


def evaluate_rows(df, proba):
    """Row-level metrics on a candidate table whose rows all have a true candidate (CV)."""
    key = df.trip.astype(str) + "_" + df.obs.astype(str)
    mask = metrics.top1_from_scores(key.values, proba)
    m = metrics.candidate_metrics(df.y.values, mask, proba)
    m["top1_acc"] = float(df.y.values[mask].sum() / key.nunique())
    return m


def evaluate_partition(df, obs_table, proba, arcs_by_trip):
    """Full evaluation on a partition: every GPS observation counts.

    Returns (summary dict, per-observation predicted arcs dict trip->array)."""
    key = df.trip.astype(str) + "_" + df.obs.astype(str)
    mask = metrics.top1_from_scores(key.values, proba)
    cm = metrics.candidate_metrics(df.y.values, mask, proba)
    top = df.loc[mask, ["trip", "obs", "arc"]]
    preds, conts, correct, total = {}, [], 0, 0
    for tid, g in obs_table.groupby("trip"):
        p = np.full(len(g), -1, dtype=np.int64)
        t = top[top.trip == tid]
        p[t.obs.values] = t.arc.values
        preds[tid] = p
        correct += int(np.sum(p == g.ref_arc.values))
        total += len(g)
        conts.append(metrics.route_continuity(p, arcs_by_trip[tid]))
    cm["top1_acc"] = correct / total
    cm["route_continuity"] = float(np.nanmean(conts))
    return cm, preds


def arcs_tables(partition_dir, trip_ids):
    return {t: np.loadtxt(partition_dir / t / f"{t}.arcs", dtype=np.int64, ndmin=2) for t in trip_ids}


def timed_predict(model, X):
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


__all__ = ["FEATURES"]
