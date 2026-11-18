"""Evaluation metrics.

Definitions (used identically for every model and for the API):

* Observation-level Top-1 accuracy: for every GPS observation take the
  candidate with the highest score; correct if it equals the reference arc.
  Denominator = ALL observations (observations whose reference arc is not in
  the candidate set, or that have no candidates, count as errors).
* Candidate-level precision / recall / F1 / Macro-F1: every (observation,
  candidate) row is a binary decision; the predicted positive is the top-1
  candidate of its observation. Macro-F1 = mean of the F1 of class 1
  ("is the true segment") and class 0 ("is not"). A per-arc multi-class
  Macro-F1 is not meaningful because arc ids are trip-local.
* Log loss: binary cross-entropy of the classifier's candidate probabilities.
* Candidate miss rate: fraction of observations whose reference arc is not in
  the candidate set (an upper bound on achievable accuracy).
* Route continuity: fraction of consecutive observation pairs whose predicted
  arcs are identical or directly connected (end node of one = start node of
  the next). 1.0 = a physically drivable sequence.
* Route (sequence-level) precision / recall / F1: overlap, weighted by arc
  length, between the set of arcs in the predicted route and the reference
  route.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def top1_from_scores(obs, scores):
    """Return a boolean mask selecting the arg-max row per observation."""
    df = pd.DataFrame({"obs": np.asarray(obs), "s": np.asarray(scores)})
    idx = df.groupby("obs", sort=False)["s"].idxmax().values
    mask = np.zeros(len(df), bool)
    mask[idx] = True
    return mask


def candidate_metrics(y, top1_mask, proba=None):
    y = np.asarray(y).astype(int)
    pred = top1_mask.astype(int)
    out = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }
    if proba is not None:
        out["log_loss"] = log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])
    return out


def route_continuity(arcs_seq, arcs_table):
    s = np.asarray(arcs_seq)
    valid = s >= 0
    a, b = s[:-1], s[1:]
    ok = valid[:-1] & valid[1:]
    if ok.sum() == 0:
        return float("nan")
    a, b = a[ok], b[ok]
    cont = (a == b) | (arcs_table[a, 1] == arcs_table[b, 0])
    return float(cont.mean())


def route_overlap(pred_arcs, ref_arcs, arc_length):
    p = set(int(a) for a in pred_arcs if a >= 0)
    r = set(int(a) for a in ref_arcs)
    if not p or not r:
        return {"route_precision": 0.0, "route_recall": 0.0, "route_f1": 0.0}
    inter = sum(arc_length[a] for a in p & r)
    prec = inter / max(sum(arc_length[a] for a in p), 1e-9)
    rec = inter / max(sum(arc_length[a] for a in r), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def observation_accuracy(pred_arcs, ref_arcs):
    pred_arcs = np.asarray(pred_arcs)
    ref_arcs = np.asarray(ref_arcs)
    return float(np.mean(pred_arcs == ref_arcs))
