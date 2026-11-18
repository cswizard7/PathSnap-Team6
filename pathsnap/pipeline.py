"""End-to-end inference pipeline (used by the API, the map scripts and the HMM experiments).

predict_trip() never reads the .route file: the trip is loaded with
with_route=False, so reference labels cannot enter the prediction.
Scoring against the reference is a separate, server-side step (score_trip).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import config, data_io, features, hmm, labels, metrics, network


class PathSnapModel:
    def __init__(self, models_dir=config.MODELS_DIR):
        models_dir = Path(models_dir)
        self.clf = joblib.load(models_dir / "final_model.joblib")
        self.cfg = json.load(open(models_dir / "model_config.json"))
        hp = json.load(open(models_dir / "hmm_config.json")) if (models_dir / "hmm_config.json").exists() else {}
        self.hmm_params = hmm.HMMParams(**hp.get("params", {}))
        self.features = self.cfg["features"]

    def predict_trip(self, partition_dir, trip_id):
        timings = {}
        t0 = time.perf_counter()
        trip = data_io.load_trip(partition_dir, trip_id, with_route=False)
        net = network.RoadNetwork(trip)
        timings["load_and_index_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cand = features.build_candidate_table(trip, net, labels=None,
                                              radius=self.cfg["candidate_radius_m"], k=self.cfg["max_candidates"])
        timings["features_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = self.clf.predict_proba(cand[self.features].values)[:, 1] if len(cand) else np.zeros(0)
        timings["classifier_s"] = time.perf_counter() - t0
        n = len(trip.xy)
        clf_arcs = np.full(n, -1, dtype=np.int64)
        if len(cand):
            mask = metrics.top1_from_scores(cand.obs.values, proba)
            clf_arcs[cand.obs.values[mask]] = cand.arc.values[mask]
        t0 = time.perf_counter()
        dec = hmm.viterbi(trip, net, cand, proba, self.hmm_params)
        route = hmm.reconstruct_route(net, dec["arcs"])
        timings["hmm_viterbi_s"] = time.perf_counter() - t0
        return {"trip": trip, "net": net, "candidates": cand, "proba": proba,
                "classifier_arcs": clf_arcs, "hmm_arcs": dec["arcs"], "route": route,
                "n_breaks": dec["n_breaks"], "n_segments": dec["n_segments"], "timings": timings}


def score_trip(partition_dir, trip_id, pred):
    """Server-side evaluation against the reference route (never sent to the model)."""
    ref_trip = data_io.load_trip(partition_dir, trip_id, with_route=True)
    if ref_trip.route is None:
        return None
    ref_obs, _ = labels.align_route(ref_trip)
    net, cand = pred["net"], pred["candidates"]
    in_cand = np.zeros(len(ref_obs), bool)
    if len(cand):
        hit = cand.arc.values == ref_obs[cand.obs.values]
        in_cand[cand.obs.values[hit]] = True
    out = {
        "n_observations": int(len(ref_obs)),
        "hmm_top1_accuracy": metrics.observation_accuracy(pred["hmm_arcs"], ref_obs),
        "classifier_top1_accuracy": metrics.observation_accuracy(pred["classifier_arcs"], ref_obs),
        "candidate_miss_rate": float(1 - in_cand.mean()),
        "hmm_route_continuity": metrics.route_continuity(pred["hmm_arcs"], ref_trip.arcs),
        "classifier_route_continuity": metrics.route_continuity(pred["classifier_arcs"], ref_trip.arcs),
    }
    out.update(metrics.route_overlap(pred["route"], ref_trip.route, net.length))
    if len(cand):
        y = (cand.arc.values == ref_obs[cand.obs.values]).astype(int)
        mask = metrics.top1_from_scores(cand.obs.values, pred["proba"])
        cm = metrics.candidate_metrics(y, mask, pred["proba"])
        out.update({"classifier_macro_f1": cm["macro_f1"], "classifier_log_loss": cm["log_loss"]})
    return out
