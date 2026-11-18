"""PathSnap FastAPI backend (Laptop 1: model + API).

Endpoints
  GET  /health            -> status, model info, partition served
  GET  /trips             -> the trip ids available in the evaluation folder
  POST /predict           -> body {"trip_id": "..."}; runs the full pipeline
                             (candidates -> features -> classifier -> HMM/Viterbi),
                             then scores the prediction SERVER-SIDE against the
                             reference route and returns predicted route + metrics.

The reference route/labels are only read inside pathsnap.pipeline.score_trip,
AFTER the prediction is complete, and are never returned to the client.

Run (from the repository root, Windows CMD):
  set PATHSNAP_EVAL_DIR=data\\evaluator_test
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config, data_io
from pathsnap.pipeline import PathSnapModel, score_trip
from pathsnap.visual import _arcs_to_lines, dedupe_consecutive

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(config.ROOT / "api_requests.log", encoding="utf-8")])
log = logging.getLogger("pathsnap.api")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", str(config.EVALUATOR_TEST_DIR)))
if not EVAL_DIR.is_absolute():
    EVAL_DIR = config.ROOT / EVAL_DIR
MODELS_DIR = Path(os.environ.get("PATHSNAP_MODELS_DIR", str(config.MODELS_DIR)))

app = FastAPI(title="PathSnap Map-Matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = PathSnapModel(MODELS_DIR)
log.info("Model loaded: %s | HMM params: %s", MODEL.cfg["model"], MODEL.hmm_params)
log.info("Serving trips from: %s -> %s", EVAL_DIR, data_io.list_trips(EVAL_DIR))


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL.cfg["model"], "hmm_params": vars(MODEL.hmm_params),
            "eval_dir": EVAL_DIR.name, "n_trips": len(data_io.list_trips(EVAL_DIR))}


@app.get("/trips")
def trips(request: Request):
    ids = data_io.list_trips(EVAL_DIR)
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(ids))
    return {"partition": EVAL_DIR.name, "trips": ids}


def _round_lines(lines):
    return [[[round(lat, 6), round(lon, 6)] for lat, lon in line] for line in lines]


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    client = request.client.host if request.client else "?"
    if tid not in data_io.list_trips(EVAL_DIR):        # whitelist -> no path traversal
        log.warning("POST /predict from %s: unknown trip %r", client, tid)
        raise HTTPException(status_code=404, detail=f"trip {tid} not found")
    log.info("POST /predict from %s: trip=%s -> running pipeline", client, tid)
    t0 = time.perf_counter()
    pred = MODEL.predict_trip(EVAL_DIR, tid)
    t_pred = time.perf_counter() - t0
    scores = score_trip(EVAL_DIR, tid, pred)          # server-side only
    trip = pred["trip"]
    hmm_arcs = pred["hmm_arcs"].tolist()
    resp = {
        "trip_id": tid,
        "n_observations": len(hmm_arcs),
        "gps": [[round(float(la), 6), round(float(lo), 6)] for lo, la in trip.track[:, :2]],
        "predicted_arc_per_observation": hmm_arcs,
        "predicted_route_lines": _round_lines(_arcs_to_lines(trip, pred["route"])),
        "classifier_only_route_lines": _round_lines(
            _arcs_to_lines(trip, dedupe_consecutive([a for a in pred["classifier_arcs"].tolist() if a >= 0]))),
        "n_hmm_breaks": pred["n_breaks"], "n_hmm_segments": pred["n_segments"],
        "metrics": scores,
        "timings_s": {**{k: round(v, 3) for k, v in pred["timings"].items()},
                      "total_prediction_s": round(t_pred, 3),
                      "total_with_scoring_s": round(time.perf_counter() - t0, 3)},
    }
    if scores:
        log.info("trip=%s n_obs=%d | HMM top-1=%.4f | classifier top-1=%.4f | continuity=%.4f | route F1=%.4f | %.2fs",
                 tid, len(hmm_arcs), scores["hmm_top1_accuracy"], scores["classifier_top1_accuracy"],
                 scores["hmm_route_continuity"], scores["route_f1"], t_pred)
    else:
        log.info("trip=%s n_obs=%d | no reference route available | %.2fs", tid, len(hmm_arcs), t_pred)
    return resp
