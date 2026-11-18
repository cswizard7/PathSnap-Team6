# PathSnap - Map-Matching & Route Reconstruction (UE24CS352A Hackathon)

PathSnap matches noisy GPS trajectories to a road network and reconstructs the
driven route:

```
raw trip files -> cleaning + local metric projection -> candidate road segments (<= 8 within 50 m)
   -> 20 candidate features -> classifier P(segment | GPS point) -> HMM / Viterbi decoding
   -> per-observation road segment + continuous predicted route
```

Deployment uses two laptops: **Laptop 1** runs the FastAPI backend (model, HMM,
road network, server-side scoring); **Laptop 2** runs the Streamlit evaluator UI.

> **AI-assistance disclosure.** Parts of this repository's code and documentation
> were written with the help of an AI assistant (Claude, Anthropic), as allowed
> with disclosure by the event policy. All experiments were actually executed;
> every number in `results/` is produced by the scripts in this repository.

---

## 1. Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library code: data loading/cleaning (`data_io.py`), projection & geometry (`geo.py`), road network + candidate search + shortest paths (`network.py`), reference-label alignment (`labels.py`), feature engineering (`features.py`), model zoo (`models.py`), metrics (`metrics.py`), HMM/Viterbi (`hmm.py`), inference pipeline (`pipeline.py`), Folium maps (`visual.py`) |
| `scripts/` | Numbered, reproducible experiment steps (see section 4) |
| `api/` | FastAPI backend (`main.py`) |
| `ui/` | Streamlit evaluator UI (`app.py`) |
| `models/` | Final model (`final_model.joblib`), `model_config.json`, `hmm_config.json` |
| `results/eda/` | Inventory, statistics, data-quality notes, plots, Folium EDA maps |
| `results/features/` | Candidate statistics (large feature tables are regenerated, not committed) |
| `results/models/` | CV folds, model comparison table, plots, confusion matrices, training histories |
| `results/bayes_opt/` | Full Bayesian-optimisation trial record, best params, tuned vs untuned |
| `results/hmm/` | Decoder-parameter search, classifier-only vs HMM comparison, per-trip results |
| `results/maps/` | Interactive Folium maps (GPS, network, reference, predicted route) |
| `data/` | Raw data goes here (not committed) |
| `docs/` | Evaluation-day runbook, viva preparation guide |

## 2. Installation (Windows 11, Command Prompt)

Requires **64-bit Python 3.11, 3.12 or 3.13** (`python --version`).

```bat
cd path\to\PathSnap
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Laptop 2 (UI only) can instead run `pip install -r requirements-ui.txt`.

## 3. Data placement

Copy the provided trip folders so that the layout is:

```
data\train\00000001\00000001.track  (.nodes .arcs .route)   <- 80 trips
data\student_test\00000000\...                              <- 10 trips
data\evaluator_test\<id>\...                                <- 10 trips, on evaluation day only
```

## 4. Reproducing every result (run from the repository root)

| Step | Command | Time* | Output |
|---|---|---|---|
| 1 EDA | `python scripts\01_eda.py` | ~3 min | `results/eda/` |
| 1b EDA checklist | `python scripts\01b_eda_checklist.py` | <1 min | `results/eda/checklist/` |
| 2 Candidates + features | `python scripts\02_build_features.py` | ~1 min | `results/features/` |
| 3 Train/compare 7 models | `python scripts\03_train_models.py` | ~40 min | `results/models/` |
| 3b Training histories | `python scripts\03b_learning_curves.py` | ~5 min | `results/models/training_histories.png` |
| 3c Comparison table/plots | `python scripts\03c_compare_models.py` | <1 min | `results/models/model_comparison.csv` |
| 4 Bayesian optimisation | `python scripts\04_bayes_opt.py --trials 15` | ~25 min | `results/bayes_opt/`, `models/final_model.joblib` |
| 5 HMM/Viterbi experiments | `python scripts\05_hmm_decoding.py` | ~10 min | `results/hmm/`, `models/hmm_config.json` |
| 6 Folium maps | `python scripts\06_make_maps.py` | ~1 min | `results/maps/*.html` |
| 7 (fallback) refit final model | `python scripts\07_refit_final_model.py` | <1 min | `models/final_model.joblib` |

\*on a 2-core machine. All randomness uses seed 42.

**Running the training on Kaggle instead:** `notebooks/pathsnap_kaggle_training.ipynb` is a
self-contained notebook (it writes these same modules/scripts and runs steps 1-6). Import it into
Kaggle, attach the data as a dataset, and use *Save Version -> Save & Run All*. Download
`pathsnap_outputs.zip` from the Output tab and unzip it into the repository root (it contains
`results/` and `models/`). If the laptops' scikit-learn version differs from the one recorded in
`models/training_environment.json`, either install that version or rebuild the model locally with
`python scripts\02_build_features.py` then `python scripts\07_refit_final_model.py`.

## 5. Running the two-laptop deployment

Both laptops must be on the **same Wi-Fi / hotspot**.

**Laptop 1 - API**

```bat
cd path\to\PathSnap
.venv\Scripts\activate
set PATHSNAP_EVAL_DIR=data\evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

* Find its IP address with `ipconfig` (IPv4 Address of the Wi-Fi adapter, e.g. `192.168.1.23`).
* If Windows Firewall asks, allow Python on **Private** networks. Or allow the port once (CMD as Administrator):
  `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
* The CMD window shows every request (`GET /trips`, `POST /predict` with trip id, accuracy, timing); the same log is appended to `api_requests.log`.
* Check: open `http://localhost:8000/docs` (interactive API docs) on Laptop 1.

**Laptop 2 - UI**

```bat
cd path\to\PathSnap
.venv\Scripts\activate
set PATHSNAP_API_URL=http://192.168.1.23:8000
streamlit run ui\app.py
```

The browser opens at `http://localhost:8501`. The UI lists the trips served by
the API, sends the selected trip id to `POST /predict`, and shows the predicted
route on a Folium map with the server-computed metrics.

**Testing before evaluation day** (evaluator data not yet available): start the API with
`set PATHSNAP_EVAL_DIR=data\student_test` to serve the 10 student-testing trips instead.

### API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | - | status, model name, HMM parameters, partition served |
| GET | `/trips` | - | `{"partition": ..., "trips": [ids]}` |
| POST | `/predict` | `{"trip_id": "00000000"}` | GPS points, predicted arc per observation, predicted route polylines, classifier-only polylines, server-side metrics, timings |

Example from CMD: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\":\"00000000\"}"`

## 6. Leakage safeguards

* Trips are never split across partitions; CV uses `GroupKFold` by trip.
* Evaluator-testing data is read only by the API at request time.
* `pipeline.predict_trip` loads trips with `with_route=False`; reference routes are read only by
  `pipeline.score_trip` **after** prediction, and are never returned to the UI.
* No predictions are hard-coded; every request runs the full pipeline (visible in the API log).
* Hyperparameters (classifier and decoder) are chosen with training data (CV / out-of-fold) only.

## 7. Method summary

* **Projection:** per-trip local equirectangular (tangent-plane) projection to metres; trips lie on
  several continents, so one UTM zone would not fit.
* **Labels:** `.route` is an arc sequence, not one label per GPS row. Every observation is assigned
  to a route arc by an exact monotone dynamic-programming alignment (`pathsnap/labels.py`).
  Two training trips whose reference route does not follow the GPS (median distance > 30 m) are excluded from fitting.
* **Candidates:** up to 8 directed arcs within 50 m (radius chosen from the training GPS-noise distribution).
* **Models compared:** Logistic Regression, Gaussian NB, KNN, Decision Tree, Random Forest,
  HistGradientBoosting, MLP (+ a nearest-segment baseline).
* **Bayesian optimisation:** Optuna TPE on the selected model, objective = group-aware CV Macro-F1.
* **HMM:** hidden states = candidate arcs; emissions = normalised classifier probabilities;
  transitions = Newson-Krumm route-vs-straight-line distance model with Dijkstra shortest paths;
  gaps > 60 s split the chain, infeasible transitions trigger a break-and-restart.

Results: see `results/` and the technical report.
