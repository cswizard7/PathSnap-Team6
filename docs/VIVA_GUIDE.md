# Viva preparation guide

Evaluators may ask **any** member about **any** part (dataset, preprocessing, features, model
comparison, model selection, Bayesian optimisation, HMM, Viterbi, API, UI, results). Each member
owns one area below (matching their commits) but everyone should read the whole guide.
Numbers quoted here come from `results/`; re-check them there before the viva.

---

## Member 1 - Data understanding, cleaning, projection, labels, EDA
Files: `pathsnap/data_io.py`, `geo.py`, `labels.py`, `visual.py`, `scripts/01_eda.py`, `results/eda/`

**Dataset structure.** 100 trips: 80 train, 10 student-test, 10 evaluator-test (not given to us).
Each trip folder has `.track` (lon, lat, t), `.nodes` (lon, lat; row = node id), `.arcs` (from, to;
row = arc id, *directed*), `.route` (chronological arc ids). Every trip has its **own** road network
- arc ids are only meaningful inside that trip. 208,417 GPS points in train, 19,160 in student-test.

**Key EDA findings** (`results/eda/data_quality_notes.md`)
* No missing values, no non-increasing timestamps. ~94 % of steps are exactly 1 s; 1,260 gaps > 5 s,
  135 gaps > 30 s; the largest is 153,064 s (~42 h) in trip 00000090.
* Trips are on several continents (lon -100..137) -> per-trip projection.
* `.route` is an arc sequence (e.g. 525 arcs for 1,315 GPS rows in trip 1), so it is **not** one label
  per GPS row. Consecutive route arcs are connected (99.9 %).
* Most roads are two-way and stored as two opposite directed arcs (median 95 %) - the nearest
  segment alone cannot tell the direction; that is why the nearest-segment baseline is only 54 %.
* GPS-to-reference distance (clean training trips): median 5.0 m, p90 13.6 m, p99 70 m.
* Trips **00000033** and **00000054** have reference routes that do not follow the GPS (median 122 m and
  140 m away) -> treated as label errors and excluded from model fitting (kept in reporting).

**Why the local projection?** Lon/lat are angles; 1 deg of longitude is 111 km x cos(lat). We project
each trip onto a tangent plane at its mean position: `x = R*dlon*cos(lat0)`, `y = R*dlat`. Error is far
below 1 % over a city-scale trip; no external library; works on any continent (one UTM zone would not).

**How are per-observation labels made?** (`labels.py`) Monotone alignment by dynamic programming:
assign every GPS point i to a route index j_i, with j_0 <= j_1 <= ... (the vehicle never goes back in
the route), minimising the sum of point-to-segment distances.
`D[i,j] = c(i,j) + min_{j'<=j} D[i-1,j']`, computed with a running prefix minimum -> O(N*M).
Back-pointers give the labels. Limitation: labels are derived, so they inherit route errors
(trips 33/54) and are ambiguous exactly at junctions.

**Cleaning** (same for every partition, uses no labels): drop non-finite/out-of-range rows, sort by
time, drop duplicate timestamps (none found).

**Folium maps**: `visual.trip_map` - layers for GPS (blue dots), road network near the trajectory
(grey), reference route (green), classifier-only (orange dashed), predicted route (red), legend,
layer control, fit-bounds.

---

## Member 2 - Candidate generation, features, metrics
Files: `pathsnap/network.py`, `features.py`, `metrics.py`, `training.py`, `scripts/02_build_features.py`,
`scripts/03c_compare_models.py`

**Candidates.** Segments are sampled every 10 m into a KD-tree; for each GPS point we query
radius + 6 m, then compute exact point-to-segment distance and keep the **8 nearest arcs within 50 m**.
50 m was chosen from the training GPS-noise distribution: going from 50 m to 100 m only reduces the
miss rate from 1.4 % to 0.6 % (mostly label errors) while adding many more wrong candidates.
Candidate miss rate: 0.4 % on student-test. ~6.1 candidates per point on average.

**Problem framing.** Each (GPS point, candidate arc) pair is one row with a binary target "is this the
reference arc". The classifier's probabilities are normalised over the candidates of each point ->
P(arc | point). Top-1 = arg-max. These probabilities are the HMM emissions.

**20 features** (`features.py` docstring has the list): distance and its rank/gap/ratio among the
candidates, number of candidates, heading difference (GPS direction vs *directed* arc bearing, over
+/-1 and +/-3 points), heading-valid flag (stationary vehicles), speed, arc length, projection position
on the arc and how far it falls outside, distance of the previous/next point and of a +/-3 window to
the same arc (temporal context), node degrees (junction complexity), two-way flag, time gaps.
Why: distance alone cannot resolve parallel roads or the two directions of a road; heading resolves
direction; window distance resolves points near junctions/parallel streets.

**Metrics** (`metrics.py` docstring): observation-level top-1 accuracy (denominator = all points);
candidate-level precision/recall/F1/Macro-F1 using the top-1 decision (Macro-F1 = mean F1 of the two
classes - a per-arc multi-class F1 is meaningless because arc ids are trip-local); log loss;
candidate miss rate; route continuity (consecutive predicted arcs identical or connected);
length-weighted route precision/recall/F1.

**Primary selection metric**: CV Macro-F1 (required by the brief, group-aware so it measures
generalisation to unseen trips), with student-test top-1, log loss (probability quality matters for
the HMM) and inference time as secondary criteria.

---

## Member 3 - Models, comparison, Bayesian optimisation
Files: `pathsnap/models.py`, `scripts/03_train_models.py`, `03b_learning_curves.py`, `04_bayes_opt.py`,
`07_refit_final_model.py`, `results/models/`, `results/bayes_opt/`, `models/`

**Protocol.** 5-fold GroupKFold by trip on the training trips (a trip never appears in both fit and
validation); final fit on all training rows; comparison on the 10 student-test trips; evaluator data
never used. Seeds fixed (42).

**7 models** (configs in `models.py`): Logistic Regression, Gaussian NB, KNN (k=25, distance-weighted,
100k-row sample because prediction cost grows with training size), Decision Tree, Random Forest,
HistGradientBoosting, MLP (64,32). Plus a non-learned nearest-segment baseline.
Results table: `results/models/model_comparison.csv`.

**Reading the results.** All models beat the nearest-segment baseline (53.7 % -> 85.6-88.6 % top-1).
GaussianNB is worst (independence assumption is violated: distance, rank, gap, ratio are strongly
correlated) and has very poor log loss (over-confident). KNN is the slowest at inference (~1 s per
1000 points) and has worse probabilities. Tree ensembles, boosting and MLP are within one fold-std of
each other. Overfitting check: train vs CV Macro-F1 gap is small for all (e.g. HGB 0.949 vs 0.937);
DT and KNN show the largest gaps.
*Known issue:* the recorded KNN `train_macro_f1` / `train_top1` (0.9483096914 / 0.9148321103) are
identical to the Decision Tree's and could not be reproduced: recomputing KNN's in-sample metrics
with the same code and seed gives 0.9483198135 / 0.9148487881. The difference is in the 5th decimal
and does not change any conclusion; the CV and student-test numbers are unaffected.

**Why HistGradientBoosting was selected** (be honest - RF had the highest *untuned* CV Macro-F1):
* Untuned, RF (0.9392) beat HGB (0.9374) on every one of the 5 folds, but only by 0.0004-0.0035
  (mean 0.0018), well inside the fold-to-fold spread (std ~0.013). HGB had the best student-test
  top-1 (0.8862) and Macro-F1 (0.9327) of the untuned models.
* HGB trains ~3x faster than RF (~15 s vs ~41 s per CV fit), which made a proper Bayesian search with
  group-aware CV affordable on our hardware.
* Good probability quality (log loss 0.114) -> usable HMM emissions; handles unscaled features and
  interactions natively.
* Outcome: tuned HGB reached 5-fold CV Macro-F1 **0.9403** (> RF 0.9392), student-test top-1 0.8957,
  Macro-F1 0.9384, log loss 0.0918 (`results/bayes_opt/tuned_vs_untuned.csv`).

**Bayesian optimisation** (`04_bayes_opt.py`): Optuna **TPE** sampler (a sequential model-based /
Bayesian method: it models p(params | good score) and p(params | bad score) from past trials and
proposes points maximising their ratio - expected improvement). Objective = mean 3-fold GroupKFold
CV Macro-F1 on training trips. Space: learning_rate log[0.02,0.3], max_iter [100,600], max_leaf_nodes
[15,127], min_samples_leaf log[20,1000], l2_regularization log[1e-4,10], max_features [0.5,1].
Trial 0 = untuned config. Number of trials: see `best_params.json` (n_trials). Record: `results/bayes_opt/trials.csv`; best:
`best_params.json`; tuned vs untuned with the same 5-fold CV + student test: `tuned_vs_untuned.csv`.

**Training curves**: only HGB (per boosting iteration) and MLP (per epoch) have histories
(`training_histories.png`); no artificial curves for the other models. These curves use the
*initial* (untuned) configurations on an 80/20 trip split:
* HGB (lr 0.1, 300 iterations): validation log loss is lowest at iteration 81 (0.0832) and then
  rises to 0.1069 at 300 while training loss stays between 0.083 and 0.092 (iterations 50-300)
  -> the untuned HGB overfits
  (in probability terms) after ~80 iterations. The tuned model uses a smaller learning rate (0.044),
  strong L2 (6.4), larger leaves (85) and feature subsampling (0.51), which counteracts this.
* MLP: validation log loss falls from 0.1151 (epoch 1) to 0.0870 (epoch 20) and 0.0862 (epoch 30),
  always below the training loss -> no overfitting within 30 epochs; the 20 epochs used in the
  model comparison are close to, but not exactly at, the best epoch (28, 0.0854).

---

## Member 4 - HMM/Viterbi, pipeline, API, UI, deployment
Files: `pathsnap/hmm.py`, `pipeline.py`, `scripts/05_hmm_decoding.py`, `06_make_maps.py`, `api/main.py`,
`ui/app.py`, `README.md`, `docs/`

**Hidden states**: at each GPS point, the candidate arcs (with the projected position on the arc).
**Emission**: `w * log(p_c / sum p)` - the normalised classifier probability (floor 1e-6).
**Transition** (Newson & Krumm 2009): `-|d_route - d_gc| / beta`, where d_gc is the straight-line
distance between consecutive GPS points and d_route is the network distance between the two
projected positions (rest of arc a + Dijkstra shortest path end(a)->start(b) + offset on b; on the same
arc the difference of positions, with 15 m backward tolerance for noise). Unreachable -> -inf.
Search limit = max(300 m, 4*d_gc + 100 m); Dijkstra results cached per node.
**Viterbi**: `V_i(b) = E_i(b) + max_a [V_{i-1}(a) + T(a,b)]`, keep arg-max back-pointers, back-track
from the best final state. Log-space -> no underflow.
**Gaps/missing candidates**: points without candidates get no state (prediction -1) and the chain
bridges over them; gaps > 60 s split the chain; if no transition is feasible (disconnected segments),
the chain breaks and restarts from emissions (counted as `n_breaks`).
**Decoder parameters** (beta, emission weight) tuned by grid search on 30 training trips using
**out-of-fold** probabilities (models that never saw those trips) - `results/hmm/`.
**Route reconstruction**: consecutive decoded arcs that are not adjacent are joined with the
shortest path, giving a continuous drivable route for the map.

**Decoder tuning result** (`results/hmm/decoder_param_search_train_oof.csv`): best beta = 2 m,
emission weight = 2 (train-OOF top-1 0.9250 vs 0.8965 classifier-only on the same 30 trips).
Saved in `models/hmm_config.json`. Note: multiplying every log-score by a constant does not change
the Viterbi arg-max, so the decoder depends only on the product beta x w (= 4 here); the 15 grid
points give 9 distinct products (1 ... 80). The best product (4) lies between tested neighbours
2.5 (0.9243) and 5 (0.9246), i.e. inside the range, not at its edge. Large products (weak
transitions, e.g. 80 -> 0.9136) behave more like the classifier alone.

**Effect on the student-test trips** (`results/hmm/classifier_vs_hmm_student_test.csv`; top-1 and
continuity are observation-weighted over the 10 trips, route F1 is the mean over trips):

| Final tuned HGB | classifier only | + HMM/Viterbi |
|---|---|---|
| Top-1 accuracy | 0.8957 | **0.9307** |
| Route continuity | 0.9411 | **0.9825** |
| Route F1 (length-weighted) | 0.8432 | **0.9681** |

* HMM improved top-1 for **every** model (e.g. GaussianNB 0.8416 -> 0.9027, LR 0.8682 -> 0.9061)
  and on **every** one of the 10 trips (`student_test_per_trip_final_model.csv`).
* Largest gain: 00000077 (0.7773 -> 0.8384) and 00000076 (0.9246 -> 0.9738). Smallest: 00000007
  (0.9335 -> 0.9483) and 00000000 (0.9473 -> 0.9628), which were already good.
* Where it did *not* fully help: 00000077 stays the worst trip (0.8384); 00000065 has route
  precision 0.788 (recall 1.0) - the shortest-path filling adds arcs that are not on the reference
  route - and 2 chain breaks; 00000000, 00000056 and 00000065 had 2 breaks each (no feasible
  transition -> restart). Continuity is 0.98, not 1.0, because between two 1 s points the decoded
  arcs can be several arcs apart (connected by a path, not directly adjacent) and because of breaks.
* **Honest caveat on model selection**: after HMM decoding the ranking changes - RandomForest + HMM
  0.9352, untuned HGB + HMM 0.9319, tuned HGB + HMM 0.9307. The final model was chosen on CV
  Macro-F1 of the classifier (the metric we fixed in advance), and RF was never tuned. The gaps are
  small (<0.5 pp on 10 trips) and we did not re-select on student-test results. A better approach
  next time: make the post-HMM accuracy (on out-of-fold training probabilities) the BO objective.
* Decoding time for all 10 trips: 7.2 s in total for the final model.

**API** (FastAPI): `/health`, `/trips`, `POST /predict {trip_id}`. Trip id is whitelisted against the
folder (no path traversal). `predict_trip` loads the trip **without** the route; `score_trip` reads the
reference only after prediction to compute accuracy server-side; the response never contains the
reference. Every request is logged to the CMD window and `api_requests.log`.
**UI** (Streamlit): lists the trips from `/trips`, sends the selected id, shows metric cards and a Folium
map (GPS, predicted route, optional classifier-only layer), CSV download of per-point predictions.

---

## Likely questions - quick answers
* *Why not use raw lon/lat distance?* Degrees are not metres and longitude shrinks with latitude.
* *What is a "candidate miss"?* The true arc is not among the 8 candidates within 50 m -> unrecoverable.
* *Why group k-fold?* Points of one trip are highly correlated; random splits would leak and inflate scores.
* *Overfitting?* Compare `train_macro_f1` vs `cv_macro_f1_mean` in `model_comparison.csv`; HGB/MLP curves.
* *Where could the system fail?* Parallel roads closer than GPS noise, long gaps, missing roads in the
  network, wrong reference routes (33, 54).
* *Is anything hard-coded?* No - the API log shows the full pipeline running per request.
