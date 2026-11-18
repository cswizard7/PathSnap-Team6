# First-round evaluation - preparation sheet

The brief's work process is: understand problem -> explore data -> processing/features -> train and
compare >= 5 models -> select -> Bayesian optimisation -> HMM/Viterbi -> export -> API -> UI ->
integrate -> report. A first-round check will most likely ask "how far are you and do you understand
it". Have the following open and be ready to talk for ~5-8 minutes, then answer questions.

## What to have open on the laptop
1. `results/eda/maps/eda_student_test_00000000.html` (Folium: GPS, network, reference route) - in a browser.
2. `results/eda/data_quality_notes.md` and `results/eda/gps_to_reference_distance.png`.
3. `results/models/model_comparison.csv` + `results/models/cv_macro_f1_comparison.png`.
4. `results/bayes_opt/optimisation_history.png` and `tuned_vs_untuned.csv`.
5. `results/hmm/classifier_vs_hmm.png` and one map from `results/maps/` (predicted vs reference).
6. A live demo: API in one CMD window, UI in the browser (student-test trips) - see `docs/EVALUATION_DAY.md`.

## Suggested flow (who speaks)
| Part | Speaker | Points |
|---|---|---|
| Problem + data (1.5 min) | Member 1 | GPS noise; 4 file types; route is an arc sequence -> we derive per-point labels with monotone DP alignment; trip-local networks; per-trip metric projection; issues found (trips 33/54 wrong routes, 42 h gap in trip 90, two-way arcs) |
| Candidates + features (1.5 min) | Member 2 | 8 arcs within 50 m (why 50 m); binary (point, candidate) framing; 20 features incl. heading vs directed arc and temporal window; metrics definitions; candidate miss rate 0.4 % |
| Models + Bayesian opt (1.5 min) | Member 3 | 7 models + baseline, GroupKFold by trip; table; why HGB; TPE search, objective, trials, tuned vs untuned |
| HMM + system (1.5 min) | Member 4 | states/emissions/transitions/Viterbi; gains over classifier-only; API/UI demo with logs |

## Numbers to remember
See `docs/VIVA_GUIDE.md` (kept in sync with `results/`).

## If they ask what is left
* Final technical report (PDF) with screenshots of UI + API logs.
* Running on the 10 evaluator trips once they are provided (drop into `data/evaluator_test/`).
