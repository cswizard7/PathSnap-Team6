# Raw console logs of the experiment runs (18 Nov 2026, times in UTC)

* `train_a.log` - `scripts/03_train_models.py --models LogisticRegression GaussianNB DecisionTree HistGradientBoosting`
* `train_b.log` - `scripts/03_train_models.py --models RandomForest MLP KNN`
* `bayes.log`   - `scripts/04_bayes_opt.py --trials 15` (Optuna trial-by-trial output, then tuned vs untuned)

Each line `<model> fold k: macroF1=... top1=...` is printed while the 5-fold GroupKFold
cross-validation runs; the per-fold numbers match `results/models/cv_folds_<model>.csv`.
