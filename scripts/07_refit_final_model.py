"""Step 7 (optional) - Re-fit the final model from the recorded best hyperparameters.

Use this on a new machine if models/final_model.joblib cannot be loaded
(e.g. a different scikit-learn version). It needs only the training data and
results/bayes_opt/best_params.json, and takes well under a minute.

Run:  python scripts/02_build_features.py   (if results/features/ is missing)
      python scripts/07_refit_final_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap import config, training
from pathsnap.features import FEATURES


def main():
    best = json.load(open(config.RESULTS_DIR / "bayes_opt" / "best_params.json"))["best_params"]
    params = dict(best, early_stopping=False, random_state=config.RANDOM_SEED)
    tr = training.load_training_rows()
    m = HistGradientBoostingClassifier(**params).fit(tr[FEATURES].values, tr.y.values)
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(m, config.MODELS_DIR / "final_model.joblib", compress=3)
    json.dump({"model": "HistGradientBoostingClassifier", "params": params, "features": FEATURES,
               "candidate_radius_m": config.CANDIDATE_RADIUS_M, "max_candidates": config.MAX_CANDIDATES},
              open(config.MODELS_DIR / "model_config.json", "w"), indent=2)
    print("saved", config.MODELS_DIR / "final_model.joblib")


if __name__ == "__main__":
    main()
