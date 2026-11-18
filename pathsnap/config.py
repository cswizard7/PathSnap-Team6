"""Central configuration. All paths are relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
EVALUATOR_TEST_DIR = DATA_DIR / "evaluator_test"   # filled on evaluation day, never used in development
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FEATURES_DIR = ROOT / "results" / "features"

RANDOM_SEED = 42

# Candidate generation (chosen from the training-partition EDA, see results/eda/)
CANDIDATE_RADIUS_M = 50.0     # search radius around every GPS point
MAX_CANDIDATES = 8            # keep the K nearest road segments inside the radius
DENSIFY_STEP_M = 10.0         # sample spacing used to index segments in the KD-tree

# Graph cropping: keep only road nodes within this margin of the trajectory bbox
GRAPH_MARGIN_M = 300.0

# Cross-validation
N_FOLDS = 5
