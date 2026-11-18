"""Model zoo: every model experimented with, with its initial configuration.

All models are binary classifiers on (observation, candidate) rows and expose
predict_proba, so their outputs can be used directly as HMM emissions.
"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_SEED

# family, short description, factory, max training rows (None = all; subsampling
# is only used where the algorithm's cost would otherwise be impractical)
MODEL_SPECS = {
    "LogisticRegression": dict(
        family="Linear / probabilistic",
        config="StandardScaler + L2 logistic regression, C=1.0, lbfgs, max_iter=1000",
        factory=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        max_rows=None),
    "GaussianNB": dict(
        family="Linear / probabilistic",
        config="Gaussian Naive Bayes, var_smoothing=1e-9",
        factory=lambda: GaussianNB(),
        max_rows=None),
    "KNN": dict(
        family="Instance-based",
        config="StandardScaler + KNN k=25, distance-weighted, Euclidean; fit on 100k-row sample",
        factory=lambda: make_pipeline(StandardScaler(),
                                      KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        max_rows=100_000),
    "DecisionTree": dict(
        family="Tree-based",
        config="CART, gini, max_depth=12, min_samples_leaf=50",
        factory=lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=RANDOM_SEED),
        max_rows=None),
    "RandomForest": dict(
        family="Ensemble (bagging)",
        config="100 trees, max_depth=18, min_samples_leaf=5, max_features=sqrt, max_samples=0.25",
        factory=lambda: RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=5,
                                               max_samples=0.25, n_jobs=-1, random_state=RANDOM_SEED),
        max_rows=None),
    "HistGradientBoosting": dict(
        family="Boosting",
        config="300 iterations, learning_rate=0.1, max_leaf_nodes=31, l2=0, log-loss",
        factory=lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                       early_stopping=False, random_state=RANDOM_SEED),
        max_rows=None),
    "MLP": dict(
        family="Neural network",
        config="StandardScaler + MLP (64,32) ReLU, Adam lr=1e-3, alpha=1e-4, batch=1024, 20 epochs; fit on 300k-row sample",
        factory=lambda: make_pipeline(StandardScaler(),
                                      MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                                    learning_rate_init=1e-3, max_iter=20, random_state=RANDOM_SEED)),
        max_rows=300_000),
}
