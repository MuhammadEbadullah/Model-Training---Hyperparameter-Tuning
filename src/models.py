"""Model zoo + hyper-parameter search spaces.

* Randomized search  -> large / continuous spaces (XGBoost, Random Forest)
* Grid search        -> small, discrete spaces (Gradient Boosting, Logistic Regression)

All parameter names are prefixed with ``model__`` because the estimator is the
last step of a Pipeline named ``model``.
"""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import RANDOM_STATE

# Churn rate is ~26.5% -> negative/positive ratio ~2.77
IMBALANCE_RATIO = 2.77


def get_models() -> dict:
    """Default-parameter models (the baseline everyone should beat)."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1, verbosity=0
        ),
    }


# (search_type, parameter space)
SEARCH_SPACES = {
    "LogisticRegression": (
        "grid",
        {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "GradientBoosting": (
        "grid",
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
    ),
    "RandomForest": (
        "random",
        {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": [None, 5, 8, 12, 16],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", 0.3, 0.5],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "XGBoost": (
        "random",
        {
            "model__n_estimators": randint(100, 600),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_depth": randint(2, 8),
            "model__min_child_weight": randint(1, 10),
            "model__subsample": uniform(0.6, 0.4),         # 0.6 .. 1.0
            "model__colsample_bytree": uniform(0.5, 0.5),  # 0.5 .. 1.0
            "model__gamma": uniform(0, 5),
            "model__reg_alpha": loguniform(1e-3, 10),
            "model__reg_lambda": loguniform(1e-2, 10),
            "model__scale_pos_weight": [1, IMBALANCE_RATIO],
        },
    ),
}


# --------------------------------------------------------------------------- #
# XGBoost space v2: obtained by giving an AI assistant a fresh, explicit prompt
# (see docs/AI_GRID_SUGGESTION.md). Compared against SEARCH_SPACES["XGBoost"]
# (v1) in src/ai_grid_experiment.py.
# --------------------------------------------------------------------------- #
AI_SUGGESTED_XGB_SPACE = {
    "model__n_estimators": randint(200, 800),          # many shallow trees + small learning rate
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__max_depth": randint(3, 7),                 # 3..6: small tabular data -> shallow trees
    "model__min_child_weight": randint(1, 8),
    "model__subsample": uniform(0.7, 0.3),             # 0.7 .. 1.0
    "model__colsample_bytree": uniform(0.6, 0.4),      # 0.6 .. 1.0
    "model__gamma": uniform(0, 3),
    "model__reg_alpha": loguniform(1e-3, 5),
    "model__reg_lambda": loguniform(0.5, 20),          # stronger L2 than XGBoost's default (1)
    "model__scale_pos_weight": [1, 1.66, IMBALANCE_RATIO],  # none / sqrt(ratio) / full ratio
}
