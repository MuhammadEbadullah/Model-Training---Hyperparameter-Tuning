"""Train baselines, tune with Grid/Randomized search, evaluate, save.

Usage (from project root):
    python -m src.tune            # full run
    python -m src.tune --quick    # fast smoke run (3-fold CV, few iterations)
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from . import reporting
from .config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from .data import load_data, split_data
from .models import SEARCH_SPACES, get_models
from .preprocessing import build_preprocessor

SCORING = {"roc_auc": "roc_auc", "f1": "f1", "pr_auc": "average_precision"}


def make_pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def test_metrics(est, X_test, y_test) -> dict:
    proba = est.predict_proba(X_test)[:, 1]
    pred = est.predict(X_test)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def main(quick: bool, cv_folds: int, n_iter_rf: int, n_iter_xgb: int) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train {X_train.shape} | Test {X_test.shape} | churn rate {y.mean():.3f}")

    if quick:
        cv_folds, n_iter_rf, n_iter_xgb = 3, 4, 6
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    n_iters = {"RandomForest": n_iter_rf, "XGBoost": n_iter_xgb}

    # ---------------------------------------------------------------- 1. baseline
    print("\n=== 1) Baseline: default hyper-parameters ===")
    default_cv, default_test = {}, {}
    for name, model in get_models().items():
        pipe = make_pipeline(model)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        default_cv[name] = {k: res[f"test_{k}"].mean() for k in SCORING}
        default_test[name] = test_metrics(pipe.fit(X_train, y_train), X_test, y_test)
        print(f"{name:20s} CV AUC {default_cv[name]['roc_auc']:.4f} | test AUC {default_test[name]['roc_auc']:.4f}")

    # ---------------------------------------------------------------- 2. tuning
    print("\n=== 2) Hyper-parameter optimisation ===")
    tuned, tuned_cv, tuned_test, best_params, timings, cv_tables = {}, {}, {}, {}, {}, {}
    for name, model in get_models().items():
        kind, space = SEARCH_SPACES[name]
        pipe = make_pipeline(model)
        common = dict(scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1)
        if kind == "grid":
            search = GridSearchCV(pipe, space, **common)
        else:
            search = RandomizedSearchCV(
                pipe, space, n_iter=n_iters[name], random_state=RANDOM_STATE, **common
            )
        t0 = time.time()
        search.fit(X_train, y_train)
        timings[name] = round(time.time() - t0, 1)

        i = search.best_index_
        tuned[name] = search.best_estimator_
        tuned_cv[name] = {k: search.cv_results_[f"mean_test_{k}"][i] for k in SCORING}
        tuned_test[name] = test_metrics(search.best_estimator_, X_test, y_test)
        best_params[name] = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()}
        cv_tables[name] = pd.DataFrame(search.cv_results_)
        cv_tables[name].to_csv(REPORTS_DIR / f"cv_results_{name}.csv", index=False)
        n_fits = len(cv_tables[name]) * cv_folds
        print(f"{name:20s} [{kind:6s}] {len(cv_tables[name]):3d} candidates / {n_fits:4d} fits / {timings[name]:6.1f}s "
              f"-> CV AUC {tuned_cv[name]['roc_auc']:.4f} | test AUC {tuned_test[name]['roc_auc']:.4f}")

    # ---------------------------------------------------------------- 3. summary
    summary = pd.DataFrame({
        "default_cv_auc": {n: default_cv[n]["roc_auc"] for n in tuned},
        "tuned_cv_auc": {n: tuned_cv[n]["roc_auc"] for n in tuned},
        "default_test_auc": {n: default_test[n]["roc_auc"] for n in tuned},
        "tuned_test_auc": {n: tuned_test[n]["roc_auc"] for n in tuned},
    })
    summary["cv_auc_gain"] = summary["tuned_cv_auc"] - summary["default_cv_auc"]
    summary["test_auc_gain"] = summary["tuned_test_auc"] - summary["default_test_auc"]
    summary["search_seconds"] = pd.Series(timings)
    print("\n", summary.round(4).to_string())
    summary.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")

    # Model selection uses CV score ONLY (the test set is never used to choose)
    best_name = summary["tuned_cv_auc"].idxmax()
    best_est = tuned[best_name]
    print(f"\nBest model by CV ROC-AUC: {best_name}")
    print(classification_report(y_test, best_est.predict(X_test), target_names=["Stay", "Churn"]))

    # ---------------------------------------------------------------- 4. persist
    metrics = {
        "best_model": best_name,
        "best_params": best_params,
        "default_cv": default_cv,
        "tuned_cv": tuned_cv,
        "default_test": default_test,
        "tuned_test": tuned_test,
        "search_seconds": timings,
        "settings": {"cv_folds": cv_folds, "n_iter_rf": n_iter_rf, "n_iter_xgb": n_iter_xgb,
                     "scoring_primary": "roc_auc", "random_state": RANDOM_STATE},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    joblib.dump(best_est, MODELS_DIR / "best_model.joblib")
    print(f"Saved models/best_model.joblib ({best_name})")

    # ---------------------------------------------------------------- 5. plots
    reporting.plot_model_comparison(summary, REPORTS_DIR / "model_comparison.png")
    reporting.plot_roc_all(tuned, X_test, y_test, REPORTS_DIR / "roc_curves.png")
    reporting.plot_confusion(best_est, X_test, y_test, best_name, REPORTS_DIR / "confusion_matrix.png")
    reporting.plot_feature_importance(best_est, REPORTS_DIR / "feature_importance.png")
    reporting.plot_param_effects(cv_tables["XGBoost"], REPORTS_DIR / "xgboost_param_effects.png")
    print("Saved plots to reports/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n-iter-rf", type=int, default=25)
    ap.add_argument("--n-iter-xgb", type=int, default=40)
    a = ap.parse_args()
    main(a.quick, a.cv, a.n_iter_rf, a.n_iter_xgb)
