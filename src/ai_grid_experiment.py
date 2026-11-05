"""Compare two XGBoost search spaces under IDENTICAL settings.

v1 = original space,  v2 = space suggested by an AI assistant (docs/AI_GRID_SUGGESTION.md).
A single random search is noisy, so each space is run with 3 different search seeds.

    python -m src.ai_grid_experiment
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import RANDOM_STATE, REPORTS_DIR
from .data import load_data, split_data
from .models import AI_SUGGESTED_XGB_SPACE, SEARCH_SPACES, get_models
from .tune import SCORING, make_pipeline, test_metrics

N_ITER = 40
SEARCH_SEEDS = [0, 1, 2]
SPACES = {"v1 original": SEARCH_SPACES["XGBoost"][1], "v2 AI-suggested": AI_SUGGESTED_XGB_SPACE}


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    rows, all_trials = [], {name: [] for name in SPACES}
    for name, space in SPACES.items():
        for seed in SEARCH_SEEDS:
            t0 = time.time()
            search = RandomizedSearchCV(
                make_pipeline(get_models()["XGBoost"]), space, n_iter=N_ITER, random_state=seed,
                scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            res = pd.DataFrame(search.cv_results_)
            all_trials[name].extend(res["mean_test_roc_auc"].tolist())
            top10 = res["mean_test_roc_auc"].nlargest(10).mean()
            row = {
                "space": name, "search_seed": seed,
                "best_cv_auc": search.best_score_,
                "top10_mean_cv_auc": top10,
                "median_trial_cv_auc": res["mean_test_roc_auc"].median(),
                "test_auc": test_metrics(search.best_estimator_, X_test, y_test)["roc_auc"],
                "seconds": time.time() - t0,
            }
            rows.append(row)
            print(f"{name:16s} seed {seed}: best CV {row['best_cv_auc']:.4f} | top10 {top10:.4f} | "
                  f"median trial {row['median_trial_cv_auc']:.4f} | test {row['test_auc']:.4f}")

    df = pd.DataFrame(rows)
    summary = df.groupby("space")[["best_cv_auc", "top10_mean_cv_auc", "median_trial_cv_auc", "test_auc"]].agg(["mean", "std"]).round(4)
    print("\n", summary.to_string())
    df.round(4).to_csv(REPORTS_DIR / "ai_grid_comparison.csv", index=False)
    (REPORTS_DIR / "ai_grid_comparison.json").write_text(
        json.dumps({"runs": rows, "n_iter": N_ITER, "search_seeds": SEARCH_SEEDS}, indent=2, default=float)
    )

    pd.DataFrame({k: pd.Series(v) for k, v in all_trials.items()}).round(5).to_csv(
        REPORTS_DIR / "ai_grid_all_trials.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot([all_trials[n] for n in SPACES], showfliers=True)
    ax.set_xticklabels(list(SPACES))  # works on every matplotlib version
    ax.set_ylabel("CV ROC-AUC of every trial (3 seeds x 40 trials)")
    ax.set_title("How good is a random draw from each search space?")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ai_grid_comparison.png", dpi=150)
    print("saved reports/ai_grid_comparison.*")


if __name__ == "__main__":
    main()
