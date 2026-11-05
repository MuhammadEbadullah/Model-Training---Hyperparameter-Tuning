"""Plots used in the README / LinkedIn post."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_model_comparison(summary: pd.DataFrame, path):
    """Default vs tuned ROC-AUC (CV and hold-out test)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(summary))
    w = 0.38
    for ax, (d, t, title) in zip(
        axes,
        [("default_cv_auc", "tuned_cv_auc", "5-fold CV ROC-AUC (train set)"),
         ("default_test_auc", "tuned_test_auc", "Hold-out test ROC-AUC")],
    ):
        b1 = ax.bar(x - w / 2, summary[d], w, label="Default params")
        b2 = ax.bar(x + w / 2, summary[t], w, label="Tuned params")
        ax.bar_label(b1, fmt="%.3f", fontsize=8)
        ax.bar_label(b2, fmt="%.3f", fontsize=8)
        ax.set_xticks(x, summary.index, rotation=15)
        ax.set_title(title)
        ax.set_ylim(summary[[d, t]].min().min() - 0.03, summary[[d, t]].max().max() + 0.02)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_all(fitted: dict, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, est in fitted.items():
        RocCurveDisplay.from_estimator(est, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves - tuned models (hold-out)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(est, X_test, y_test, name, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        est, X_test, y_test, ax=ax, colorbar=False, display_labels=["Stay", "Churn"]
    )
    ax.set_title(f"{name} - confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(est, path, top_n=15):
    names = est[:-1].get_feature_names_out()
    model = est.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(model.coef_.ravel())
    s = pd.Series(imp, index=names).sort_values().tail(top_n)
    s.index = [i.split("__", 1)[-1] for i in s.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    s.plot.barh(ax=ax)
    ax.set_title("Top feature importances (best model)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_param_effects(cv_results: pd.DataFrame, path, params=None, title="XGBoost"):
    """Scatter of each hyper-parameter vs mean CV ROC-AUC (random-search trials)."""
    params = params or ["learning_rate", "max_depth", "n_estimators", "subsample"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, p in zip(axes.ravel(), params):
        col = f"param_model__{p}"
        ax.scatter(cv_results[col].astype(float), cv_results["mean_test_roc_auc"], alpha=0.7)
        if p == "learning_rate":
            ax.set_xscale("log")
        ax.set_xlabel(p)
        ax.set_ylabel("CV ROC-AUC")
    fig.suptitle(f"{title}: effect of each hyper-parameter ({len(cv_results)} trials)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
