"""Shared utilities for the HEA heart disease analysis notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_curve, auc, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve

RANDOM_STATE = 42

# Column name groups (based on the processed dataset)
_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
_BINARY = ["sex", "fbs", "exang"]
_MULTICLASS = ["cp", "restecg", "thal", "slope"]


def load_processed_data() -> pd.DataFrame:
    """Load the clean processed CSV (no missing values expected)."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "processed", "heart_disease_processed.csv")
    df = pd.read_csv(path)
    return df


def get_feature_target_split(
    df: pd.DataFrame,
    drop_source: bool = True,
    drop_high_missing: bool = False,
) -> tuple:
    """Split DataFrame into features (X) and target (y).

    Parameters
    ----------
    drop_source : bool
        If True, exclude source_code from features.
    drop_high_missing : bool
        If True, also drop thal and slope (high-missingness columns).

    Returns
    -------
    X, y : pd.DataFrame, pd.Series
    """
    y = df["num"]
    X = df.drop(columns=["num"])
    if drop_source and "source_code" in X.columns:
        X = X.drop(columns=["source_code"])
    if drop_high_missing:
        to_drop = [c for c in ["thal", "slope"] if c in X.columns]
        X = X.drop(columns=to_drop)
    return X, y


def get_column_groups(X: pd.DataFrame) -> dict:
    """Classify columns in X into continuous, binary, and multiclass groups.

    Only includes columns that are actually present in X.
    """
    cols = set(X.columns)
    return {
        "continuous": [c for c in _CONTINUOUS if c in cols],
        "binary": [c for c in _BINARY if c in cols],
        "multiclass": [c for c in _MULTICLASS if c in cols]
            + (["source_code"] if "source_code" in cols else []),
    }


def build_preprocessor(column_groups: dict) -> ColumnTransformer:
    """Build a ColumnTransformer for feature encoding (no imputation).

    - Continuous: StandardScaler
    - Binary: passthrough
    - Multiclass: OneHotEncoder (drop='first')
    """
    transformers = []
    if column_groups["continuous"]:
        transformers.append(
            ("continuous", StandardScaler(), column_groups["continuous"])
        )
    if column_groups["binary"]:
        transformers.append(
            ("binary", "passthrough", column_groups["binary"])
        )
    if column_groups["multiclass"]:
        transformers.append(
            ("multiclass",
             OneHotEncoder(drop="first", sparse_output=False),
             column_groups["multiclass"])
        )
    return ColumnTransformer(transformers, remainder="drop")


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, ax=None, label=None, save_path=None):
    """Plot ROC curve and return the AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    lbl = f"{label} (AUC = {roc_auc:.3f})" if label else f"AUC = {roc_auc:.3f}"
    ax.plot(fpr, tpr, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    if save_path:
        ax.get_figure().savefig(save_path, dpi=150, bbox_inches="tight")
    return roc_auc


def plot_calibration_curve(y_true, y_prob, n_bins=10, ax=None, label=None,
                           save_path=None):
    """Plot calibration curve and return the Brier score."""
    brier = brier_score_loss(y_true, y_prob)
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    lbl = f"{label} (Brier = {brier:.3f})" if label else f"Brier = {brier:.3f}"
    ax.plot(mean_pred, fraction_pos, "s-", label=lbl)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    if save_path:
        ax.get_figure().savefig(save_path, dpi=150, bbox_inches="tight")
    return brier


def classification_report_df(y_true, y_pred, y_prob) -> pd.DataFrame:
    """Return a single-row DataFrame with key classification metrics."""
    return pd.DataFrame({
        "accuracy": [accuracy_score(y_true, y_pred)],
        "precision": [precision_score(y_true, y_pred)],
        "recall": [recall_score(y_true, y_pred)],
        "f1": [f1_score(y_true, y_pred)],
        "roc_auc": [roc_auc_score(y_true, y_prob)],
        "brier_score": [brier_score_loss(y_true, y_prob)],
    }).round(4)
