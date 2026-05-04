"""Sklearn baseline models for fraud detection."""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, precision_recall_curve, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def evaluate(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Dictionary with auc, precision, recall, f1.
    """
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_optimal_threshold(y_true: pd.Series, y_proba: pd.Series) -> tuple[float, dict[str, float]]:
    """Find the threshold that maximises F1 on the given split.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[min(best_idx, len(thresholds) - 1)])
    y_pred = (y_proba >= best_t).astype(int)
    return best_t, {
        "threshold": best_t,
        "precision_at_threshold": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_threshold": recall_score(y_true, y_pred, zero_division=0),
        "f1_at_threshold": f1_score(y_true, y_pred, zero_division=0),
    }


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """Train a logistic regression baseline with class weighting.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression treinado")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with class weighting.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of trees.

    Returns:
        Fitted RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest treinado com %d árvores", n_estimators)
    return model


def get_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving fraud ratio.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Fraction of data for testing.
        random_state: Reproducibility seed.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)  # type: ignore[no-any-return]
