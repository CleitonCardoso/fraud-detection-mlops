"""Sklearn baseline models for fraud detection."""
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
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
