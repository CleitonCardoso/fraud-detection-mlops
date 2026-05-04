"""Tests for baseline models — determinism, output shape, prediction range."""

import pandas as pd
import pytest

from src.models.baseline import (
    evaluate,
    get_splits,
    train_logistic_regression,
    train_random_forest,
)


def test_logistic_regression_predicts_probabilities(feature_target_split):
    X, y = feature_target_split
    X_train, X_test, y_train, y_test = get_splits(X, y)
    model = train_logistic_regression(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_random_forest_predicts_probabilities(feature_target_split):
    X, y = feature_target_split
    X_train, X_test, y_train, y_test = get_splits(X, y)
    model = train_random_forest(X_train, y_train, n_estimators=10)
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_evaluate_returns_all_metrics(feature_target_split):
    X, y = feature_target_split
    X_train, X_test, y_train, y_test = get_splits(X, y)
    model = train_logistic_regression(X_train, y_train)
    y_proba = pd.Series(model.predict_proba(X_test)[:, 1])
    y_pred = pd.Series((y_proba >= 0.5).astype(int))
    metrics = evaluate(y_test, y_pred, y_proba)
    assert set(metrics.keys()) == {"auc", "precision", "recall", "f1"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_stratified_split_preserves_fraud_ratio(feature_target_split):
    X, y = feature_target_split
    X_train, X_test, y_train, y_test = get_splits(X, y)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_temporal_split_no_leakage(raw_transactions):
    from src.features.feature_engineering import compute_features, split_features_target

    df = compute_features(raw_transactions)
    X, y = split_features_target(df)
    time_series = raw_transactions["Time"]

    X_train, X_test, y_train, y_test = get_splits(X, y, time_series=time_series)

    train_max_time = time_series.loc[X_train.index].max()
    test_min_time = time_series.loc[X_test.index].min()
    assert train_max_time < test_min_time, "Vazamento temporal entre treino e teste"
    assert len(X_test) == max(1, int(len(X) * 0.2))


def test_mlp_predicts_probabilities(feature_target_split):
    pytest.importorskip("torch", reason="torch not installed")
    from src.models.mlp import predict_proba_mlp, train_mlp

    X, y = feature_target_split
    X_train, X_test, y_train, _ = get_splits(X, y)
    model = train_mlp(
        X_train.to_numpy(dtype="float32"), y_train.to_numpy(dtype="float32"), epochs=2
    )
    proba = predict_proba_mlp(model, X_test.to_numpy(dtype="float32"))
    assert proba.shape == (len(X_test),)
    assert (proba >= 0).all() and (proba <= 1).all()
