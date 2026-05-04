"""Tests for feature engineering — schema contracts and invariants."""

import pandas as pd

from src.features.feature_engineering import compute_features, split_features_target


def test_output_has_expected_columns(raw_transactions: pd.DataFrame) -> None:
    result, _ = compute_features(raw_transactions)
    assert "Amount_scaled" in result.columns
    assert "Time_scaled" in result.columns
    assert "Hour" in result.columns
    assert "Amount_log" in result.columns
    assert "Time" not in result.columns
    assert "Amount" not in result.columns


def test_no_nulls_after_transform(raw_transactions: pd.DataFrame) -> None:
    result, _ = compute_features(raw_transactions)
    assert result.isnull().sum().sum() == 0


def test_row_count_preserved(raw_transactions: pd.DataFrame) -> None:
    result, _ = compute_features(raw_transactions)
    assert len(result) == len(raw_transactions)


def test_hour_within_valid_range(raw_transactions: pd.DataFrame) -> None:
    result, _ = compute_features(raw_transactions)
    assert ((result["Hour"] >= 0) & (result["Hour"] < 24)).all()


def test_amount_log_non_negative(raw_transactions: pd.DataFrame) -> None:
    result, _ = compute_features(raw_transactions)
    assert (result["Amount_log"] >= 0).all()


def test_split_preserves_total_rows(engineered_features: pd.DataFrame) -> None:
    X, y = split_features_target(engineered_features)
    assert len(X) + len(y) == len(engineered_features) * 2  # both same length
    assert len(X) == len(y)


def test_target_not_in_features(engineered_features: pd.DataFrame) -> None:
    X, _ = split_features_target(engineered_features)
    assert "Class" not in X.columns


def test_target_is_binary(engineered_features: pd.DataFrame) -> None:
    _, y = split_features_target(engineered_features)
    assert set(y.unique()).issubset({0, 1})
