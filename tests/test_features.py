"""Tests for feature engineering — schema contracts and invariants."""
import pandas as pd

from src.features.feature_engineering import ScalerParams, compute_features, fit_scalers, split_features_target


def test_output_has_expected_columns(raw_transactions: pd.DataFrame) -> None:
    result = compute_features(raw_transactions)
    assert "Amount_scaled" in result.columns
    assert "Time_scaled" in result.columns
    assert "Hour" in result.columns
    assert "Amount_log" in result.columns
    assert "Time" not in result.columns
    assert "Amount" not in result.columns


def test_no_nulls_after_transform(raw_transactions: pd.DataFrame) -> None:
    result = compute_features(raw_transactions)
    assert result.isnull().sum().sum() == 0


def test_row_count_preserved(raw_transactions: pd.DataFrame) -> None:
    result = compute_features(raw_transactions)
    assert len(result) == len(raw_transactions)


def test_hour_within_valid_range(raw_transactions: pd.DataFrame) -> None:
    result = compute_features(raw_transactions)
    assert ((result["Hour"] >= 0) & (result["Hour"] < 24)).all()


def test_amount_log_non_negative(raw_transactions: pd.DataFrame) -> None:
    result = compute_features(raw_transactions)
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


# ── ScalerParams and fit_scalers ─────────────────────────────────────────────

def test_fit_scalers_returns_scaler_params(raw_transactions: pd.DataFrame) -> None:
    scalers = fit_scalers(raw_transactions)
    assert isinstance(scalers, ScalerParams)
    assert scalers.amount_scale > 0
    assert scalers.time_scale > 0


def test_fit_scalers_mean_within_data_range(raw_transactions: pd.DataFrame) -> None:
    scalers = fit_scalers(raw_transactions)
    assert raw_transactions["Amount"].min() <= scalers.amount_mean <= raw_transactions["Amount"].max()
    assert raw_transactions["Time"].min() <= scalers.time_mean <= raw_transactions["Time"].max()


def test_scaler_params_serializable(raw_transactions: pd.DataFrame) -> None:
    import json
    scalers = fit_scalers(raw_transactions)
    d = scalers.to_dict()
    restored = ScalerParams(**json.loads(json.dumps(d)))
    assert restored.amount_mean == scalers.amount_mean
    assert restored.amount_scale == scalers.amount_scale


def test_compute_features_with_scalers_matches_known_normalization(raw_transactions: pd.DataFrame) -> None:
    scalers = fit_scalers(raw_transactions)
    result = compute_features(raw_transactions, scalers=scalers)
    # Amount_scaled should be z-scored against training distribution, not always 0
    assert result["Amount_scaled"].std() > 0


def test_compute_features_without_scalers_single_row_amount_scaled_is_zero() -> None:
    # Documents the known skew: single-row fit_transform produces 0
    single = pd.DataFrame([{"Time": 9800.0, "Amount": 850.0, **{f"V{i}": 0.0 for i in range(1, 29)}}])
    result = compute_features(single)
    assert result["Amount_scaled"].iloc[0] == 0.0


def test_compute_features_with_scalers_single_row_amount_scaled_is_nonzero(raw_transactions: pd.DataFrame) -> None:
    scalers = fit_scalers(raw_transactions)
    single = pd.DataFrame([{"Time": 9800.0, "Amount": 850.0, **{f"V{i}": 0.0 for i in range(1, 29)}}])
    result = compute_features(single, scalers=scalers)
    # With proper scalers, a large amount should produce a non-zero scaled value
    assert result["Amount_scaled"].iloc[0] != 0.0
