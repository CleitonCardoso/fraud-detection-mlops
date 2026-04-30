"""Tests for monitoring metrics and feature store."""
import pandas as pd
import pytest


def test_prometheus_metrics_importable():
    from src.monitoring.metrics import (
        agent_latency,
        drift_psi_gauge,
        fraud_score,
        model_auc_gauge,
        prediction_latency,
        request_counter,
    )
    assert prediction_latency is not None
    assert fraud_score is not None
    assert request_counter is not None
    assert agent_latency is not None
    assert drift_psi_gauge is not None
    assert model_auc_gauge is not None


def test_prediction_latency_accepts_observation():
    from src.monitoring.metrics import prediction_latency
    prediction_latency.observe(0.05)


def test_fraud_score_accepts_observation():
    from src.monitoring.metrics import fraud_score
    fraud_score.observe(0.87)


def test_request_counter_increments():
    from src.monitoring.metrics import request_counter
    request_counter.labels(endpoint="/test", status="ok").inc()


def test_drift_psi_gauge_set():
    from src.monitoring.metrics import drift_psi_gauge
    drift_psi_gauge.labels(feature="V14").set(0.15)


def test_model_auc_gauge_set():
    from src.monitoring.metrics import model_auc_gauge
    model_auc_gauge.set(0.97)


def test_feature_store_create_and_load(tmp_path, monkeypatch):
    import src.features.feature_store as fs
    monkeypatch.setattr(fs, "STORE_PATH", tmp_path / "feature_store.parquet")

    df = pd.DataFrame({"transaction_id": [1, 2, 3], "V1": [0.1, 0.2, 0.3]})
    fs.upsert_features(df, id_col="transaction_id")

    loaded = fs.load_features()
    assert len(loaded) == 3
    assert list(loaded["transaction_id"]) == [1, 2, 3]


def test_feature_store_upsert_deduplicates(tmp_path, monkeypatch):
    import src.features.feature_store as fs
    monkeypatch.setattr(fs, "STORE_PATH", tmp_path / "feature_store.parquet")

    df1 = pd.DataFrame({"transaction_id": [1, 2], "V1": [0.1, 0.2]})
    fs.upsert_features(df1, id_col="transaction_id")

    df2 = pd.DataFrame({"transaction_id": [2, 3], "V1": [0.99, 0.3]})
    fs.upsert_features(df2, id_col="transaction_id")

    loaded = fs.load_features()
    assert len(loaded) == 3
    assert float(loaded.loc[loaded["transaction_id"] == 2, "V1"].iloc[0]) == pytest.approx(0.99)


def test_feature_store_load_raises_when_missing(tmp_path, monkeypatch):
    import src.features.feature_store as fs
    monkeypatch.setattr(fs, "STORE_PATH", tmp_path / "nonexistent.parquet")

    with pytest.raises(FileNotFoundError):
        fs.load_features()
