"""Smoke tests for the agent tools — no real LLM calls."""
from unittest.mock import MagicMock, patch
import json

import pytest

langchain = pytest.importorskip("langchain", reason="langchain not installed — skipping agent tests")


def test_transaction_lookup_returns_strings():
    with patch("src.agent.tools.retrieve") as mock_retrieve:
        mock_retrieve.return_value = ["Chunk A sobre fraude.", "Chunk B sobre PSI."]
        from src.agent.tools import transaction_lookup
        result = transaction_lookup.invoke("O que é PSI?")
    assert isinstance(result, str)
    assert "Chunk" in result


def test_drift_report_returns_message_when_no_report(tmp_path):
    with patch("src.agent.tools.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        from src.agent.tools import drift_report
        result = drift_report.invoke("")
    assert isinstance(result, str)


def test_fraud_predictor_returns_error_without_model():
    with patch("src.agent.tools.mlflow.sklearn.load_model", side_effect=Exception("no model")):
        from src.agent.tools import fraud_predictor
        tx = json.dumps({"Time": 100.0, "Amount": 50.0, **{f"V{i}": 0.0 for i in range(1, 29)}})
        result = fraud_predictor.invoke(tx)
    data = json.loads(result)
    assert "error" in data


def test_fraud_predictor_rejects_invalid_json():
    from src.agent.tools import fraud_predictor
    result = fraud_predictor.invoke("not valid json")
    data = json.loads(result)
    assert "error" in data
