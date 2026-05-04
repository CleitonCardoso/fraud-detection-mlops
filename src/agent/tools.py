"""Custom tools for the fraud detection ReAct agent."""

import json
import json as _json_module
import logging
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from langchain_core.tools import tool

from src.agent.rag_pipeline import retrieve

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_production_model():
    """Load the production model once and cache it for the process lifetime."""
    return mlflow.sklearn.load_model("models:/fraud_detector_rf@Production")


@tool
def fraud_predictor(transaction_json: str) -> str:
    """Predict fraud probability for a transaction and explain the top risk factors.

    Args:
        transaction_json: JSON string with transaction fields (Amount, Time, V1-V28).

    Returns:
        JSON string with fraud_score, label, and top SHAP feature contributions.
    """
    try:
        transaction = json.loads(transaction_json)
    except json.JSONDecodeError:
        return json.dumps(
            {"error": "transaction_json inválido — deve ser um JSON válido."}
        )

    try:
        model = _load_production_model()
    except Exception:
        return json.dumps(
            {
                "error": "Modelo não encontrado no MLflow Registry. Execute make train primeiro."
            }
        )

    from src.features.feature_engineering import ScalerParams, compute_features

    try:
        scalers_path = Path("data/processed/scalers.json")
        scalers = (
            ScalerParams(**_json_module.loads(scalers_path.read_text()))
            if scalers_path.exists()
            else None
        )
        features = compute_features(pd.DataFrame([transaction]), scalers=scalers).drop(
            columns=["Class"], errors="ignore"
        )
    except Exception as e:
        return json.dumps({"error": f"Erro no feature engineering: {e}"})

    proba = model.predict_proba(features)[0][1]
    label = "fraude" if proba >= 0.5 else "legítima"

    top_risk = []
    try:
        import shap

        shap_values = shap.TreeExplainer(model).shap_values(features)[1][0]
        top_risk = [
            {"feature": f, "contribution": round(float(v), 4)}
            for f, v in sorted(
                zip(features.columns, shap_values, strict=True),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
        ]
    except Exception:
        pass

    return json.dumps(
        {
            "fraud_score": round(float(proba), 4),
            "label": label,
            "top_risk_factors": top_risk,
        },
        ensure_ascii=False,
    )


@tool
def transaction_lookup(query: str) -> str:
    """Search the fraud knowledge base for relevant context about a query.

    Args:
        query: Natural language question about fraud patterns or the system.

    Returns:
        Relevant context chunks from the knowledge base.
    """
    chunks = retrieve(query, k=3)
    return "\n\n".join(f"• {chunk}" for chunk in chunks)


@tool
def drift_report(feature: str = "") -> str:
    """Return the latest drift status for a feature or overall system.

    Args:
        feature: Feature name to check (e.g. 'Amount_scaled'). Empty for overall summary.

    Returns:
        Human-readable drift status with PSI value and recommended action.
    """
    config_path = Path("configs/monitoring_config.yaml")
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        warning_threshold = config["drift"]["psi_warning_threshold"]
        retrain_threshold = config["drift"]["psi_retrain_threshold"]
    else:
        warning_threshold, retrain_threshold = 0.1, 0.2

    report_path = Path("data/processed/drift_report.html")
    if not report_path.exists():
        return "Relatório de drift não encontrado. Execute 'make drift' para gerar."

    return (
        f"Último relatório de drift gerado em {report_path.stat().st_mtime}.\n"
        f"Thresholds: warning PSI > {warning_threshold}, retrain PSI > {retrain_threshold}.\n"
        f"Relatório completo disponível em {report_path}.\n"
        f"Para análise de feature específica, consulte o dashboard Grafana em http://localhost:3000."
    )
