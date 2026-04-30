"""FastAPI application for the fraud detection MLOps system."""
import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from src.monitoring.metrics import (
    agent_latency,
    fraud_score,
    model_auc_gauge,
    prediction_latency,
    request_counter,
)
from src.security.guardrails import InputGuardrail, OutputGuardrail

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

input_guard = InputGuardrail()
output_guard = OutputGuardrail()

_model = None


def _load_model():
    global _model
    try:
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(uri)
        _model = mlflow.sklearn.load_model("models:/fraud_detector_rf@Production")
        logger.info("Modelo carregado do MLflow Registry")
    except Exception as e:
        logger.warning("Modelo não disponível no Registry: %s — usando fallback", e)
        _model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="MLOps system for credit card fraud detection — FIAP MLET Datathon Phase 05",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())


# ── Schemas ──────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in BRL")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

    model_config = {"json_schema_extra": {"example": {"Time": 9800.0, "Amount": 850.0, "V14": -6.5}}}


class PredictResponse(BaseModel):
    fraud_score: float
    label: str
    threshold: float = 0.5


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4096)
    model_name: str = Field(default="gpt-4o-mini")


class AgentResponse(BaseModel):
    answer: str
    steps: int


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: TransactionRequest) -> PredictResponse:
    """Run fraud prediction on a single transaction."""
    request_counter.labels(endpoint="/predict", status="started").inc()
    start = time.perf_counter()

    import pandas as pd
    from src.features.feature_engineering import compute_features

    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível. Execute make train.")

    try:
        df = pd.DataFrame([request.model_dump()])
        features = compute_features(df).drop(columns=["Class"], errors="ignore")
        score = float(_model.predict_proba(features)[0][1])
    except Exception as e:
        request_counter.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    elapsed = time.perf_counter() - start
    prediction_latency.observe(elapsed)
    fraud_score.observe(score)
    request_counter.labels(endpoint="/predict", status="ok").inc()

    logger.info("Predição: score=%.4f label=%s latency=%.3fs", score, "fraude" if score >= 0.5 else "legítima", elapsed)

    return PredictResponse(
        fraud_score=round(score, 4),
        label="fraude" if score >= 0.5 else "legítima",
    )


@app.post("/agent/query", response_model=AgentResponse)
def agent_query(request: AgentRequest) -> AgentResponse:
    """Query the ReAct fraud analysis agent."""
    request_counter.labels(endpoint="/agent/query", status="started").inc()

    valid, reason = input_guard.validate(request.query)
    if not valid:
        request_counter.labels(endpoint="/agent/query", status="blocked").inc()
        raise HTTPException(status_code=400, detail=reason)

    sanitized_query = input_guard.sanitize(request.query)

    from src.agent.react_agent import query as agent_query_fn

    start = time.perf_counter()
    try:
        result = agent_query_fn(sanitized_query, model_name=request.model_name)
    except Exception as e:
        request_counter.labels(endpoint="/agent/query", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    elapsed = time.perf_counter() - start
    agent_latency.observe(elapsed)

    safe_answer = output_guard.sanitize(result["answer"])
    request_counter.labels(endpoint="/agent/query", status="ok").inc()

    logger.info("Agent query concluída em %.2fs com %d steps", elapsed, result["steps"])

    return AgentResponse(answer=safe_answer, steps=result["steps"])
