"""FastAPI application for the fraud detection MLOps system."""
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from src.features.feature_engineering import ScalerParams, compute_features
from src.monitoring.metrics import (
    agent_latency,
    drift_psi_gauge,
    fraud_score,
    model_auc_gauge,
    prediction_latency,
    request_counter,
)
from src.security.guardrails import InputGuardrail, OutputGuardrail
from src.serving.auth import verify_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

input_guard = InputGuardrail()
output_guard = OutputGuardrail()

_model = None
_scalers: ScalerParams | None = None
_threshold: float = 0.25  # overwritten at startup from MLflow model tag

_SCALERS_PATH = "data/processed/scalers.json"
_DEFAULT_THRESHOLD = 0.25


def _load_scalers() -> None:
    global _scalers
    try:
        raw = json.loads(Path(_SCALERS_PATH).read_text())
        _scalers = ScalerParams(**raw)
        logger.info("Scaler params carregados de %s", _SCALERS_PATH)
    except Exception as e:
        logger.warning("Scaler params não disponíveis em %s: %s — Amount/Time serão z-scored por lote", _SCALERS_PATH, e)
        _scalers = None


def _load_threshold() -> None:
    """Read fraud_threshold tag from the @Production model version in MLflow."""
    global _threshold
    try:
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias("fraud_detector_rf", "Production")
        tag_value = mv.tags.get("fraud_threshold")
        if tag_value is not None:
            _threshold = float(tag_value)
            logger.info("Threshold carregado do MLflow: %.4f (model v%s)", _threshold, mv.version)
        else:
            _threshold = _DEFAULT_THRESHOLD
            logger.warning("Tag 'fraud_threshold' não encontrada no modelo — usando default %.4f", _DEFAULT_THRESHOLD)
    except Exception as e:
        _threshold = _DEFAULT_THRESHOLD
        logger.warning("Não foi possível carregar threshold do MLflow: %s — usando default %.4f", e, _DEFAULT_THRESHOLD)


def _load_model() -> None:
    global _model
    _load_scalers()
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        run = mlflow.search_runs(
            experiment_names=[os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")],
            filter_string="tags.model_name = 'fraud_detector_rf'",
            order_by=["metrics.auc DESC"],
            max_results=1,
        )
        if not run.empty:  # type: ignore[union-attr]
            model_auc_gauge.set(run.iloc[0].get("metrics.auc", 0.0))  # type: ignore[union-attr]
        _model = mlflow.sklearn.load_model("models:/fraud_detector_rf@Production")
        _load_threshold()
        logger.info("Modelo carregado do MLflow Registry")
    except Exception as e:
        logger.warning("Modelo não disponível no Registry: %s — usando fallback", e)
        _model = None
    _seed_drift_metrics()


def _seed_drift_metrics() -> None:
    try:
        cfg = yaml.safe_load(Path("configs/monitoring_config.yaml").read_text())
        features = cfg.get("drift", {}).get("features_to_monitor", ["Amount_scaled", "V14", "Hour"])
    except Exception:
        features = ["Amount_scaled", "V14", "Hour"]
    for feature in features:
        drift_psi_gauge.labels(feature=feature).set(0.0)


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

app.mount("/metrics/", make_asgi_app())


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request, exc: RequestValidationError) -> JSONResponse:
    # Starlette's default JSON encoder cannot serialize float nan/inf that may appear
    # in the 'input' field of Pydantic error details. Replace them with None.
    def _safe(v: object) -> object:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return v

    errors = [dict(e, input=_safe(e.get("input"))) for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def _json_decode_error_handler(request, exc: Exception) -> JSONResponse:
    import json as _json
    if isinstance(exc, _json.JSONDecodeError):
        return JSONResponse(status_code=422, content={"detail": "Request body must be valid JSON"})
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Schemas ──────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        # rejects float('nan') and float('inf') — prevents them reaching sklearn
        json_schema_extra={"example": {"Time": 9800.0, "Amount": 850.0, "V14": -6.5}},
    )

    Time: float = Field(..., ge=0, allow_inf_nan=False, description="Seconds elapsed since first transaction")
    Amount: float = Field(..., ge=0, allow_inf_nan=False, description="Transaction amount in BRL")
    V1: float = Field(default=0.0, allow_inf_nan=False)
    V2: float = Field(default=0.0, allow_inf_nan=False)
    V3: float = Field(default=0.0, allow_inf_nan=False)
    V4: float = Field(default=0.0, allow_inf_nan=False)
    V5: float = Field(default=0.0, allow_inf_nan=False)
    V6: float = Field(default=0.0, allow_inf_nan=False)
    V7: float = Field(default=0.0, allow_inf_nan=False)
    V8: float = Field(default=0.0, allow_inf_nan=False)
    V9: float = Field(default=0.0, allow_inf_nan=False)
    V10: float = Field(default=0.0, allow_inf_nan=False)
    V11: float = Field(default=0.0, allow_inf_nan=False)
    V12: float = Field(default=0.0, allow_inf_nan=False)
    V13: float = Field(default=0.0, allow_inf_nan=False)
    V14: float = Field(default=0.0, allow_inf_nan=False)
    V15: float = Field(default=0.0, allow_inf_nan=False)
    V16: float = Field(default=0.0, allow_inf_nan=False)
    V17: float = Field(default=0.0, allow_inf_nan=False)
    V18: float = Field(default=0.0, allow_inf_nan=False)
    V19: float = Field(default=0.0, allow_inf_nan=False)
    V20: float = Field(default=0.0, allow_inf_nan=False)
    V21: float = Field(default=0.0, allow_inf_nan=False)
    V22: float = Field(default=0.0, allow_inf_nan=False)
    V23: float = Field(default=0.0, allow_inf_nan=False)
    V24: float = Field(default=0.0, allow_inf_nan=False)
    V25: float = Field(default=0.0, allow_inf_nan=False)
    V26: float = Field(default=0.0, allow_inf_nan=False)
    V27: float = Field(default=0.0, allow_inf_nan=False)
    V28: float = Field(default=0.0, allow_inf_nan=False)


class PredictResponse(BaseModel):
    fraud_score: float
    label: str
    threshold: float


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


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict(request: TransactionRequest) -> PredictResponse:
    """Run fraud prediction on a single transaction."""
    request_counter.labels(endpoint="/predict", status="started").inc()
    start = time.perf_counter()

    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível. Execute make train.")

    try:
        features = compute_features(pd.DataFrame([request.model_dump()]), scalers=_scalers).drop(columns=["Class"], errors="ignore")
        score = float(_model.predict_proba(features)[0][1])
    except Exception as e:
        request_counter.labels(endpoint="/predict", status="error").inc()
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed — check server logs.") from e

    elapsed = time.perf_counter() - start
    prediction_latency.observe(elapsed)
    fraud_score.observe(score)
    request_counter.labels(endpoint="/predict", status="ok").inc()
    label = "fraude" if score >= _threshold else "legítima"
    logger.info("Predição: score=%.4f label=%s threshold=%.4f latency=%.3fs", score, label, _threshold, elapsed)

    return PredictResponse(
        fraud_score=round(score, 4),
        label=label,
        threshold=_threshold,
    )


@app.post("/agent/query", response_model=AgentResponse, dependencies=[Depends(verify_api_key)])
def agent_query(request: AgentRequest) -> AgentResponse:
    """Query the ReAct fraud analysis agent."""
    from src.agent.react_agent import query as agent_query_fn

    request_counter.labels(endpoint="/agent/query", status="started").inc()

    valid, reason = input_guard.validate(request.query)
    if not valid:
        request_counter.labels(endpoint="/agent/query", status="blocked").inc()
        raise HTTPException(status_code=400, detail=reason)

    start = time.perf_counter()
    try:
        result = agent_query_fn(input_guard.sanitize(request.query), model_name=request.model_name)
    except Exception as e:
        request_counter.labels(endpoint="/agent/query", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    elapsed = time.perf_counter() - start
    agent_latency.observe(elapsed)
    request_counter.labels(endpoint="/agent/query", status="ok").inc()
    logger.info("Agent query concluída em %.2fs com %d steps", elapsed, result["steps"])

    return AgentResponse(answer=output_guard.sanitize(result["answer"]), steps=result["steps"])
