"""Training pipeline with MLflow tracking and Model Registry."""
import json
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import yaml

from src.features.feature_engineering import (
    compute_features,
    fit_scalers,
    split_features_target,
)
from src.features.feature_store import upsert_features
from src.models.baseline import (
    evaluate,
    find_optimal_threshold,
    get_splits,
    train_logistic_regression,
    train_random_forest,
)
from src.monitoring.metrics import (
    champion_challenger_delta_gauge,
    retrain_triggered_total,
)

try:
    from src.models.mlp import predict_proba_mlp, train_mlp
except ImportError:
    train_mlp = None  # type: ignore[assignment]
    predict_proba_mlp = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _get_dvc_hash(path: str) -> str:
    dvc_file = Path(path + ".dvc")
    if not dvc_file.exists():
        return "unknown"
    meta = yaml.safe_load(dvc_file.read_text())
    return meta.get("outs", [{}])[0].get("md5", "unknown")  # type: ignore[no-any-return]


def _log_experiment(
    run_name: str,
    model: Any,
    params: dict,
    metrics: dict,
    tags: dict,
    log_model_fn: Callable,
    registered_model_name: str,
) -> None:
    """Log a single training run to MLflow — params, metrics, tags, and model artifact."""
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        run_tags = {**tags, "model_version": active_run.info.run_id[:12]}
        mlflow.set_tags(run_tags)
        log_model_fn(model, "model", registered_model_name=registered_model_name)
        logger.info("%s — AUC: %.4f  F1: %.4f", run_name, metrics["auc"], metrics["f1"])


def _base_tags(model_name: str, dvc_hash: str, owner: str, git_sha: str) -> dict:
    return {
        "model_name": model_name,
        "model_type": "classification",
        "training_data_version": dvc_hash,
        "owner": owner,
        "risk_level": "high",
        "fairness_checked": "false",
        "git_sha": git_sha,
    }


def _get_champion_auc(registered_model_name: str) -> float | None:
    """Return AUC of the current @Production model, or None if no champion exists."""
    try:
        client = mlflow.MlflowClient()
        alias_mv = client.get_model_version_by_alias(
            registered_model_name, "Production"
        )
        run = mlflow.get_run(alias_mv.run_id or "")
        return run.data.metrics.get("auc")  # type: ignore[no-any-return]
    except Exception:
        return None


def _promote_if_better(
    registered_model_name: str,
    challenger_auc: float,
    min_delta: float = 0.005,
) -> bool:
    """Promote challenger to @Production if it beats the champion by min_delta.

    Returns True if promoted.
    """
    client = mlflow.MlflowClient()
    champion_auc = _get_champion_auc(registered_model_name)

    if champion_auc is None:
        logger.info("Nenhum champion encontrado — promovendo challenger diretamente")
        delta = challenger_auc
    else:
        delta = challenger_auc - champion_auc
        logger.info(
            "Champion AUC=%.4f  Challenger AUC=%.4f  Delta=%.4f  (mínimo=%.4f)",
            champion_auc,
            challenger_auc,
            delta,
            min_delta,
        )

    champion_challenger_delta_gauge.set(delta)

    if champion_auc is None or delta >= min_delta:
        versions = client.search_model_versions(f"name='{registered_model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(
            registered_model_name, "Production", latest.version
        )
        logger.info("Challenger promovido a @Production (versão %s)", latest.version)
        retrain_triggered_total.labels(reason="drift", outcome="promoted").inc()
        return True

    logger.info("Challenger não promovido — delta insuficiente")
    retrain_triggered_total.labels(reason="drift", outcome="rejected").inc()
    return False


def run_training(data_path: str = "data/raw/creditcard.csv") -> None:
    """Full training pipeline: load → features → train → log → register."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))

    logger.info("Carregando dados de %s", data_path)
    df_raw = pd.read_csv(data_path)
    logger.info(
        "Dataset: %d linhas, fraudes: %.2f%%", len(df_raw), df_raw["Class"].mean() * 100
    )

    scalers = fit_scalers(df_raw)
    scaler_path = Path("data/processed/scalers.json")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_text(json.dumps(scalers.to_dict()))
    logger.info("Scaler params salvos em %s", scaler_path)

    df = compute_features(df_raw, scalers=scalers)
    upsert_features(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = get_splits(X, y, time_series=df_raw["Time"])

    train_time_window = (
        float(df_raw.loc[X_train.index, "Time"].min()),
        float(df_raw.loc[X_train.index, "Time"].max()),
    )
    test_time_window = (
        float(df_raw.loc[X_test.index, "Time"].min()),
        float(df_raw.loc[X_test.index, "Time"].max()),
    )
    logger.info("Split temporal: train=%s test=%s", train_time_window, test_time_window)

    split_metadata = {
        "split_strategy": "temporal",
        "train_time_min": train_time_window[0],
        "train_time_max": train_time_window[1],
        "test_time_min": test_time_window[0],
        "test_time_max": test_time_window[1],
    }

    git_sha = _get_git_sha()
    dvc_hash = _get_dvc_hash(data_path)
    owner = os.getenv("OWNER_EMAIL", "owner@example.com")

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    y_proba = pd.Series(lr_model.predict_proba(X_test)[:, 1])
    _log_experiment(
        run_name="logistic_regression",
        model=lr_model,
        params={
            "model_type": "logistic_regression",
            "class_weight": "balanced",
            "max_iter": 1000,
            **split_metadata,
        },
        metrics=evaluate(y_test, pd.Series((y_proba >= 0.5).astype(int)), y_proba),
        tags=_base_tags("fraud_detector_lr", dvc_hash, owner, git_sha),
        log_model_fn=mlflow.sklearn.log_model,
        registered_model_name="fraud_detector_lr",
    )

    # Random Forest — champion-challenger promotion
    rf_model = train_random_forest(X_train, y_train)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    optimal_threshold, threshold_metrics = find_optimal_threshold(y_test, pd.Series(y_proba))
    rf_metrics = evaluate(y_test, pd.Series((y_proba >= optimal_threshold).astype(int)), pd.Series(y_proba))
    rf_tags = {
        **_base_tags("fraud_detector_rf", dvc_hash, owner, git_sha),
        "fraud_threshold": str(round(optimal_threshold, 4)),
    }
    _log_experiment(
        run_name="random_forest",
        model=rf_model,
        params={
            "model_type": "random_forest",
            "n_estimators": 100,
            "class_weight": "balanced",
            **split_metadata,
        },
        metrics={**rf_metrics, **threshold_metrics},
        tags=rf_tags,
        log_model_fn=mlflow.sklearn.log_model,
        registered_model_name="fraud_detector_rf",
    )
    logger.info(
        "Threshold ótimo (F1): %.4f — precision=%.4f recall=%.4f f1=%.4f",
        optimal_threshold,
        threshold_metrics["precision_at_threshold"],
        threshold_metrics["recall_at_threshold"],
        threshold_metrics["f1_at_threshold"],
    )

    # Write threshold tag to the latest version and to @Production (may differ when promotion is skipped)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='fraud_detector_rf'")
    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        threshold_str = str(round(optimal_threshold, 4))
        client.set_model_version_tag(latest.name, latest.version, "fraud_threshold", threshold_str)
        try:
            prod_mv = client.get_model_version_by_alias("fraud_detector_rf", "Production")
            if prod_mv.version != latest.version:
                client.set_model_version_tag(prod_mv.name, prod_mv.version, "fraud_threshold", threshold_str)
                logger.info(
                    "Tag 'fraud_threshold=%s' escrita em v%s (latest) e v%s (@Production)",
                    threshold_str, latest.version, prod_mv.version,
                )
        except Exception:
            pass

    min_delta = yaml.safe_load(Path("configs/monitoring_config.yaml").read_text())["retraining"]["champion_min_delta_auc"]
    _promote_if_better("fraud_detector_rf", rf_metrics["auc"], min_delta=min_delta)

    # PyTorch MLP (skipped if torch is not available)
    try:
        if train_mlp is None:
            raise ImportError("torch not available")
        X_train_np = X_train.to_numpy(dtype="float32")
        X_test_np = X_test.to_numpy(dtype="float32")
        mlp_model = train_mlp(X_train_np, y_train.to_numpy(dtype="float32"))
        y_proba_np = predict_proba_mlp(mlp_model, X_test_np)
        _log_experiment(
            run_name="mlp_pytorch",
            model=mlp_model,
            params={
                "model_type": "mlp",
                "epochs": 20,
                "batch_size": 512,
                "lr": 1e-3,
                "hidden_dims": "[128,64,32]",
                **split_metadata,
            },
            metrics=evaluate(
                y_test,
                pd.Series((y_proba_np >= 0.5).astype(int)),
                pd.Series(y_proba_np),
            ),
            tags=_base_tags("fraud_detector_mlp", dvc_hash, owner, git_sha),
            log_model_fn=mlflow.pytorch.log_model,
            registered_model_name="fraud_detector_mlp",
        )
    except ImportError:
        logger.warning("torch não instalado — MLP ignorado. Execute: pip install torch")

    logger.info(
        "Treino completo. Acesse o MLflow em %s",
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )


if __name__ == "__main__":
    run_training()
