"""Drift detection using Evidently with dual PSI thresholds."""
import json
import logging
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.monitoring.metrics import drift_psi_gauge

logger = logging.getLogger(__name__)

_config = yaml.safe_load(Path("configs/monitoring_config.yaml").read_text())
PSI_WARNING = _config["drift"]["psi_warning_threshold"]
PSI_RETRAIN = _config["drift"]["psi_retrain_threshold"]
FEATURES_TO_MONITOR = _config["drift"].get(
    "features_to_monitor", ["Amount_scaled", "V14", "Hour", "Amount_log1p"]
)

DRIFT_STATUS_PATH = "data/processed/drift_status.json"


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    Returns:
        PSI value. 0–0.1 stable, 0.1–0.2 warning, >0.2 retrain.
    """
    ref_counts, edges = pd.cut(reference, bins=bins, retbins=True)
    cur_counts = pd.cut(current, bins=edges)

    ref_pct = ref_counts.value_counts(normalize=True, sort=False) + 1e-6
    cur_pct = cur_counts.value_counts(normalize=True, sort=False) + 1e-6

    psi = sum(
        (c - r) * math.log(c / r)
        for r, c in zip(ref_pct, cur_pct, strict=False)
        if r > 0 and c > 0
    )
    return float(abs(psi))


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "data/processed/drift_report.html",
) -> dict:
    """Run Evidently drift report, emit Prometheus metrics, log to MLflow.

    Returns:
        Dict with feature PSI values plus 'retrain_needed' bool.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in FEATURES_TO_MONITOR if c in reference_df.columns and c in current_df.columns]
    ref_sub = reference_df[cols].dropna()
    cur_sub = current_df[cols].dropna()

    data_def = DataDefinition(numerical_columns=cols)
    ref_ds = Dataset.from_pandas(ref_sub, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur_sub, data_definition=data_def)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
    snapshot.save_html(output_path)
    logger.info("Relatório Evidently salvo em %s", output_path)

    drift_results: dict[str, float] = {}
    retrain_needed = False

    for col in cols:
        psi_val = compute_psi(ref_sub[col], cur_sub[col])
        drift_results[col] = psi_val
        drift_psi_gauge.labels(feature=col).set(psi_val)

        if psi_val > PSI_RETRAIN:
            retrain_needed = True
            logger.warning("RETRAIN TRIGGER — %s PSI=%.3f > %.1f", col, psi_val, PSI_RETRAIN)
        elif psi_val > PSI_WARNING:
            logger.warning("DRIFT WARNING  — %s PSI=%.3f > %.1f", col, psi_val, PSI_WARNING)
        else:
            logger.info("Stable           — %s PSI=%.3f", col, psi_val)

    _log_drift_to_mlflow(drift_results, retrain_needed)
    _write_drift_status(drift_results, retrain_needed)

    return {**drift_results, "retrain_needed": retrain_needed}


def _log_drift_to_mlflow(psi_results: dict[str, float], retrain_needed: bool) -> None:
    """Log PSI values and retrain flag to an MLflow run for historical tracking."""
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))
        with mlflow.start_run(run_name="drift_check"):
            mlflow.log_metrics({f"psi_{k}": v for k, v in psi_results.items()})
            mlflow.log_metric("retrain_needed", int(retrain_needed))
            mlflow.set_tag("run_type", "drift_check")
        logger.info("PSI values logged to MLflow")
    except Exception as e:
        logger.warning("MLflow drift logging falhou (non-fatal): %s", e)


def _write_drift_status(psi_results: dict[str, float], retrain_needed: bool) -> None:
    """Write drift_status.json for downstream consumers (e.g. GitHub Actions)."""
    status = {
        "timestamp": datetime.now(UTC).isoformat(),
        "retrain_needed": retrain_needed,
        "psi": psi_results,
        "thresholds": {"warning": PSI_WARNING, "retrain": PSI_RETRAIN},
    }
    Path(DRIFT_STATUS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(DRIFT_STATUS_PATH).write_text(json.dumps(status, indent=2))
    logger.info("drift_status.json escrito em %s", DRIFT_STATUS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    ref = pd.read_parquet("data/processed/feature_store.parquet")
    import numpy as np
    cur = ref.sample(frac=0.1, random_state=42).copy()
    for col in ["Amount_scaled", "V14"]:
        if col in cur.columns:
            cur[col] = cur[col] + np.random.normal(0, cur[col].std() * 0.5, len(cur))

    results = run_drift_report(ref, cur)
    print("\n── PSI Results ──────────────────")
    for feat, psi in sorted(
        ((k, v) for k, v in results.items() if k != "retrain_needed"),
        key=lambda x: -x[1],
    ):
        status = "RETRAIN" if psi > PSI_RETRAIN else ("WARNING" if psi > PSI_WARNING else "OK")
        print(f"  {feat:20s} PSI={psi:.4f}  {status}")

    if results["retrain_needed"]:
        print("\nRetrain necessário — saindo com código 1")
        sys.exit(1)
