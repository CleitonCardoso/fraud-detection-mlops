"""Drift detection using Evidently with dual PSI thresholds."""
import logging
from pathlib import Path

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.monitoring.metrics import drift_psi_gauge

logger = logging.getLogger(__name__)

_config = yaml.safe_load(Path("configs/monitoring_config.yaml").read_text())
PSI_WARNING = _config["drift"]["psi_warning_threshold"]
PSI_RETRAIN = _config["drift"]["psi_retrain_threshold"]


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    Args:
        reference: Distribution from training data.
        current: Distribution from recent production data.
        bins: Number of buckets for discretization.

    Returns:
        PSI value. 0–0.1 stable, 0.1–0.2 warning, >0.2 retrain.
    """
    ref_counts, edges = pd.cut(reference, bins=bins, retbins=True)
    cur_counts = pd.cut(current, bins=edges)

    ref_pct = ref_counts.value_counts(normalize=True, sort=False) + 1e-6
    cur_pct = cur_counts.value_counts(normalize=True, sort=False) + 1e-6

    psi = ((cur_pct - ref_pct) * (cur_pct / ref_pct).apply(lambda x: x if x > 0 else 1e-6).apply(
        __import__("math").log
    )).sum()
    return float(abs(psi))


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "data/processed/drift_report.html",
) -> dict[str, float]:
    """Run full Evidently drift report and emit Prometheus + retrain signal.

    Args:
        reference_df: Training data used as baseline.
        current_df: Recent production data.
        output_path: Where to save the HTML report.

    Returns:
        Dictionary mapping feature name to PSI score.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info("Relatório Evidently salvo em %s", output_path)

    result = report.as_dict()
    drift_results: dict[str, float] = {}

    for metric in result.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            share = metric["result"].get("share_of_drifted_columns", 0.0)
            logger.info("Proporção de colunas com drift: %.2f", share)

        if metric.get("metric") == "ColumnDriftMetric":
            col = metric["result"].get("column_name", "unknown")
            psi_val = compute_psi(
                reference_df[col].dropna(),
                current_df[col].dropna(),
            ) if col in reference_df.columns and col in current_df.columns else 0.0

            drift_results[col] = psi_val
            drift_psi_gauge.labels(feature=col).set(psi_val)

            if psi_val > PSI_RETRAIN:
                logger.warning("RETRAIN TRIGGER — %s PSI=%.3f > %.1f", col, psi_val, PSI_RETRAIN)
            elif psi_val > PSI_WARNING:
                logger.warning("DRIFT WARNING — %s PSI=%.3f > %.1f", col, psi_val, PSI_WARNING)

    return drift_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ref = pd.read_parquet("data/processed/feature_store.parquet")
    cur = ref.sample(frac=0.1, random_state=99)
    scores = run_drift_report(ref, cur)
    for feat, psi in sorted(scores.items(), key=lambda x: -x[1])[:10]:
        print(f"{feat}: PSI={psi:.4f}")
