"""Drift detection using Evidently with dual PSI thresholds."""
import logging
import math
from pathlib import Path

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
        for r, c in zip(ref_pct, cur_pct)
        if r > 0 and c > 0
    )
    return float(abs(psi))


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "data/processed/drift_report.html",
) -> dict[str, float]:
    """Run Evidently drift report and emit Prometheus metrics + retrain signal."""
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Only use numeric columns present in both datasets
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

    # Compute PSI per feature and push to Prometheus
    drift_results: dict[str, float] = {}
    for col in cols:
        psi_val = compute_psi(ref_sub[col], cur_sub[col])
        drift_results[col] = psi_val
        drift_psi_gauge.labels(feature=col).set(psi_val)

        if psi_val > PSI_RETRAIN:
            logger.warning("RETRAIN TRIGGER — %s PSI=%.3f > %.1f", col, psi_val, PSI_RETRAIN)
        elif psi_val > PSI_WARNING:
            logger.warning("DRIFT WARNING  — %s PSI=%.3f > %.1f", col, psi_val, PSI_WARNING)
        else:
            logger.info("Stable           — %s PSI=%.3f", col, psi_val)

    return drift_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    ref = pd.read_parquet("data/processed/feature_store.parquet")
    # Simulate production drift: add Gaussian noise to 10% sample
    import numpy as np
    cur = ref.sample(frac=0.1, random_state=42).copy()
    for col in ["Amount_scaled", "V14"]:
        if col in cur.columns:
            cur[col] = cur[col] + np.random.normal(0, cur[col].std() * 0.5, len(cur))

    scores = run_drift_report(ref, cur)
    print("\n── PSI Results ──────────────────")
    for feat, psi in sorted(scores.items(), key=lambda x: -x[1]):
        status = "🔴 RETRAIN" if psi > PSI_RETRAIN else ("🟡 WARNING" if psi > PSI_WARNING else "🟢 OK")
        print(f"  {feat:20s} PSI={psi:.4f}  {status}")
