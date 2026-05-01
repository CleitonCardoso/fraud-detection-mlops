"""Feature store with incremental materialization — never full-flush."""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STORE_PATH = Path("data/processed/feature_store.parquet")


def upsert_features(new_features: pd.DataFrame, id_col: str = "transaction_id") -> None:
    """Incrementally upsert features into the store — never overwrites everything.

    Args:
        new_features: DataFrame with new or updated feature rows.
        id_col: Column used as unique identifier for upsert logic.
    """
    if STORE_PATH.exists():
        existing = pd.read_parquet(STORE_PATH)
        combined = pd.concat([existing, new_features])
        if id_col in combined.columns:
            combined = combined.drop_duplicates(subset=[id_col], keep="last")
        logger.info(
            "Upsert: %d existentes + %d novos = %d total",
            len(existing), len(new_features), len(combined),
        )
        result = combined
    else:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Feature store criado com %d registros", len(new_features))
        result = new_features

    result.to_parquet(STORE_PATH, index=False)


def load_features() -> pd.DataFrame:
    """Load all features from the store.

    Returns:
        Full feature DataFrame from the store.

    Raises:
        FileNotFoundError: If the store has not been materialized yet.
    """
    if not STORE_PATH.exists():
        raise FileNotFoundError(f"Feature store não encontrado em {STORE_PATH}. Execute make train primeiro.")
    df = pd.read_parquet(STORE_PATH)
    logger.info("Feature store carregado: %d registros", len(df))
    return df
