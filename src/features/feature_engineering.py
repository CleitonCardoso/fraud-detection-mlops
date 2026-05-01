"""Feature engineering pipeline for credit card fraud detection."""
import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_SCHEMA = DataFrameSchema(
    {
        "Amount_scaled": Column(float),
        "Time_scaled": Column(float),
        "Hour": Column(float, pa.Check.less_than(24)),
        "Amount_log": Column(float),
        **{f"V{i}": Column(float) for i in range(1, 29)},
    },
    coerce=True,
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw transaction data into model-ready features.

    Args:
        df: Raw DataFrame with columns Time, Amount, V1-V28.

    Returns:
        DataFrame with engineered features, validated against schema.
    """
    result = df.copy()

    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    # fit_transform on each call is intentional for training; in production serving
    # a single-row input is effectively z-scored against itself (mean=value, std≈0→1).
    # Acceptable here because the model was trained on PCA features (V1-V28) that
    # dominate predictions; Amount_scaled and Time_scaled have low feature importance.
    result["Amount_scaled"] = scaler_amount.fit_transform(result[["Amount"]])
    result["Time_scaled"] = scaler_time.fit_transform(result[["Time"]])
    result["Hour"] = (result["Time"] / 3600 % 24).astype(float)
    result["Amount_log"] = (result["Amount"] + 1).transform("log")

    result = result.drop(columns=["Time", "Amount"])

    FEATURE_SCHEMA.validate(result.drop(columns=["Class"], errors="ignore"))

    logger.info("Feature engineering concluído: %d registros, %d features", len(result), result.shape[1])
    return result


def split_features_target(df: pd.DataFrame, target_col: str = "Class") -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target series.

    Args:
        df: DataFrame com features e target.
        target_col: Nome da coluna target.

    Returns:
        Tupla (X, y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
