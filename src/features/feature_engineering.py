"""Feature engineering pipeline for credit card fraud detection."""

import logging
from dataclasses import asdict, dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ScalerParams:
    """Mean and scale for Amount and Time, fitted on the full training set."""

    amount_mean: float
    amount_scale: float
    time_mean: float
    time_scale: float

    def to_dict(self) -> dict:
        return asdict(self)


def fit_scalers(df: pd.DataFrame) -> ScalerParams:
    """Fit StandardScalers on a DataFrame and return their parameters."""
    sa = StandardScaler().fit(df[["Amount"]])
    st = StandardScaler().fit(df[["Time"]])
    return ScalerParams(
        amount_mean=float(sa.mean_[0]),
        amount_scale=float(sa.scale_[0]),
        time_mean=float(st.mean_[0]),
        time_scale=float(st.scale_[0]),
    )


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


def compute_features(
    df: pd.DataFrame, scalers: ScalerParams | None = None
) -> pd.DataFrame:
    """Transform raw transaction data into model-ready features.

    Args:
        df: Raw DataFrame with columns Time, Amount, V1-V28.
        scalers: Pre-fitted scaler params from training. When None (e.g. during
                 training itself), scalers are fitted on the input batch.

    Returns:
        DataFrame with engineered features, validated against schema.
    """
    result = df.copy()

    if scalers is not None:
        result["Amount_scaled"] = (
            result["Amount"] - scalers.amount_mean
        ) / scalers.amount_scale
        result["Time_scaled"] = (
            result["Time"] - scalers.time_mean
        ) / scalers.time_scale
    else:
        result["Amount_scaled"] = StandardScaler().fit_transform(result[["Amount"]])
        result["Time_scaled"] = StandardScaler().fit_transform(result[["Time"]])
    result["Hour"] = (result["Time"] / 3600 % 24).astype(float)
    result["Amount_log"] = (result["Amount"] + 1).transform("log")

    result = result.drop(columns=["Time", "Amount"])

    FEATURE_SCHEMA.validate(result.drop(columns=["Class"], errors="ignore"))

    logger.info(
        "Feature engineering concluído: %d registros, %d features",
        len(result),
        result.shape[1],
    )
    return result


def split_features_target(
    df: pd.DataFrame, target_col: str = "Class"
) -> tuple[pd.DataFrame, pd.Series]:
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
