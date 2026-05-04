"""Shared fixtures for all tests — synthetic data only, never real data."""

import numpy as np
import pandas as pd
import pytest
from faker import Faker

fake = Faker(locale="pt_BR")
Faker.seed(42)
np.random.seed(42)


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Synthetic credit card transactions mimicking the Kaggle dataset schema."""
    n = 200
    n_fraud = 10

    legit = pd.DataFrame(
        {
            "Time": np.random.uniform(0, 172800, n - n_fraud),
            "Amount": np.abs(np.random.normal(50, 80, n - n_fraud)),
            **{f"V{i}": np.random.normal(0, 1, n - n_fraud) for i in range(1, 29)},
            "Class": 0,
        }
    )

    fraud = pd.DataFrame(
        {
            "Time": np.random.uniform(0, 172800, n_fraud),
            "Amount": np.abs(np.random.normal(300, 200, n_fraud)),
            **{f"V{i}": np.random.normal(0, 3, n_fraud) for i in range(1, 29)},
            "Class": 1,
        }
    )

    return (
        pd.concat([legit, fraud], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


@pytest.fixture
def engineered_features(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Pre-computed features from raw transactions."""
    from src.features.feature_engineering import compute_features

    df, _ = compute_features(raw_transactions)
    return df


@pytest.fixture
def feature_target_split(
    engineered_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """X, y split from engineered features."""
    from src.features.feature_engineering import split_features_target

    return split_features_target(engineered_features)


@pytest.fixture
def sample_query() -> str:
    return "Esta transação de R$850 às 3h da manhã é suspeita?"


@pytest.fixture
def sample_transaction() -> dict:
    return {
        "Time": 9800.0,
        "Amount": 850.0,
        **{f"V{i}": float(np.random.normal(0, 1)) for i in range(1, 29)},
    }
