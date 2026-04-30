"""PyTorch MLP for fraud detection with class imbalance handling."""
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class FraudMLP(nn.Module):
    """Multi-layer perceptron for binary fraud classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.3) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> FraudMLP:
    """Train FraudMLP with weighted BCE loss to handle class imbalance.

    Args:
        X_train: Training features as numpy array.
        y_train: Training labels as numpy array.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.

    Returns:
        Trained FraudMLP model in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Treinando MLP em %s", device)

    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = FraudMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

    model.eval()
    return model


def predict_proba_mlp(model: FraudMLP, X: np.ndarray) -> np.ndarray:
    """Run inference and return fraud probabilities.

    Args:
        model: Trained FraudMLP in eval mode.
        X: Feature matrix as numpy array.

    Returns:
        1-D array of fraud probabilities.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_t)
        return torch.sigmoid(logits).cpu().numpy()
