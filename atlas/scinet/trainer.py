"""SciNet training utilities."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-5
    encoder_sparsity: float = 0.0


@dataclass
class TrainResult:
    final_loss: float
    initial_loss: float
    loss_history: List[float]
    model: object


def train_scinet(model, X: np.ndarray, y: np.ndarray, config: TrainConfig | None = None) -> TrainResult:
    """Train a SciNet model using Adam optimizer and MSE loss.

    Parameters
    ----------
    model:
        A SciNet instance.
    X:
        Input array of shape (N, input_dim).
    y:
        Target array of shape (N, output_dim).
    config:
        Training configuration. Defaults to TrainConfig().

    Returns
    -------
    TrainResult
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    if config is None:
        config = TrainConfig()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    loss_history: List[float] = []
    initial_loss: float | None = None

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            z = model.encode(xb)
            pred = model.decode(z)
            loss = criterion(pred, yb)
            if config.encoder_sparsity > 0.0:
                loss = loss + config.encoder_sparsity * z.abs().mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if initial_loss is None:
            initial_loss = avg_loss

    final_loss = loss_history[-1] if loss_history else float("inf")
    if initial_loss is None:
        initial_loss = final_loss

    return TrainResult(
        final_loss=final_loss,
        initial_loss=initial_loss,
        loss_history=loss_history,
        model=model,
    )
