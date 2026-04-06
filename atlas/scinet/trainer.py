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
    use_cosine_schedule: bool = True
    min_lr: float = 1e-5
    val_fraction: float = 0.0


@dataclass
class TrainResult:
    final_loss: float
    initial_loss: float
    loss_history: List[float]
    model: object
    val_loss_history: List[float] | None = None
    final_val_loss: float | None = None


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

    # Validation split
    val_loader = None
    if config.val_fraction > 0.0:
        n_val = max(1, int(len(X_t) * config.val_fraction))
        n_train = len(X_t) - n_val
        perm = torch.randperm(len(X_t))
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
        val_ds = TensorDataset(X_t[val_idx], y_t[val_idx])
        loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    else:
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    scheduler = None
    if config.use_cosine_schedule:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=config.min_lr,
        )

    loss_history: List[float] = []
    val_loss_history: List[float] = [] if val_loader is not None else None
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

        if scheduler is not None:
            scheduler.step(epoch)

        # Validation loss
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item()
                    val_batches += 1
            val_loss_history.append(val_loss / max(val_batches, 1))
            model.train()

    final_loss = loss_history[-1] if loss_history else float("inf")
    if initial_loss is None:
        initial_loss = final_loss

    final_val_loss = val_loss_history[-1] if val_loss_history else None

    return TrainResult(
        final_loss=final_loss,
        initial_loss=initial_loss,
        loss_history=loss_history,
        model=model,
        val_loss_history=val_loss_history,
        final_val_loss=final_val_loss,
    )
