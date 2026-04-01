"""Bottleneck dimension selection via AIC and bottleneck vector extraction."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class KSelectionResult:
    best_k: int
    aic_scores: Dict[int, float]
    losses: Dict[int, float]
    models: Dict[int, object]


def find_optimal_k(
    X: np.ndarray,
    y: np.ndarray,
    k_range: Sequence[int] = (1, 2, 3, 4),
    epochs_per_k: int = 100,
    n_seeds: int = 1,
) -> KSelectionResult:
    """Train SciNet for each bottleneck dimension and pick the best via AIC.

    AIC = N * log(MSE) + 2 * n_params

    Parameters
    ----------
    X:
        Input array of shape (N, input_dim).
    y:
        Target array of shape (N, output_dim).
    k_range:
        Sequence of bottleneck dimensions to try.
    epochs_per_k:
        Number of training epochs for each K.
    n_seeds:
        Number of random seeds to average over per K (best loss kept).

    Returns
    -------
    KSelectionResult
    """
    import torch
    from atlas.scinet.model import SciNet
    from atlas.scinet.trainer import TrainConfig, train_scinet

    input_dim = X.shape[1]
    output_dim = y.shape[1] if y.ndim > 1 else 1
    N = X.shape[0]

    aic_scores: Dict[int, float] = {}
    losses: Dict[int, float] = {}
    best_models: Dict[int, object] = {}

    config = TrainConfig(epochs=epochs_per_k)

    for k in k_range:
        best_loss = float("inf")
        best_model = None

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SciNet(input_dim=input_dim, bottleneck_dim=k, output_dim=output_dim)
            result = train_scinet(model, X, y, config)
            if result.final_loss < best_loss:
                best_loss = result.final_loss
                best_model = model

        n_params = sum(p.numel() for p in best_model.parameters())
        # AIC using log of MSE loss (final_loss is already MSE)
        mse = max(best_loss, 1e-12)  # avoid log(0)
        aic = N * np.log(mse) + 2 * n_params

        aic_scores[k] = float(aic)
        losses[k] = float(best_loss)
        best_models[k] = best_model

    best_k = min(aic_scores, key=aic_scores.__getitem__)

    return KSelectionResult(
        best_k=best_k,
        aic_scores=aic_scores,
        losses=losses,
        models=best_models,
    )


def extract_bottleneck_vectors(model, X: np.ndarray) -> np.ndarray:
    """Encode input data through the model bottleneck and return as numpy array.

    Parameters
    ----------
    model:
        A trained SciNet instance.
    X:
        Input array of shape (N, input_dim).

    Returns
    -------
    np.ndarray of shape (N, bottleneck_dim).
    """
    import torch

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        Z = model.encode(X_t)
    return Z.numpy()
