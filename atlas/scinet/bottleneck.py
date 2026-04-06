"""Bottleneck dimension selection via AIC/validation and bottleneck vector extraction."""
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
    val_losses: Dict[int, float] = field(default_factory=dict)
    selection_method: str = "aic"


def find_optimal_k(
    X: np.ndarray,
    y: np.ndarray,
    k_range: Sequence[int] = (1, 2, 3, 4),
    epochs_per_k: int = 100,
    n_seeds: int = 3,
    encoder_hidden: Optional[List[int]] = None,
    decoder_hidden: Optional[List[int]] = None,
    train_config: Optional[object] = None,
    val_fraction: float = 0.2,
    selection_method: str = "val_loss",
    bottleneck_activation: str = "none",
) -> KSelectionResult:
    """Train SciNet for each bottleneck dimension and pick the best.

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
    encoder_hidden:
        Hidden layer sizes for the encoder.
    decoder_hidden:
        Hidden layer sizes for the decoder.
    train_config:
        Optional TrainConfig; overrides *epochs_per_k* when supplied.
    val_fraction:
        Fraction of data to hold out for validation-based selection.
        Only used when *selection_method* is ``"val_loss"`` or ``"elbow"``.
    selection_method:
        ``"val_loss"`` — pick K that minimizes validation MSE (default).
        ``"aic"`` — classic AIC = N*log(MSE) + 2*n_params.
        ``"elbow"`` — pick K where val loss improvement drops below 5%.
    bottleneck_activation:
        Activation on bottleneck: ``"none"``, ``"tanh"``, or ``"sigmoid"``.

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

    # For validation-based methods, split data once (shared across all K)
    use_val = selection_method in ("val_loss", "elbow") and val_fraction > 0
    if use_val:
        n_val = max(1, int(N * val_fraction))
        perm = np.random.RandomState(42).permutation(N)
        train_idx, val_idx = perm[n_val:], perm[:n_val]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        N_train = len(X_train)
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
        N_train = N

    # Build train config — force encoder_sparsity=0 during K selection
    if train_config is not None:
        config = train_config
    else:
        config = TrainConfig(epochs=epochs_per_k)
    # Override: no sparsity during K selection (biases toward small K)
    config = TrainConfig(
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        encoder_sparsity=0.0,
        use_cosine_schedule=config.use_cosine_schedule,
        min_lr=config.min_lr,
        val_fraction=0.0,  # we handle val split ourselves
    )

    aic_scores: Dict[int, float] = {}
    losses: Dict[int, float] = {}
    val_losses: Dict[int, float] = {}
    best_models: Dict[int, object] = {}

    for k in k_range:
        best_loss = float("inf")
        best_val_loss = float("inf")
        best_model = None

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model = SciNet(input_dim=input_dim, bottleneck_dim=k, output_dim=output_dim,
                           encoder_hidden=encoder_hidden, decoder_hidden=decoder_hidden,
                           bottleneck_activation=bottleneck_activation)
            result = train_scinet(model, X_train, y_train, config)

            # Evaluate on validation set if available
            if use_val:
                model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32)
                    y_v = torch.tensor(y_val, dtype=torch.float32)
                    pred = model(X_v)
                    v_loss = float(torch.nn.functional.mse_loss(pred, y_v).item())
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_loss = result.final_loss
                    best_model = model
            else:
                if result.final_loss < best_loss:
                    best_loss = result.final_loss
                    best_model = model

        n_params = sum(p.numel() for p in best_model.parameters())
        mse = max(best_loss, 1e-12)
        aic = N_train * np.log(mse) + 2 * n_params

        aic_scores[k] = float(aic)
        losses[k] = float(best_loss)
        val_losses[k] = float(best_val_loss) if use_val else float(best_loss)
        best_models[k] = best_model

    # Select best K based on method
    if selection_method == "val_loss" and use_val:
        best_k = min(val_losses, key=val_losses.__getitem__)
    elif selection_method == "elbow" and use_val:
        sorted_ks = sorted(k_range)
        best_k = sorted_ks[0]
        for i in range(1, len(sorted_ks)):
            prev_k, curr_k = sorted_ks[i - 1], sorted_ks[i]
            prev_vl = val_losses[prev_k]
            curr_vl = val_losses[curr_k]
            rel_improvement = (prev_vl - curr_vl) / max(prev_vl, 1e-12)
            if rel_improvement > 0.05:
                best_k = curr_k
            else:
                break
    else:
        best_k = min(aic_scores, key=aic_scores.__getitem__)

    return KSelectionResult(
        best_k=best_k,
        aic_scores=aic_scores,
        losses=losses,
        models=best_models,
        val_losses=val_losses,
        selection_method=selection_method,
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
