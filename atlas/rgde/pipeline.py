"""RGDE Pipeline: full orchestration of Steps 4a-4f."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
from atlas.dsl.expr import Expr
from atlas.rgde.type_builder import DSLType, build_type
from atlas.rgde.constraint_finder import find_constraints
from atlas.rgde.evaluator import evaluate_extension, EvaluationResult
from atlas.types import FitMetrics

logger = logging.getLogger(__name__)

@dataclass
class RGDEConfig:
    k_range: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    scinet_epochs: int = 200
    scinet_lr: float = 1e-3
    scinet_seeds: int = 3
    encoder_sparsity: float = 0.01
    sr_niterations: int = 40
    sr_maxsize: int = 25
    constraint_max_degree: int = 2
    constraint_max_residual: float = 0.05
    min_r2_improvement: float = 0.1

@dataclass
class RGDEResult:
    success: bool
    dsl_type: DSLType | None
    decoder_formula: Expr | None
    r2_before: float
    r2_after: float
    evaluation: EvaluationResult | None
    k_selected: int | None
    encoder_r2: dict[int, float] | None = None

def run_rgde(X: np.ndarray, y: np.ndarray, var_names: list[str],
             r2_before: float, env_id: str,
             config: RGDEConfig | None = None) -> RGDEResult:
    if config is None:
        config = RGDEConfig()
    try:
        import torch
        from atlas.scinet.model import SciNet
        from atlas.scinet.trainer import train_scinet, TrainConfig
        from atlas.scinet.bottleneck import find_optimal_k, extract_bottleneck_vectors
    except ImportError:
        logger.warning("PyTorch not installed, RGDE unavailable")
        return RGDEResult(success=False, dsl_type=None, decoder_formula=None,
                          r2_before=r2_before, r2_after=-1.0, evaluation=None, k_selected=None)

    X_f = X.astype(np.float32)
    y_f = y.reshape(-1, 1).astype(np.float32) if y.ndim == 1 else y.astype(np.float32)

    # Step 4a: Find optimal K
    logger.info(f"RGDE Step 4a: Finding optimal K for {env_id}")
    k_result = find_optimal_k(X_f, y_f, k_range=config.k_range,
                               epochs_per_k=config.scinet_epochs, n_seeds=config.scinet_seeds)
    K = k_result.best_k
    model = k_result.models[K]
    logger.info(f"RGDE: Selected K={K} for {env_id}")
    Z = extract_bottleneck_vectors(model, X_f)

    # Step 4b: SR on encoder
    logger.info(f"RGDE Step 4b: SR on encoder ({K} dims)")
    from atlas.rgde.encoder_sr import run_encoder_sr
    encoder_result = run_encoder_sr(X, Z, var_names,
                                     niterations=config.sr_niterations, maxsize=config.sr_maxsize)

    # Step 4c: Find constraints
    logger.info("RGDE Step 4c: Finding constraints")
    constraints = find_constraints(Z, max_degree=config.constraint_max_degree,
                                    max_residual=config.constraint_max_residual)
    logger.info(f"RGDE: Found {len(constraints)} constraints")

    # Step 4d: Build type
    dsl_type = build_type(env_id, encoder_result.formulas, constraints)

    # Step 4e: SR on decoder (z -> y)
    logger.info("RGDE Step 4e: SR on decoder")
    z_var_names = [f"z_{k}" for k in range(K)]
    decoder_formula = None
    r2_after = -1.0
    try:
        from atlas.sr.pysr_wrapper import run_sr, SRConfig
        sr_config = SRConfig(niterations=config.sr_niterations, maxsize=config.sr_maxsize)
        y_flat = y_f.ravel() if y_f.shape[1] == 1 else np.mean(y_f, axis=1)
        sr_result = run_sr(Z, y_flat, z_var_names, sr_config)
        if sr_result.best_formula is not None:
            decoder_formula = sr_result.best_formula
            r2_after = sr_result.best_r_squared
    except ImportError:
        logger.warning("PySR not installed, skipping decoder SR")
    except Exception as e:
        logger.warning(f"Decoder SR failed: {e}")

    # Step 4f: Pareto evaluation
    mdl_before = 10.0
    mdl_after = decoder_formula.size() if decoder_formula else float("inf")
    type_cost = dsl_type.mdl_cost()
    evaluation = evaluate_extension(r2_before=r2_before, r2_after=r2_after,
                                     mdl_before=mdl_before, mdl_after=mdl_after,
                                     type_mdl_cost=type_cost, min_r2_improvement=config.min_r2_improvement)
    success = evaluation.accepted and decoder_formula is not None
    return RGDEResult(success=success, dsl_type=dsl_type if success else None,
                      decoder_formula=decoder_formula if success else None,
                      r2_before=r2_before, r2_after=r2_after, evaluation=evaluation,
                      k_selected=K, encoder_r2=encoder_result.r_squared_per_dim)
