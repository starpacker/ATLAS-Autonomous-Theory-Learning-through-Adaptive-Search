"""RGDE Step 4b: SR on encoder outputs — z_k = f_k(knobs) for each bottleneck dim."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from atlas.dsl.expr import Expr

@dataclass
class EncoderSRResult:
    formulas: dict[int, Expr]
    r_squared_per_dim: dict[int, float]
    success: bool

def run_encoder_sr(X: np.ndarray, Z: np.ndarray, var_names: list[str],
                   niterations: int = 40, maxsize: int = 25,
                   r2_threshold: float = 0.8) -> EncoderSRResult:
    try:
        from atlas.sr.pysr_wrapper import run_sr, SRConfig
    except ImportError:
        return EncoderSRResult(formulas={}, r_squared_per_dim={}, success=False)

    K = Z.shape[1]
    formulas: dict[int, Expr] = {}
    r2_per_dim: dict[int, float] = {}
    config = SRConfig(niterations=niterations, maxsize=maxsize)

    for k in range(K):
        try:
            result = run_sr(X, Z[:, k], var_names, config)
            if result.best_formula is not None:
                formulas[k] = result.best_formula
                r2_per_dim[k] = result.best_r_squared
            else:
                r2_per_dim[k] = -1.0
        except Exception:
            r2_per_dim[k] = -1.0

    success = len(formulas) == K and all(r2 > r2_threshold for r2 in r2_per_dim.values())
    return EncoderSRResult(formulas=formulas, r_squared_per_dim=r2_per_dim, success=success)
