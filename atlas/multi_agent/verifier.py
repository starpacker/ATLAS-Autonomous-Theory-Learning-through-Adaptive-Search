"""Experiment-centric verification with global MDL criterion.

Evaluates a proposed DSL extension by checking its MDL impact across all
experiments.  Supports two verification strategies:

1. **SR-based verification** (``verify_proposal_sr``): actually runs symbolic
   regression M times with and without the extension's concepts on each
   experiment and computes paired per-experiment delta-MDL.  This is the
   approach specified in the design doc (Section 3.2).

2. **Estimate-based verification** (legacy): uses heuristic estimates from
   existing fit metrics.  Cheaper but less accurate.

Uses statistical significance (pooled noise) to filter spurious gains.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of global MDL verification for a proposal."""
    delta_total_mdl: float
    pooled_noise: float
    per_env_results: dict[str, dict]
    should_adopt: bool
    reason: str


def compute_global_mdl_delta(per_env_deltas: dict[str, dict]) -> VerificationResult:
    """Compute global MDL delta from per-experiment results.

    Args:
        per_env_deltas: {env_id: {"mu": mean_delta_mdl, "sigma": std_delta_mdl}}

    Returns:
        VerificationResult with adoption decision
    """
    if not per_env_deltas:
        return VerificationResult(
            delta_total_mdl=0.0, pooled_noise=float("inf"),
            per_env_results={}, should_adopt=False,
            reason="No experiment data"
        )

    delta_total = sum(d["mu"] for d in per_env_deltas.values())
    pooled_variance = sum(d["sigma"] ** 2 for d in per_env_deltas.values())
    pooled_noise = math.sqrt(pooled_variance) if pooled_variance > 0 else 0.0

    significant = is_statistically_significant(delta_total, pooled_noise)
    should_adopt = delta_total < 0 and significant

    if should_adopt:
        reason = f"Global MDL decreased by {abs(delta_total):.2f} (noise={pooled_noise:.2f})"
    elif delta_total >= 0:
        reason = f"Global MDL increased by {delta_total:.2f}"
    else:
        reason = f"MDL decrease {abs(delta_total):.2f} not significant (noise={pooled_noise:.2f})"

    return VerificationResult(
        delta_total_mdl=delta_total,
        pooled_noise=pooled_noise,
        per_env_results=dict(per_env_deltas),
        should_adopt=should_adopt,
        reason=reason,
    )


def is_statistically_significant(delta_total: float, pooled_noise: float,
                                  threshold: float = 2.0) -> bool:
    """Check if delta_total exceeds threshold * pooled_noise."""
    if pooled_noise <= 0:
        return delta_total < 0
    return abs(delta_total) > threshold * pooled_noise


# ---------------------------------------------------------------------------
# SR-based per-experiment verification
# ---------------------------------------------------------------------------

def verify_proposal_sr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    var_names: list[str],
    concept_columns: Optional[dict[str, np.ndarray]] = None,
    concept_columns_test: Optional[dict[str, np.ndarray]] = None,
    n_seeds: int = 3,
    sr_niterations: int = 20,
    sr_maxsize: int = 20,
    base_seed: int = 0,
) -> dict:
    """Run SR with and without extension concepts; return paired delta-MDL.

    For each seed s in [base_seed, base_seed + n_seeds):
      1. Run SR on (X_train, y_train) with ``var_names`` only          → MDL_base_s
      2. Run SR on (X_train_aug, y_train) with augmented var_names     → MDL_ext_s
      3. delta_MDL_s = MDL_ext_s - MDL_base_s   (negative = extension helps)

    Returns ``{"mu": mean(delta_MDL), "sigma": std(delta_MDL),
               "n_seeds": actual_seeds_run, "method": "sr"}``.

    If PySR is not available, returns None.

    Parameters
    ----------
    X_train, y_train : arrays
        Training data.
    X_test, y_test : arrays
        Held-out test data for evaluation.
    var_names : list[str]
        Base variable names.
    concept_columns : dict[str, ndarray], optional
        Extra columns from the extension's encoding formulas, keyed by name.
        Shape of each array: (n_train_samples,).
    concept_columns_test : dict[str, ndarray], optional
        Same concept columns computed on test data.
    n_seeds : int
        Number of independent SR runs per condition.
    sr_niterations, sr_maxsize : int
        SR hyperparameters (kept small for verification speed).
    base_seed : int
        Starting random seed.
    """
    try:
        from atlas.sr.pysr_wrapper import run_sr, SRConfig
    except ImportError:
        return None

    # Build augmented data (with extension concepts)
    if concept_columns:
        extra_train = np.column_stack(list(concept_columns.values()))
        extra_test = np.column_stack(list(
            (concept_columns_test or concept_columns).values()))
        X_train_aug = np.hstack([X_train, extra_train])
        X_test_aug = np.hstack([X_test, extra_test])
        var_names_aug = list(var_names) + list(concept_columns.keys())
    else:
        X_train_aug = X_train
        X_test_aug = X_test
        var_names_aug = list(var_names)

    delta_mdls: list[float] = []

    for s in range(n_seeds):
        seed = base_seed + s
        config_base = SRConfig(
            niterations=sr_niterations, maxsize=sr_maxsize,
            populations=5,
        )
        config_ext = SRConfig(
            niterations=sr_niterations, maxsize=sr_maxsize,
            populations=5,
        )

        # Baseline: SR without extension concepts
        try:
            result_base = run_sr(X_train, y_train, var_names, config=config_base)
            mdl_base = _best_test_mdl(result_base, X_test, y_test, var_names)
        except Exception as exc:
            logger.debug("Verification SR (base) failed seed %d: %s", seed, exc)
            continue

        # Extended: SR with extension concepts
        try:
            result_ext = run_sr(X_train_aug, y_train, var_names_aug, config=config_ext)
            mdl_ext = _best_test_mdl(result_ext, X_test_aug, y_test, var_names_aug)
        except Exception as exc:
            logger.debug("Verification SR (ext) failed seed %d: %s", seed, exc)
            continue

        delta_mdls.append(mdl_ext - mdl_base)

    if not delta_mdls:
        return None

    return {
        "mu": float(np.mean(delta_mdls)),
        "sigma": float(np.std(delta_mdls)) if len(delta_mdls) > 1 else float(abs(np.mean(delta_mdls)) * 0.1),
        "n_seeds": len(delta_mdls),
        "method": "sr",
    }


def _best_test_mdl(sr_result, X_test: np.ndarray, y_test: np.ndarray,
                    var_names: list[str]) -> float:
    """Evaluate the best SR formula on test data and return MDL.

    MDL = expression size (complexity).  We pick the formula with the
    best R² on the test set (among those with R² > 0).
    """
    best_mdl = float("inf")
    best_r2 = -1.0

    for expr in (sr_result.formulas or []):
        try:
            y_pred = np.array([
                expr.evaluate(dict(zip(var_names, row)))
                for row in X_test
            ])
            valid = np.isfinite(y_pred) & np.isfinite(y_test)
            if np.sum(valid) < 5:
                continue
            ss_res = np.sum((y_test[valid] - y_pred[valid]) ** 2)
            ss_tot = np.sum((y_test[valid] - np.mean(y_test[valid])) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            if r2 > best_r2:
                best_r2 = r2
                best_mdl = float(expr.mdl_cost())
        except Exception:
            continue

    return best_mdl
