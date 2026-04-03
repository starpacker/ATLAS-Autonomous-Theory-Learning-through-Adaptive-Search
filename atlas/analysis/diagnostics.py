"""D1–D4 diagnostic tests for experiment data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DiagnosticResult:
    diagnostic_id: str
    triggered: bool
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


def diagnose_stochasticity(
    repeated_outputs: list[np.ndarray],
    threshold: float = 0.05,
) -> DiagnosticResult:
    """D1: Detect non-deterministic/stochastic outputs.

    Computes the mean coefficient of variation (std/|mean|) across all
    positions.  If the average CV exceeds *threshold*, the system is flagged
    as stochastic.
    """
    if len(repeated_outputs) < 2:
        return DiagnosticResult("D1", False, 0.0, {"reason": "not enough repeats"})

    stacked = np.stack(repeated_outputs, axis=0).astype(float)  # (n_repeats, n_outputs)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)

    # Avoid division by zero: positions where mean is 0 use absolute std
    abs_mean = np.abs(mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(abs_mean > 1e-15, std / abs_mean, std)

    mean_cv = float(np.mean(cv))
    triggered = mean_cv > threshold
    confidence = min(1.0, mean_cv / (threshold + 1e-15))

    return DiagnosticResult(
        "D1",
        triggered,
        confidence,
        {"mean_cv": mean_cv, "threshold": threshold},
    )


def diagnose_discreteness(
    outputs: np.ndarray,
    max_clusters: int = 10,
) -> DiagnosticResult:
    """D2: Detect whether outputs cluster into a small number of discrete values.

    Uses rounding to 4 significant figures to count unique values.  If the
    number of unique values is much smaller than the number of samples we
    flag the output as discrete.
    """
    outputs = np.asarray(outputs, dtype=float).ravel()
    n = len(outputs)
    if n == 0:
        return DiagnosticResult("D2", False, 0.0, {"reason": "empty"})

    # Round to 4 significant figures then count unique values
    scale = np.max(np.abs(outputs))
    if scale == 0:
        rounded = np.zeros_like(outputs)
    else:
        rounded = np.round(outputs / scale, 4) * scale

    unique_vals = np.unique(rounded)
    n_clusters = len(unique_vals)

    # Triggered when clusters are few and well below the sample count.
    # Use n_clusters <= sqrt(n) as the "sparse cluster" threshold — this
    # scales naturally and avoids the fixed-fraction being too strict for
    # small sample counts (e.g. 2 clusters out of 10 samples).
    triggered = (n_clusters <= max_clusters) and (n_clusters <= np.sqrt(n))
    # Confidence: 1 cluster = maximally discrete (1.0), at the effective
    # threshold = barely discrete (small but nonzero).  Use (n-1)/(max-1)
    # so the boundary case (n_clusters == threshold) still yields > 0.
    effective_threshold = max(min(max_clusters, int(np.sqrt(n))), 1)
    confidence = 1.0 - (n_clusters - 1) / max(effective_threshold, 1)

    return DiagnosticResult(
        "D2",
        triggered,
        max(0.0, confidence),
        {"n_clusters": n_clusters, "max_clusters": max_clusters, "n_samples": n},
    )


def diagnose_residual_structure(
    residuals: np.ndarray,
    significance: float = 0.05,
) -> DiagnosticResult:
    """D4: Detect structure (periodicity / autocorrelation) in model residuals.

    Uses two sub-tests:
    1. FFT concentration: fraction of power in the top-5 frequencies.
    2. Lag-1 autocorrelation: |r(1)| exceeds a threshold.

    Either sub-test triggering causes D4 to fire.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    n = len(residuals)
    if n < 4:
        return DiagnosticResult("D4", False, 0.0, {"reason": "too few samples"})

    # --- FFT concentration ---
    fft_mag = np.abs(np.fft.rfft(residuals - residuals.mean()))
    fft_mag[0] = 0.0  # ignore DC
    total_power = float(np.sum(fft_mag ** 2))
    if total_power < 1e-30:
        fft_conc = 0.0
    else:
        top_k = min(5, len(fft_mag))
        top_power = float(np.sum(np.sort(fft_mag ** 2)[::-1][:top_k]))
        fft_conc = top_power / total_power

    # --- Lag-1 autocorrelation ---
    r = residuals - residuals.mean()
    var = float(np.var(r))
    if var < 1e-30:
        lag1 = 0.0
    else:
        lag1 = float(np.dot(r[:-1], r[1:])) / ((n - 1) * var)

    # Thresholds calibrated so that pure noise rarely triggers
    fft_threshold = 0.5   # top-5 frequencies hold >50 % of power
    lag1_threshold = 0.3  # |r(1)| > 0.3

    triggered = (fft_conc > fft_threshold) or (abs(lag1) > lag1_threshold)
    confidence = max(fft_conc / fft_threshold, abs(lag1) / lag1_threshold)
    confidence = min(1.0, confidence)

    return DiagnosticResult(
        "D4",
        triggered,
        confidence,
        {
            "fft_concentration": fft_conc,
            "lag1_autocorr": lag1,
            "fft_threshold": fft_threshold,
            "lag1_threshold": lag1_threshold,
        },
    )


def diagnose_bottleneck_dimension(
    dataset,
    scinet_epochs: int = 50,
    max_output_dim: int = 50,
) -> DiagnosticResult:
    """D3: Progressive bottleneck search — K_bottleneck > n_active_knobs.

    Trains SciNet autoencoders with increasing bottleneck dimension K and
    uses AIC to pick the best K.  If K_best > n_knobs the experiment has
    hidden structure that raw symbolic regression cannot capture.  This is
    the signal that triggers RGDE.

    The finding "this system needs K>N latent dimensions" is itself
    scientifically interesting — it means there are unobserved degrees of
    freedom (no prior physics knowledge is injected).
    """
    try:
        from atlas.scinet.bottleneck import find_optimal_k
    except ImportError:
        return DiagnosticResult("D3", False, 0.0,
                                {"reason": "PyTorch not available"})

    first_det = dataset.detector_names[0]
    X = dataset.knob_array().astype(np.float32)
    y = dataset.detector_array(first_det).astype(np.float32)

    n_knobs = X.shape[1]
    n_samples = X.shape[0]

    if n_samples < 50:
        return DiagnosticResult("D3", False, 0.0,
                                {"reason": "too few samples",
                                 "n_samples": n_samples})

    # Subsample output positions for array detectors (speed)
    if y.ndim > 1 and y.shape[1] > max_output_dim:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(y.shape[1], max_output_dim, replace=False))
        y = y[:, indices]

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Test K from 1 up to n_knobs + 2, capped at 6 for speed
    max_k = min(max(n_knobs + 2, 3), 6)
    k_range = list(range(1, max_k + 1))

    try:
        result = find_optimal_k(X, y, k_range=k_range,
                                epochs_per_k=scinet_epochs)
    except Exception as exc:
        return DiagnosticResult("D3", False, 0.0, {"error": str(exc)})

    k_bottleneck = result.best_k
    triggered = k_bottleneck > n_knobs

    # Confidence: relative AIC improvement of K_best over K = n_knobs
    confidence = 0.0
    if triggered and n_knobs in result.aic_scores:
        aic_n = result.aic_scores[n_knobs]
        aic_best = result.aic_scores[k_bottleneck]
        if abs(aic_n) > 1e-10:
            confidence = min(1.0, max(0.0,
                                      (aic_n - aic_best) / abs(aic_n)))

    return DiagnosticResult(
        "D3",
        triggered,
        confidence,
        {
            "k_bottleneck": k_bottleneck,
            "n_knobs": n_knobs,
            "aic_scores": {k: float(v) for k, v in result.aic_scores.items()},
            "losses": {k: float(v) for k, v in result.losses.items()},
        },
    )


def diagnose_cross_experiment_inconsistency(
    env_id: str,
    all_constants: dict[str, float],
    group_tolerance: float = 0.01,
    outlier_threshold: float = 0.05,
) -> DiagnosticResult:
    """D5: Detect constants that are inconsistent across experiments.

    Groups constants from all experiments by value proximity (the same
    grouping used by constant unification).  For each group that
    includes a constant from *env_id*, checks whether that constant
    deviates from the group mean by more than *outlier_threshold*
    (relative).

    A triggered D5 means this environment has a constant that *should*
    match another experiment's constant but does not — useful for
    flagging measurement artefacts or model mis-specification.
    """
    env_consts = {k: v for k, v in all_constants.items()
                  if k.startswith(f"{env_id}:")}
    other_consts = {k: v for k, v in all_constants.items()
                    if not k.startswith(f"{env_id}:")}

    if not env_consts or not other_consts:
        return DiagnosticResult("D5", False, 0.0,
                                {"reason": "insufficient cross-env data"})

    inconsistencies: list[dict] = []

    for ek, ev in env_consts.items():
        abs_ev = abs(ev)
        if abs_ev < 1e-10:
            continue

        # Find constants from other envs that should be the same
        group_vals = [abs_ev]
        group_keys = [ek]
        for ok, ov in other_consts.items():
            abs_ov = abs(ov)
            if abs_ov < 1e-10:
                continue
            if abs(abs_ev - abs_ov) / max(abs_ev, abs_ov) < group_tolerance:
                group_vals.append(abs_ov)
                group_keys.append(ok)

        if len(group_vals) < 2:
            continue  # no cross-env match

        mean_val = float(np.mean(group_vals))
        if mean_val < 1e-10:
            continue
        deviation = abs(abs_ev - mean_val) / mean_val

        if deviation > outlier_threshold:
            inconsistencies.append({
                "constant": ek,
                "value": float(ev),
                "group_mean": mean_val,
                "deviation": deviation,
                "group_size": len(group_vals),
            })

    triggered = len(inconsistencies) > 0
    confidence = 0.0
    if inconsistencies:
        max_dev = max(d["deviation"] for d in inconsistencies)
        confidence = min(1.0, max_dev / outlier_threshold)

    return DiagnosticResult(
        "D5", triggered, confidence,
        {"n_inconsistencies": len(inconsistencies),
         "details": inconsistencies[:5]},
    )


def run_all_diagnostics(
    dataset,
    best_r_squared: float,
    residuals: np.ndarray,
    repeated_outputs: Optional[list[np.ndarray]] = None,
) -> list[DiagnosticResult]:
    """Run per-environment diagnostics (D1–D4) and return results.

    D5 (cross-experiment inconsistency) requires all-env constants and
    is computed post-hoc by the agent after constant unification.
    A placeholder is included here; the agent replaces it.
    """
    results: list[DiagnosticResult] = []

    # D1 — stochasticity
    if repeated_outputs is not None and len(repeated_outputs) >= 2:
        results.append(diagnose_stochasticity(repeated_outputs))
    else:
        results.append(DiagnosticResult("D1", False, 0.0, {"reason": "no repeated outputs provided"}))

    # D2 — discreteness (use first detector column of the dataset)
    try:
        first_det = dataset.detector_names[0]
        outputs = dataset.detector_array(first_det).ravel()
        results.append(diagnose_discreteness(outputs))
    except Exception as exc:
        results.append(DiagnosticResult("D2", False, 0.0, {"error": str(exc)}))

    # D3 — bottleneck dimension
    results.append(diagnose_bottleneck_dimension(dataset))

    # D4 — residual structure
    results.append(diagnose_residual_structure(np.asarray(residuals)))

    # D5 deferred
    results.append(DiagnosticResult("D5", False, 0.0, {"reason": "deferred"}))

    return results
