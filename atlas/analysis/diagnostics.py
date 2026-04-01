"""D1/D2/D4 diagnostic tests for experiment data."""
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
    confidence = 1.0 - (n_clusters / max(max_clusters, 1))

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


def run_all_diagnostics(
    dataset,
    best_r_squared: float,
    residuals: np.ndarray,
    repeated_outputs: Optional[list[np.ndarray]] = None,
) -> list[DiagnosticResult]:
    """Run all available diagnostics (D1, D2, D4) and return results.

    D3 (aliasing) and D5 (dimensional analysis) are deferred and not
    implemented here.
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

    # D3 deferred
    results.append(DiagnosticResult("D3", False, 0.0, {"reason": "deferred"}))

    # D4 — residual structure
    results.append(diagnose_residual_structure(np.asarray(residuals)))

    # D5 deferred
    results.append(DiagnosticResult("D5", False, 0.0, {"reason": "deferred"}))

    return results
