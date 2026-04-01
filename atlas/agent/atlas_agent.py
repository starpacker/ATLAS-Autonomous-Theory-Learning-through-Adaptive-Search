"""ATLAS single-agent main loop: integrates SR, concepts, diagnostics, and unification."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import inf
from typing import Optional

import numpy as np

from atlas.agent.dsl_state import DSLState
from atlas.data.dataset import ExperimentDataset
from atlas.environments.registry import get_environment
from atlas.sr.pysr_wrapper import SRConfig, run_sr
from atlas.sr.formula_store import FormulaStore, _extract_constants
from atlas.analysis.concepts import extract_concepts
from atlas.analysis.diagnostics import run_all_diagnostics
from atlas.analysis.pslq_unifier import unify_constants
from atlas.dsl.expr import Expr
from atlas.dsl.serialize import to_str
from atlas.types import FitMetrics

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_epochs: int = 10
    r_squared_threshold: float = 0.95
    n_samples_per_knob: int = 10
    test_fraction: float = 0.2
    sr_niterations: int = 40
    sr_populations: int = 15
    sr_maxsize: int = 25
    sr_timeout: int = 300
    seed: int = 42
    min_concept_occurrences: int = 2
    enable_rgde: bool = False
    rgde_k_range: list[int] = field(default_factory=lambda: [1, 2, 3])
    rgde_scinet_epochs: int = 200
    rgde_sr_niterations: int = 40
    rgde_sr_maxsize: int = 25


@dataclass
class EpochResult:
    epoch: int
    formulas_found: int
    concepts_found: int
    diagnostics: dict  # env_id -> list of DiagnosticResult
    constants_unified: int
    converged_envs: list[str]
    failed_envs: list[str]
    extensions_found: list[str] = field(default_factory=list)


def _evaluate_formula(expr: Expr, X: np.ndarray, y: np.ndarray, var_names: list[str]) -> FitMetrics:
    """Evaluate formula on data, compute R², residual_var, MDL."""
    y_pred = np.array([
        expr.evaluate(dict(zip(var_names, row)))
        for row in X
    ])
    # Handle NaN/inf
    valid = np.isfinite(y_pred) & np.isfinite(y)
    if np.sum(valid) < 5:
        return FitMetrics(r_squared=-1.0, residual_var=inf, mdl=inf)
    ss_res = np.sum((y[valid] - y_pred[valid]) ** 2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-30)
    return FitMetrics(
        r_squared=float(r_squared),
        residual_var=float(np.var(y[valid] - y_pred[valid])),
        mdl=float(expr.size()),
    )


class ATLASAgent:
    """Single-agent ATLAS pipeline: data collection, SR, concept extraction, diagnostics, unification."""

    def __init__(self, env_ids: list[str], config: Optional[AgentConfig] = None):
        self.env_ids = list(env_ids)
        self.config = config or AgentConfig()
        self.dsl_state = DSLState()
        self.formula_store = FormulaStore()
        self.datasets: dict[str, ExperimentDataset] = {}
        self._current_epoch = 0
        # Track fit metrics per env
        self._fit_metrics: dict[str, FitMetrics] = {}
        # Track diagnostics per env
        self._diagnostics: dict[str, list] = {}

    def collect_data(self) -> None:
        """Collect data from all environments."""
        for env_id in self.env_ids:
            logger.info("Collecting data for %s", env_id)
            try:
                env = get_environment(env_id)
                ds = ExperimentDataset.from_env(
                    env,
                    n_samples_per_knob=self.config.n_samples_per_knob,
                    seed=self.config.seed,
                )
                self.datasets[env_id] = ds
                logger.info("Collected %d samples for %s", len(ds), env_id)
            except Exception as exc:
                logger.warning("Failed to collect data for %s: %s", env_id, exc)

    def run_epoch(self) -> EpochResult:
        """Run one epoch: Steps 1-3, 5 of the ATLAS pipeline."""
        epoch = self._current_epoch
        self._current_epoch += 1

        converged_envs: list[str] = []
        failed_envs: list[str] = []
        all_best_formulas: list[Expr] = []
        epoch_diagnostics: dict = {}

        sr_config = SRConfig(
            niterations=self.config.sr_niterations,
            populations=self.config.sr_populations,
            maxsize=self.config.sr_maxsize,
        )

        # Step 1: Solve — run SR for each environment
        for env_id in self.env_ids:
            if env_id not in self.datasets:
                logger.warning("No dataset for %s, skipping", env_id)
                failed_envs.append(env_id)
                continue

            ds = self.datasets[env_id]
            if len(ds) == 0:
                logger.warning("Empty dataset for %s, skipping", env_id)
                failed_envs.append(env_id)
                continue

            # Split train/test
            train_ds, test_ds = ds.split(
                test_fraction=self.config.test_fraction,
                seed=self.config.seed,
            )

            X_train = train_ds.knob_array()
            X_test = test_ds.knob_array()
            var_names = train_ds.knob_names

            # Get scalar output — use first detector, collapse array output to scalar
            first_det = ds.detector_names[0]
            y_train_raw = train_ds.detector_array(first_det)
            y_test_raw = test_ds.detector_array(first_det)

            if y_train_raw.ndim > 1:
                y_train = np.mean(y_train_raw, axis=1)
                y_test = np.mean(y_test_raw, axis=1)
            else:
                y_train = y_train_raw
                y_test = y_test_raw

            y_train = y_train.astype(float)
            y_test = y_test.astype(float)

            # Run SR — catch ImportError gracefully
            best_expr: Optional[Expr] = None
            best_r2 = -1.0

            try:
                sr_result = run_sr(X_train, y_train, var_names, config=sr_config)
                best_expr = sr_result.best_formula
                if best_expr is not None:
                    # Evaluate on test set
                    test_metrics = _evaluate_formula(best_expr, X_test, y_test, var_names)
                    best_r2 = test_metrics.r_squared
                    self._fit_metrics[env_id] = test_metrics
                    self.formula_store.add(env_id, best_expr, test_metrics)
                    all_best_formulas.append(best_expr)
                    logger.info("SR for %s: R²=%.4f", env_id, best_r2)
                else:
                    logger.warning("SR returned no formula for %s", env_id)

            except ImportError:
                logger.warning("PySR not installed — skipping SR for %s", env_id)
                failed_envs.append(env_id)
                continue
            except Exception as exc:
                logger.warning("SR failed for %s: %s", env_id, exc)
                failed_envs.append(env_id)
                continue

            # Classify env as converged or failed
            if best_r2 >= self.config.r_squared_threshold:
                converged_envs.append(env_id)
            else:
                failed_envs.append(env_id)

        # Step 2: Extract concepts from all best formulas
        new_concepts: list = []
        if len(all_best_formulas) >= 1:
            try:
                new_concepts = extract_concepts(
                    all_best_formulas,
                    min_occurrences=self.config.min_concept_occurrences,
                )
                for concept in new_concepts:
                    if concept.name not in self.dsl_state.concepts:
                        self.dsl_state.add_concept(concept.name, concept.expr)
                        logger.debug("Added concept: %s = %s", concept.name, to_str(concept.expr))
            except Exception as exc:
                logger.warning("Concept extraction failed: %s", exc)

        # Step 3: Diagnose failed environments
        for env_id in failed_envs:
            if env_id not in self.datasets:
                continue
            ds = self.datasets[env_id]
            # Compute residuals
            best_sf = self.formula_store.get_best(env_id)
            if best_sf is not None and best_sf.expr is not None:
                X_all = ds.knob_array().astype(float)
                first_det = ds.detector_names[0]
                y_all_raw = ds.detector_array(first_det)
                if y_all_raw.ndim > 1:
                    y_all = np.mean(y_all_raw, axis=1).astype(float)
                else:
                    y_all = y_all_raw.astype(float)
                var_names = ds.knob_names
                try:
                    y_pred = np.array([
                        best_sf.expr.evaluate(dict(zip(var_names, row)))
                        for row in X_all
                    ])
                    residuals = y_all - y_pred
                except Exception:
                    residuals = y_all
            else:
                first_det = ds.detector_names[0]
                residuals = ds.detector_array(first_det)
                if residuals.ndim > 1:
                    residuals = np.mean(residuals, axis=1)
                residuals = residuals.astype(float)

            try:
                best_r2 = self._fit_metrics.get(env_id, FitMetrics(-1.0, inf, inf)).r_squared
                diag_results = run_all_diagnostics(
                    dataset=ds,
                    best_r_squared=best_r2,
                    residuals=residuals,
                )
                epoch_diagnostics[env_id] = diag_results
                self._diagnostics[env_id] = diag_results
            except Exception as exc:
                logger.warning("Diagnostics failed for %s: %s", env_id, exc)
                epoch_diagnostics[env_id] = []

        # Step 4: Extend — RGDE on failed experiments (if enabled)
        extensions_found = []
        if self.config.enable_rgde:
            for env_id in failed_envs:
                ds = self.datasets.get(env_id)
                if ds is None or len(ds) < 50:
                    continue
                best = self.formula_store.get_best(env_id)
                best_r2 = best.fit.r_squared if best else -1.0
                X = ds.knob_array()
                y = ds.detector_array(ds.detector_names[0])
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                try:
                    from atlas.rgde.pipeline import run_rgde, RGDEConfig
                    rgde_config = RGDEConfig(
                        k_range=self.config.rgde_k_range,
                        scinet_epochs=self.config.rgde_scinet_epochs,
                        sr_niterations=self.config.rgde_sr_niterations,
                        sr_maxsize=self.config.rgde_sr_maxsize,
                    )
                    rgde_result = run_rgde(X, y, ds.knob_names, r2_before=best_r2,
                                           env_id=env_id, config=rgde_config)
                    if rgde_result.success and rgde_result.dsl_type is not None:
                        self.dsl_state.add_extension(
                            name=rgde_result.dsl_type.name, ext_type="new_type",
                            definition=rgde_result.dsl_type.to_dict(),
                            trigger=f"RGDE on {env_id}, K={rgde_result.k_selected}")
                        extensions_found.append(rgde_result.dsl_type.name)
                except ImportError:
                    logger.warning("RGDE unavailable (missing PyTorch or PySR)")
                except Exception as e:
                    logger.warning(f"RGDE failed for {env_id}: {e}")

        # Step 5: Unify constants
        all_constants: dict[str, float] = {}
        for env_id in self.env_ids:
            best_sf = self.formula_store.get_best(env_id)
            if best_sf is not None:
                for i, c in enumerate(_extract_constants(best_sf.expr)):
                    # Filter trivial constants
                    if abs(c) > 1e-6 and abs(c) != 1.0:
                        all_constants[f"{env_id}:C{i}"] = c

        unified = []
        if all_constants:
            try:
                unified = unify_constants(all_constants)
            except Exception as exc:
                logger.warning("Constant unification failed: %s", exc)

        return EpochResult(
            epoch=epoch,
            formulas_found=len(all_best_formulas),
            concepts_found=len(new_concepts),
            diagnostics=epoch_diagnostics,
            constants_unified=len(unified),
            converged_envs=converged_envs,
            failed_envs=failed_envs,
            extensions_found=extensions_found,
        )

    def run(self) -> dict:
        """Run the full ATLAS pipeline and return results."""
        # Collect data if not already done
        if not self.datasets:
            self.collect_data()

        epochs_run = 0
        last_result: Optional[EpochResult] = None

        for _ in range(self.config.max_epochs):
            last_result = self.run_epoch()
            epochs_run += 1
            logger.info(
                "Epoch %d: formulas=%d, concepts=%d, converged=%s",
                last_result.epoch,
                last_result.formulas_found,
                last_result.concepts_found,
                last_result.converged_envs,
            )
            # Early exit if all envs converged
            if set(last_result.converged_envs) >= set(self.env_ids):
                logger.info("All environments converged at epoch %d", last_result.epoch)
                break

        # Build output dict
        formulas: dict[str, str] = {}
        fit_metrics: dict[str, dict] = {}
        for env_id in self.env_ids:
            best_sf = self.formula_store.get_best(env_id)
            if best_sf is not None:
                formulas[env_id] = to_str(best_sf.expr)
                fm = best_sf.fit
                fit_metrics[env_id] = {
                    "r_squared": fm.r_squared,
                    "residual_var": fm.residual_var,
                    "mdl": fm.mdl,
                }
            else:
                formulas[env_id] = ""
                fit_metrics[env_id] = {
                    "r_squared": -1.0,
                    "residual_var": inf,
                    "mdl": inf,
                }

        # Collect all constants for output
        all_constants: dict[str, float] = {}
        for env_id in self.env_ids:
            best_sf = self.formula_store.get_best(env_id)
            if best_sf is not None:
                for i, c in enumerate(_extract_constants(best_sf.expr)):
                    if abs(c) > 1e-6 and abs(c) != 1.0:
                        all_constants[f"{env_id}:C{i}"] = c

        concepts_out: dict[str, str] = {
            name: to_str(expr)
            for name, expr in self.dsl_state.concepts.items()
        }

        # Flatten diagnostics for output
        diagnostics_out: dict[str, list] = {}
        for env_id, diag_list in self._diagnostics.items():
            diagnostics_out[env_id] = [
                {
                    "id": d.diagnostic_id,
                    "triggered": d.triggered,
                    "confidence": d.confidence,
                }
                for d in diag_list
            ]

        dsl_state_out = {
            "n_operators": len(self.dsl_state.operators),
            "n_concepts": len(self.dsl_state.concepts),
            "n_extensions": len(self.dsl_state.extensions),
            "mdl_cost": self.dsl_state.mdl_cost(),
        }

        return {
            "formulas": formulas,
            "constants": all_constants,
            "concepts": concepts_out,
            "diagnostics": diagnostics_out,
            "dsl_state": dsl_state_out,
            "fit_metrics": fit_metrics,
            "epochs_run": epochs_run,
            "extensions": [e for e in self.dsl_state.extensions],
        }
