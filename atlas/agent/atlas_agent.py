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
from atlas.sr.formula_store import FormulaStore, extract_constants
from atlas.analysis.concepts import extract_concepts
from atlas.analysis.diagnostics import (
    run_all_diagnostics, diagnose_cross_experiment_inconsistency,
)
from atlas.analysis.pslq_unifier import unify_constants
from atlas.dsl.expr import Expr
from atlas.dsl.serialize import to_str
from atlas.types import FitMetrics

# Maximum number of array positions to sample when expanding array outputs.
# Expanding all 1000 positions × N knob combos produces too many rows for SR.
_MAX_POSITION_SAMPLES = 50

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


def _expand_array_output(
    X: np.ndarray,
    y_array: np.ndarray,
    var_names: list[str],
    max_positions: int = _MAX_POSITION_SAMPLES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Expand array detector outputs into tabular form with position as a feature.

    For an array output of shape (n_samples, n_positions), we create a flat
    dataset where each row is (knobs..., normalised_position) → scalar_value.

    To keep the dataset size manageable for SR we sub-sample positions
    uniformly.

    Returns (X_expanded, y_expanded, expanded_var_names).
    """
    n_samples, n_positions = y_array.shape
    # Sub-sample positions if too many
    if n_positions > max_positions:
        rng = np.random.default_rng(seed)
        pos_indices = np.sort(rng.choice(n_positions, max_positions, replace=False))
    else:
        pos_indices = np.arange(n_positions)

    n_pos = len(pos_indices)
    # Normalise positions to [0, 1]
    norm_positions = pos_indices / max(n_positions - 1, 1)

    # Build expanded arrays: each original sample × each selected position
    X_expanded = np.repeat(X, n_pos, axis=0)                 # (n_samples*n_pos, n_knobs)
    pos_col = np.tile(norm_positions, n_samples).reshape(-1, 1)  # (n_samples*n_pos, 1)
    X_expanded = np.hstack([X_expanded, pos_col])

    y_expanded = y_array[:, pos_indices].ravel()

    expanded_names = list(var_names) + ["_position"]
    return X_expanded, y_expanded, expanded_names


def _augment_with_concepts(
    X: np.ndarray,
    var_names: list[str],
    concepts: dict[str, Expr],
) -> tuple[np.ndarray, list[str]]:
    """Add columns for each DSL concept applied to each input variable.

    For a concept like ``cos2(v) = cos(v)**2``, and variables [knob_0, knob_1],
    adds columns ``concept_cos2__knob_0``, ``concept_cos2__knob_1``.

    Only single-variable concepts are expanded (multi-variable concepts would
    produce a combinatorial explosion).  The original columns are preserved.

    Returns (X_augmented, augmented_var_names).
    """
    if not concepts:
        return X, var_names

    extra_cols: list[np.ndarray] = []
    extra_names: list[str] = []

    for cname, cexpr in concepts.items():
        cvars = cexpr.variables()
        if len(cvars) != 1:
            continue  # skip multi-variable concepts
        cvar = next(iter(cvars))  # the single placeholder variable

        for j, vname in enumerate(var_names):
            col_values = np.empty(X.shape[0], dtype=float)
            for i in range(X.shape[0]):
                try:
                    col_values[i] = cexpr.evaluate({cvar: float(X[i, j])})
                except Exception:
                    col_values[i] = np.nan
            # Only add if column contains useful values
            if np.all(np.isfinite(col_values)) and np.std(col_values) > 1e-12:
                extra_cols.append(col_values)
                extra_names.append(f"{cname}__{vname}")

    if not extra_cols:
        return X, var_names

    X_aug = np.column_stack([X] + extra_cols)
    aug_names = list(var_names) + extra_names
    return X_aug, aug_names


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
        mdl=float(expr.mdl_cost()),
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
        # Prob_mode: envs flagged stochastic by D1
        self._stochastic_envs: set[str] = set()
        self._prob_datasets: dict[str, ExperimentDataset] = {}

    # Number of repeated runs to generate for D1 stochasticity detection
    _N_REPEATS_FOR_D1 = 10
    # Number of repeats per knob setting for probabilistic data collection
    _N_PROB_REPEATS = 20

    def _generate_repeated_outputs(
        self, env_id: str, n_repeats: int | None = None,
    ) -> list[np.ndarray] | None:
        """Re-run an environment with different seeds to detect stochasticity.

        Picks a single representative knob setting from the dataset and runs
        the environment *n_repeats* times with distinct seeds.  Returns a list
        of detector output arrays (one per run), suitable for D1 diagnostics.

        Returns None if the environment cannot be instantiated.
        """
        if n_repeats is None:
            n_repeats = self._N_REPEATS_FOR_D1

        ds = self.datasets.get(env_id)
        if ds is None or len(ds) == 0:
            return None

        # Use a knob setting near the middle of the dataset
        mid_idx = len(ds) // 2
        knobs = ds.get_knobs(mid_idx)
        first_det = ds.detector_names[0]

        outputs: list[np.ndarray] = []
        try:
            for r in range(n_repeats):
                seed = self.config.seed + 9999 + r  # distinct seeds
                env = get_environment(env_id, seed=seed)
                result = env.run(knobs)
                val = result[first_det]
                outputs.append(np.atleast_1d(np.asarray(val, dtype=float)))
        except Exception as exc:
            logger.debug("Repeated runs failed for %s: %s", env_id, exc)
            return None

        return outputs if len(outputs) >= 2 else None

    def _collect_probabilistic_data(
        self, env_id: str, n_repeats: int | None = None,
    ) -> ExperimentDataset | None:
        """Re-collect data with averaging for stochastic environments.

        Runs the environment *n_repeats* times at each knob setting,
        averages detector outputs, and normalizes to a probability
        distribution.  The resulting dataset is suitable for SR to
        discover P(y|x) rather than y = f(x).
        """
        if n_repeats is None:
            n_repeats = self._N_PROB_REPEATS

        orig_ds = self.datasets.get(env_id)
        if orig_ds is None or len(orig_ds) == 0:
            return None

        prob_ds = ExperimentDataset(
            env_id, orig_ds.knob_names, orig_ds.detector_names)
        first_det = orig_ds.detector_names[0]

        for knobs in orig_ds.iter_knobs():
            accumulated: np.ndarray | None = None
            n_success = 0
            for r in range(n_repeats):
                try:
                    seed = self.config.seed + 7000 + r
                    env = get_environment(env_id, seed=seed)
                    result = env.run(knobs)
                    arr = np.atleast_1d(
                        np.asarray(result[first_det], dtype=float))
                    if accumulated is None:
                        accumulated = np.zeros_like(arr)
                    accumulated += arr
                    n_success += 1
                except Exception:
                    pass

            if accumulated is not None and n_success > 0:
                averaged = accumulated / n_success
                # Normalize to probability distribution
                total = float(averaged.sum())
                if total > 0:
                    averaged = averaged / total
                prob_ds.add(knobs, {first_det: averaged})

        return prob_ds if len(prob_ds) > 0 else None

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

            # Use probabilistic dataset for stochastic envs (prob_mode)
            ds = self._prob_datasets.get(env_id, self.datasets[env_id])
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

            # Get output — use first detector
            first_det = ds.detector_names[0]
            y_train_raw = train_ds.detector_array(first_det)
            y_test_raw = test_ds.detector_array(first_det)

            # For array outputs, expand position as an extra feature
            # so SR can discover spatial patterns like cos²(k·position).
            is_array_output = y_train_raw.ndim > 1 and y_train_raw.shape[1] > 1
            if is_array_output:
                X_train, y_train, var_names = _expand_array_output(
                    X_train, y_train_raw, var_names,
                    seed=self.config.seed,
                )
                X_test, y_test, _ = _expand_array_output(
                    X_test, y_test_raw, test_ds.knob_names,
                    seed=self.config.seed,
                )
            else:
                if y_train_raw.ndim > 1:
                    y_train = y_train_raw.ravel()
                    y_test = y_test_raw.ravel()
                else:
                    y_train = y_train_raw
                    y_test = y_test_raw

            y_train = y_train.astype(float)
            y_test = y_test.astype(float)

            # Augment features with pre-computed DSL concepts so that
            # PySR can discover formulas using previously learned library functions.
            X_train_aug, var_names_aug = _augment_with_concepts(
                X_train, var_names, self.dsl_state.concepts)
            X_test_aug, _ = _augment_with_concepts(
                X_test, var_names, self.dsl_state.concepts)

            # Run SR — catch ImportError gracefully
            best_expr: Optional[Expr] = None
            best_r2 = -1.0

            try:
                sr_result = run_sr(X_train_aug, y_train, var_names_aug, config=sr_config)
                if sr_result.formulas:
                    # Evaluate ALL Pareto-front candidates on test set so
                    # the formula store can select the best R²/MDL tradeoff.
                    for expr in sr_result.formulas:
                        test_metrics = _evaluate_formula(expr, X_test_aug, y_test, var_names_aug)
                        if test_metrics.r_squared > 0.0:
                            self.formula_store.add(env_id, expr, test_metrics)
                    # Use the Pareto-selected formula for ALL downstream use
                    # (convergence, diagnostics, concepts) so everything is
                    # consistent with the formula we actually report.
                    best_sf = self.formula_store.get_best(env_id)
                    if best_sf is not None:
                        best_r2 = best_sf.fit.r_squared
                        self._fit_metrics[env_id] = best_sf.fit
                        all_best_formulas.append(best_sf.expr)
                        logger.info("SR for %s: R²=%.4f, MDL=%.1f", env_id, best_r2, best_sf.fit.mdl)
                    else:
                        logger.warning("No valid formula for %s after test evaluation", env_id)
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
            first_det = ds.detector_names[0]

            # Compute residuals — use expanded form for array outputs
            best_sf = self.formula_store.get_best(env_id)
            y_all_raw = ds.detector_array(first_det)
            is_arr = y_all_raw.ndim > 1 and y_all_raw.shape[1] > 1

            if best_sf is not None and best_sf.expr is not None:
                X_all = ds.knob_array().astype(float)
                if is_arr:
                    X_diag, y_all, vn_diag = _expand_array_output(
                        X_all, y_all_raw, ds.knob_names, seed=self.config.seed)
                else:
                    X_diag = X_all
                    y_all = (y_all_raw.ravel() if y_all_raw.ndim > 1
                             else y_all_raw).astype(float)
                    vn_diag = ds.knob_names
                # Augment with concepts so formulas using concept columns
                # can be evaluated (they were trained on augmented features).
                X_diag, vn_diag = _augment_with_concepts(
                    X_diag, vn_diag, self.dsl_state.concepts)
                try:
                    y_pred = np.array([
                        best_sf.expr.evaluate(dict(zip(vn_diag, row)))
                        for row in X_diag
                    ])
                    residuals = y_all - y_pred
                except Exception:
                    residuals = y_all
            else:
                if is_arr:
                    X_all = ds.knob_array().astype(float)
                    _, residuals, _ = _expand_array_output(
                        X_all, y_all_raw, ds.knob_names, seed=self.config.seed)
                else:
                    residuals = (y_all_raw.ravel() if y_all_raw.ndim > 1
                                 else y_all_raw).astype(float)

            # Generate repeated outputs for D1 stochasticity detection
            repeated_outputs = self._generate_repeated_outputs(env_id)

            try:
                best_r2 = self._fit_metrics.get(env_id, FitMetrics(-1.0, inf, inf)).r_squared
                diag_results = run_all_diagnostics(
                    dataset=ds,
                    best_r_squared=best_r2,
                    residuals=residuals,
                    repeated_outputs=repeated_outputs,
                )
                epoch_diagnostics[env_id] = diag_results
                self._diagnostics[env_id] = diag_results

                # D1 triggered → enable prob_mode for next epoch
                d1 = next((d for d in diag_results
                           if d.diagnostic_id == "D1"), None)
                if (d1 is not None and d1.triggered
                        and env_id not in self._stochastic_envs):
                    self._stochastic_envs.add(env_id)
                    prob_ds = self._collect_probabilistic_data(env_id)
                    if prob_ds is not None:
                        self._prob_datasets[env_id] = prob_ds
                        logger.info("prob_mode enabled for %s (D1=stochastic)",
                                    env_id)
            except Exception as exc:
                logger.warning("Diagnostics failed for %s: %s", env_id, exc)
                epoch_diagnostics[env_id] = []

        # Step 4: Extend — RGDE on failed experiments (if enabled)
        # Diagnostic linkage:
        #   D1=stochastic → skip RGDE (handled by prob_mode)
        #   D3=K>N        → run RGDE with focused k_range
        #   D3 ran but K<=N → skip RGDE (no hidden structure)
        #   D3 unavailable → fall back to default k_range
        extensions_found = []
        if self.config.enable_rgde:
            for env_id in failed_envs:
                # Skip stochastic envs — they need prob_mode, not RGDE
                if env_id in self._stochastic_envs:
                    logger.debug("Skipping RGDE for %s (stochastic, use prob_mode)", env_id)
                    continue

                ds = self.datasets.get(env_id)
                if ds is None or len(ds) < 50:
                    continue

                # Check D3 to decide whether RGDE is warranted
                d3 = next((d for d in self._diagnostics.get(env_id, [])
                           if d.diagnostic_id == "D3"), None)

                # D3 ran but K <= N: no hidden structure, skip RGDE
                if d3 is not None and not d3.triggered:
                    logger.debug("Skipping RGDE for %s (D3: K_bottleneck <= n_knobs)", env_id)
                    continue

                # Use D3's K_bottleneck to focus RGDE search
                rgde_k_range = self.config.rgde_k_range
                if d3 is not None and d3.triggered:
                    k_target = d3.details.get("k_bottleneck")
                    if k_target is not None:
                        rgde_k_range = list(range(
                            max(1, k_target - 1), k_target + 2))
                        logger.info("RGDE for %s: D3 suggests K=%d, searching %s",
                                    env_id, k_target, rgde_k_range)

                best = self.formula_store.get_best(env_id)
                best_r2 = best.fit.r_squared if best else -1.0
                X = ds.knob_array()
                y = ds.detector_array(ds.detector_names[0])
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                try:
                    from atlas.rgde.pipeline import run_rgde, RGDEConfig
                    rgde_config = RGDEConfig(
                        k_range=rgde_k_range,
                        scinet_epochs=self.config.rgde_scinet_epochs,
                        sr_niterations=self.config.rgde_sr_niterations,
                        sr_maxsize=self.config.rgde_sr_maxsize,
                    )
                    # Pass actual MDL so Pareto evaluation is meaningful
                    actual_mdl = best.fit.mdl if best and best.fit.mdl != inf else None
                    rgde_result = run_rgde(X, y, ds.knob_names, r2_before=best_r2,
                                           env_id=env_id, config=rgde_config,
                                           mdl_before=actual_mdl)
                    if rgde_result.success and rgde_result.dsl_type is not None:
                        self.dsl_state.add_extension(
                            name=rgde_result.dsl_type.name, ext_type="new_type",
                            definition=rgde_result.dsl_type.to_dict(),
                            trigger=f"RGDE on {env_id}, K={rgde_result.k_selected}",
                            source_env=env_id,
                            r2_before=rgde_result.r2_before,
                            r2_after=rgde_result.r2_after,
                            delta_r2=rgde_result.r2_after - rgde_result.r2_before,
                        )
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
                for i, c in enumerate(extract_constants(best_sf.expr)):
                    # Filter trivial constants
                    if abs(c) > 1e-6 and abs(c) != 1.0:
                        all_constants[f"{env_id}:C{i}"] = c

        unified = []
        if all_constants:
            try:
                unified = unify_constants(all_constants)
            except Exception as exc:
                logger.warning("Constant unification failed: %s", exc)

        # D5: cross-experiment inconsistency (requires all_constants)
        if len(all_constants) >= 2:
            for env_id in self.env_ids:
                d5 = diagnose_cross_experiment_inconsistency(
                    env_id, all_constants)
                # Replace the D5 stub in this env's diagnostics list
                diag_list = self._diagnostics.get(env_id, [])
                self._diagnostics[env_id] = [
                    d5 if d.diagnostic_id == "D5" else d
                    for d in diag_list
                ]
                if env_id in epoch_diagnostics:
                    epoch_diagnostics[env_id] = self._diagnostics[env_id]

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

    def get_output(self) -> dict:
        """Build the output dict from the agent's current internal state."""
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

        all_constants: dict[str, float] = {}
        for env_id in self.env_ids:
            best_sf = self.formula_store.get_best(env_id)
            if best_sf is not None:
                for i, c in enumerate(extract_constants(best_sf.expr)):
                    if abs(c) > 1e-6 and abs(c) != 1.0:
                        all_constants[f"{env_id}:C{i}"] = c

        concepts_out: dict[str, str] = {
            name: to_str(expr)
            for name, expr in self.dsl_state.concepts.items()
        }

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
            "epochs_run": self._current_epoch,
            "extensions": [e for e in self.dsl_state.extensions],
            "prob_mode_envs": list(self._stochastic_envs),
        }

    def run(self) -> dict:
        """Run the full ATLAS pipeline and return results."""
        # Collect data if not already done
        if not self.datasets:
            self.collect_data()

        last_result: Optional[EpochResult] = None

        for _ in range(self.config.max_epochs):
            last_result = self.run_epoch()
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

        return self.get_output()
