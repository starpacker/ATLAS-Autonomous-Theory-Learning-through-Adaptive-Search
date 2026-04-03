# atlas/multi_agent/orchestrator.py
"""Multi-agent orchestrator: runs multiple ATLAS agents and synthesizes results.

Mode A: Fully independent — agents run in isolation, results compared post-hoc
Mode B: Consensus sharing — agents share DSL extensions via proposal pool + verification
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from atlas.agent.atlas_agent import ATLASAgent, AgentConfig, _expand_array_output
from atlas.dsl.expr import Expr
from atlas.dsl.serialize import from_str, to_str
from atlas.multi_agent.assignment import (
    generate_assignment, AssignmentConfig, AgentAssignment,
)
from atlas.multi_agent.proposal import Proposal, ProposalPool, ProposalStatus
from atlas.multi_agent.verifier import compute_global_mdl_delta, verify_proposal_sr
from atlas.rgde.constraint_finder import Constraint
from atlas.rgde.type_builder import DSLType
from atlas.unifier.constant_unifier import (
    unify_agent_constants, AgentConstants,
)
from atlas.unifier.template_extractor import extract_templates
from atlas.unifier.theory import Theory, CompressionLayer, LawTemplate
from atlas.unifier.type_unifier import unify_types
from atlas.sr.formula_store import extract_constants

logger = logging.getLogger(__name__)


def _reconstruct_dsl_type(defn: dict) -> DSLType | None:
    """Reconstruct a DSLType from its serialized definition dict."""
    name = defn.get("name", "")
    dimension = defn.get("dimension", 0)
    source_env = defn.get("source_env", "")

    if not name or dimension <= 0:
        return None

    encoding: dict[int, Expr] = {}
    for k, v in defn.get("encoding", {}).items():
        try:
            encoding[int(k)] = from_str(v)
        except Exception:
            pass

    constraints: list[Constraint] = []
    for c_dict in defn.get("constraints", []):
        raw_terms = c_dict.get("terms", [])
        terms = [tuple(t) if isinstance(t, list) else t for t in raw_terms]
        degree = max((len(t) for t in terms), default=0)
        constraints.append(Constraint(
            coefficients=np.array([]),
            terms=terms,
            degree=degree,
            constant=c_dict.get("constant", 0.0),
            residual=c_dict.get("residual", 0.0),
            constraint_type=c_dict.get("type", "equality"),
        ))

    return DSLType(
        name=name,
        dimension=dimension,
        encoding=encoding,
        constraints=constraints,
        source_env=source_env,
    )


class RunMode(Enum):
    MODE_A = "mode_a"  # fully independent
    MODE_B = "mode_b"  # consensus sharing


@dataclass
class MultiAgentConfig:
    mode: RunMode = RunMode.MODE_A
    n_agents: int = 6
    max_epochs: int = 10
    min_envs_per_agent: int = 3
    min_coverage: int = 2
    assignment_seed: int = 42
    # Per-agent config
    agent_n_samples_per_knob: int = 10
    agent_sr_niterations: int = 40
    agent_sr_maxsize: int = 25
    agent_sr_timeout: int = 300
    agent_enable_rgde: bool = False
    agent_r_squared_threshold: float = 0.95
    # Verification (Mode B only)
    verification_seeds: int = 5


@dataclass
class MultiAgentResult:
    mode: RunMode
    n_agents: int
    agent_outputs: list[dict]
    theory: dict
    output: dict
    proposals_submitted: int = 0
    proposals_adopted: int = 0
    proposals_rejected: int = 0


class MultiAgentOrchestrator:
    """Orchestrates multiple ATLAS agents."""

    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.proposal_pool = ProposalPool()

        # Generate assignments
        assign_config = AssignmentConfig(
            n_agents=config.n_agents,
            min_envs_per_agent=config.min_envs_per_agent,
            min_coverage=config.min_coverage,
            seed=config.assignment_seed,
        )
        self.assignments = generate_assignment(assign_config)

        # Create agents
        self.agents: list[ATLASAgent] = []
        for assignment in self.assignments:
            agent_cfg = AgentConfig(
                max_epochs=config.max_epochs,
                r_squared_threshold=config.agent_r_squared_threshold,
                n_samples_per_knob=config.agent_n_samples_per_knob,
                sr_niterations=config.agent_sr_niterations,
                sr_maxsize=config.agent_sr_maxsize,
                sr_timeout=config.agent_sr_timeout,
                seed=assignment.seed,
                enable_rgde=config.agent_enable_rgde,
            )
            agent = ATLASAgent(
                env_ids=assignment.env_ids,
                config=agent_cfg,
            )
            self.agents.append(agent)

    def run(self) -> MultiAgentResult:
        """Run all agents and synthesize results."""
        if self.config.mode == RunMode.MODE_A:
            return self._run_mode_a()
        else:
            return self._run_mode_b()

    def _run_mode_a(self) -> MultiAgentResult:
        """Mode A: Run agents fully independently, then unify post-hoc."""
        agent_outputs = []
        for i, agent in enumerate(self.agents):
            logger.info(f"Running agent {i} on {agent.env_ids}")
            output = agent.run()
            agent_outputs.append(output)

        # Post-hoc unification
        theory = self._build_theory(agent_outputs)

        return MultiAgentResult(
            mode=RunMode.MODE_A,
            n_agents=len(self.agents),
            agent_outputs=agent_outputs,
            theory=theory.to_dict(),
            output={"theory": theory.to_dict(), "agent_outputs": agent_outputs},
        )

    def _run_mode_b(self) -> MultiAgentResult:
        """Mode B: Interleaved epoch-by-epoch execution with extension sharing.

        Instead of running each agent to completion, we run one epoch at a
        time across all agents.  After each round of epochs:
        1. Collect new extension proposals from this round
        2. Verify proposals using global MDL criterion
        3. Inject adopted extensions into ALL agents' DSL states
        4. Repeat until convergence or max_epochs

        This lets discoveries from early rounds benefit all agents in later
        rounds — the core value of consensus sharing.
        """
        # Phase 1: All agents collect data independently
        for agent in self.agents:
            agent.collect_data()

        # Track which extensions we have already proposed (by name)
        proposed_ext_names: set[str] = set()

        # Phase 2: Interleaved epoch-by-epoch execution
        all_converged = False
        epochs_run = 0
        mdl_history: list[float] = []
        epochs_without_adoption = 0

        for epoch in range(self.config.max_epochs):
            logger.info(f"Mode B: epoch {epoch}")
            epoch_results = []
            for i, agent in enumerate(self.agents):
                result = agent.run_epoch()
                epoch_results.append(result)

            epochs_run += 1

            # Collect NEW extension proposals from this epoch
            for i, result in enumerate(epoch_results):
                for ext_name in result.extensions_found:
                    if ext_name in proposed_ext_names:
                        continue
                    proposed_ext_names.add(ext_name)
                    ext = next(
                        (e for e in self.agents[i].dsl_state.extensions
                         if e["name"] == ext_name), None)
                    if ext is None:
                        continue

                    evidence: dict = {}
                    for env_id, fm in self.agents[i]._fit_metrics.items():
                        evidence[env_id] = {
                            "r_squared": fm.r_squared,
                            "mdl": fm.mdl,
                        }

                    ext_def = dict(ext.get("definition", {}))
                    ext_def["source_env"] = ext.get("source_env", "")
                    ext_def["delta_r2"] = ext.get("delta_r2")
                    ext_def["r2_before"] = ext.get("r2_before")
                    ext_def["r2_after"] = ext.get("r2_after")

                    proposal = Proposal(
                        proposal_id=f"PROP-{self.assignments[i].agent_id}-{ext_name}-E{epoch}",
                        source_agent=self.assignments[i].agent_id,
                        source_env=ext.get("source_env", ext.get("trigger", "unknown")),
                        trigger=ext.get("trigger", ""),
                        extension_type=ext.get("type", "unknown"),
                        extension_definition=ext_def,
                        evidence=evidence,
                    )
                    self.proposal_pool.add(proposal)

            # Build global fit snapshot (deduplicated by experiment)
            global_fit: dict[str, dict] = {}
            for agent in self.agents:
                for env_id, fm in agent._fit_metrics.items():
                    r2 = fm.r_squared
                    mdl = fm.mdl
                    if env_id not in global_fit or r2 > global_fit[env_id].get("r_squared", -1.0):
                        global_fit[env_id] = {"r_squared": r2, "mdl": mdl}

            # Verify pending proposals (track new adoptions)
            # Strategy: try SR-based verification on envs where agents have
            # datasets; fall back to estimate-based for remaining envs.
            n_adopted_before = len(self.proposal_pool.adopted())
            for proposal in self.proposal_pool.pending():
                per_env_deltas = self._verify_proposal(
                    proposal, global_fit, epoch)
                if per_env_deltas:
                    verification = compute_global_mdl_delta(per_env_deltas)
                    status = (ProposalStatus.ADOPTED if verification.should_adopt
                              else ProposalStatus.REJECTED)
                    self.proposal_pool.set_status(
                        proposal.proposal_id, status,
                        delta_total_mdl=verification.delta_total_mdl,
                        verification_details={"reason": verification.reason,
                                               "per_env": verification.per_env_results},
                    )
                else:
                    self.proposal_pool.set_status(
                        proposal.proposal_id, ProposalStatus.REJECTED,
                        delta_total_mdl=0.0,
                        verification_details={"reason": "Insufficient evidence"},
                    )

            # Inject adopted extensions into ALL agents' DSL states.
            # Also extract encoder formulas as concepts so they actually
            # influence subsequent SR runs via _augment_with_concepts.
            for proposal in self.proposal_pool.adopted():
                ext_def = proposal.extension_definition
                ext_name = ext_def.get("name", proposal.proposal_id)
                for agent in self.agents:
                    # add_extension deduplicates by name
                    agent.dsl_state.add_extension(
                        name=ext_name,
                        ext_type=proposal.extension_type,
                        definition=ext_def,
                        trigger=f"Adopted from {proposal.source_agent}",
                    )
                    # Convert encoder formulas to concepts so SR can use them.
                    # Encoding maps {dim_index: expr_string} — parse each into
                    # a concept that _augment_with_concepts can evaluate.
                    encoding = ext_def.get("encoding", {})
                    for dim_key, expr_str in encoding.items():
                        concept_name = f"{ext_name}_z{dim_key}"
                        if concept_name not in agent.dsl_state.concepts:
                            try:
                                from atlas.dsl.serialize import from_str
                                concept_expr = from_str(expr_str)
                                agent.dsl_state.add_concept(concept_name, concept_expr)
                            except Exception:
                                pass  # skip unparseable encodings

            # ---- Convergence criteria ----
            # Track adoptions
            n_adopted_after = len(self.proposal_pool.adopted())
            if n_adopted_after == n_adopted_before:
                epochs_without_adoption += 1
            else:
                epochs_without_adoption = 0

            # Track total best MDL across all unique envs
            best_mdl_per_env: dict[str, float] = {}
            for agent in self.agents:
                for env_id, fm in agent._fit_metrics.items():
                    mdl = fm.mdl
                    if mdl != float("inf") and mdl > 0:
                        if env_id not in best_mdl_per_env or mdl < best_mdl_per_env[env_id]:
                            best_mdl_per_env[env_id] = mdl
            total_mdl = sum(best_mdl_per_env.values()) if best_mdl_per_env else float("inf")
            mdl_history.append(total_mdl)

            # (C1) All envs converged AND compression stable for 3 epochs
            all_converged = all(
                set(r.converged_envs) >= set(agent.env_ids)
                for r, agent in zip(epoch_results, self.agents)
            )
            compression_stable = False
            if len(mdl_history) >= 3 and mdl_history[-3] > 0:
                recent = mdl_history[-3:]
                max_change = max(
                    abs(recent[i] - recent[i - 1]) / recent[0]
                    for i in range(1, len(recent))
                )
                compression_stable = max_change < 0.01

            if all_converged and compression_stable:
                logger.info("Mode B: converged at epoch %d "
                            "(all R²>threshold, compression stable)", epoch)
                break

            # (C2) No adoptions for 5 epochs (discovery stalled)
            if epochs_without_adoption >= 5:
                logger.info("Mode B: stopping at epoch %d "
                            "(no adoptions for 5 epochs)", epoch)
                break

        # Phase 3: Build agent outputs and theory
        agent_outputs = []
        for agent in self.agents:
            agent_outputs.append(agent.get_output())

        theory = self._build_theory(agent_outputs)

        return MultiAgentResult(
            mode=RunMode.MODE_B,
            n_agents=len(self.agents),
            agent_outputs=agent_outputs,
            theory=theory.to_dict(),
            output={"theory": theory.to_dict(), "agent_outputs": agent_outputs,
                     "proposals": [p.proposal_id for p in self.proposal_pool.all_proposals()]},
            proposals_submitted=len(self.proposal_pool.all_proposals()),
            proposals_adopted=len(self.proposal_pool.adopted()),
            proposals_rejected=len(self.proposal_pool.rejected()),
        )

    def _verify_proposal(
        self,
        proposal: Proposal,
        global_fit: dict[str, dict],
        epoch: int,
    ) -> dict[str, dict]:
        """Verify a proposal using SR-based testing where possible, estimates elsewhere.

        For each experiment where an agent has a dataset:
          1. Parse the extension's encoder formulas into concept columns
          2. Run ``verify_proposal_sr()`` with M seeds (SR with/without concepts)
          3. Use the paired delta-MDL as the per-experiment evidence

        For experiments not covered by any agent's dataset, fall back to
        the estimate-based heuristic.

        Returns {env_id: {"mu": ..., "sigma": ..., "method": "sr"|"estimate"}}.
        """
        per_env_deltas: dict[str, dict] = {}
        ext_def = proposal.extension_definition
        encoding = ext_def.get("encoding", {})

        # Collect all unique env_ids across agents (with their datasets)
        env_to_agent: dict[str, int] = {}
        for i, agent in enumerate(self.agents):
            for env_id in agent.env_ids:
                if env_id in agent.datasets and len(agent.datasets[env_id]) > 0:
                    # Prefer the first agent we find with data
                    if env_id not in env_to_agent:
                        env_to_agent[env_id] = i

        # Try SR-based verification on envs where we have data
        sr_verified_envs: set[str] = set()
        if encoding:
            for env_id, agent_idx in env_to_agent.items():
                agent = self.agents[agent_idx]
                ds = agent._prob_datasets.get(env_id, agent.datasets.get(env_id))
                if ds is None or len(ds) == 0:
                    continue

                try:
                    train_ds, test_ds = ds.split(
                        test_fraction=0.2, seed=agent.config.seed)
                    X_train = train_ds.knob_array()
                    X_test = test_ds.knob_array()
                    first_det = ds.detector_names[0]
                    y_train_raw = train_ds.detector_array(first_det)
                    y_test_raw = test_ds.detector_array(first_det)
                    var_names = train_ds.knob_names

                    # Use the same array expansion as the agent so
                    # verification tests against the same target.
                    is_array = y_train_raw.ndim > 1 and y_train_raw.shape[1] > 1
                    if is_array:
                        X_train, y_train, var_names = _expand_array_output(
                            X_train, y_train_raw, var_names,
                            seed=agent.config.seed,
                        )
                        X_test, y_test, _ = _expand_array_output(
                            X_test, y_test_raw, test_ds.knob_names,
                            seed=agent.config.seed,
                        )
                    else:
                        y_train = y_train_raw.ravel() if y_train_raw.ndim > 1 else y_train_raw
                        y_test = y_test_raw.ravel() if y_test_raw.ndim > 1 else y_test_raw

                    # Build concept columns from extension encoding
                    concept_cols_train, concept_cols_test = (
                        self._build_concept_columns(
                            encoding, X_train, X_test, var_names))

                    if concept_cols_train:
                        result = verify_proposal_sr(
                            X_train, y_train.astype(float),
                            X_test, y_test.astype(float),
                            var_names,
                            concept_columns=concept_cols_train,
                            concept_columns_test=concept_cols_test,
                            n_seeds=self.config.verification_seeds,
                            sr_niterations=max(self.config.agent_sr_niterations // 2, 10),
                            sr_maxsize=self.config.agent_sr_maxsize,
                            base_seed=agent.config.seed + epoch * 100,
                        )
                        if result is not None:
                            per_env_deltas[env_id] = result
                            sr_verified_envs.add(env_id)
                except Exception as exc:
                    logger.debug("SR verification failed for %s: %s", env_id, exc)

        # Fall back to estimate-based for remaining envs
        estimate_deltas = self._estimate_proposal_impact(
            proposal, global_fit, [])
        for env_id, delta in estimate_deltas.items():
            if env_id not in sr_verified_envs:
                delta["method"] = "estimate"
                per_env_deltas[env_id] = delta

        return per_env_deltas

    @staticmethod
    def _build_concept_columns(
        encoding: dict,
        X_train: np.ndarray,
        X_test: np.ndarray,
        var_names: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Parse encoder formulas and compute concept columns for SR.

        Returns (concept_cols_train, concept_cols_test).
        """
        from atlas.dsl.serialize import from_str as _from_str

        cols_train: dict[str, np.ndarray] = {}
        cols_test: dict[str, np.ndarray] = {}

        for dim_key, expr_str in encoding.items():
            cname = f"z{dim_key}"
            try:
                expr = _from_str(expr_str)
            except Exception:
                continue

            for label, X_data, target in [("train", X_train, cols_train),
                                           ("test", X_test, cols_test)]:
                vals = np.empty(X_data.shape[0], dtype=float)
                for i in range(X_data.shape[0]):
                    try:
                        env_dict = {var_names[j]: float(X_data[i, j])
                                    for j in range(min(len(var_names), X_data.shape[1]))}
                        vals[i] = expr.evaluate(env_dict)
                    except Exception:
                        vals[i] = np.nan
                if np.all(np.isfinite(vals)):
                    target[cname] = vals

        return cols_train, cols_test

    @staticmethod
    def _estimate_proposal_impact(
        proposal: Proposal,
        global_fit: dict[str, dict],
        agent_outputs: list[dict],
    ) -> dict[str, dict]:
        """Estimate per-environment MDL delta for a proposed extension.

        Design note: the spec calls for running SR M times with and without
        the extension, then a paired t-test.  That requires ~5000 GPU-hours
        per verification round, so we use an estimate-based approach instead:
        compare existing fit metrics with and without the extension's
        contribution.  This is a pragmatic simplification that preserves the
        decision logic (adopt if delta_total_MDL < 0) without the compute cost.

        Only includes environments where we have actual evidence:
        - Source environment: use the actual RGDE marginal delta, plus a
          DSL expansion cost.
        - Environments covered by the proposing agent (in evidence): compare
          their fit to the global best.

        Environments not covered by the proposing agent are EXCLUDED — we
        have zero information about them, and including them with mu=0
        would only inflate pooled noise and mask real signals.

        Returns {env_id: {"mu": mean_delta, "sigma": noise_estimate}}.
        """
        per_env_deltas: dict[str, dict] = {}

        # DSL expansion cost for adding a new type to the shared DSL
        dsl_expansion_penalty = 1.0

        ext_def = proposal.extension_definition
        source_env = ext_def.get("source_env") or proposal.source_env
        rgde_delta_r2 = ext_def.get("delta_r2")

        if source_env and rgde_delta_r2 is not None:
            baseline = global_fit.get(source_env, {})
            baseline_mdl = baseline.get("mdl", float("inf"))
            if baseline_mdl == float("inf") or baseline_mdl <= 0:
                baseline_mdl = 20.0

            # Convert R² improvement to MDL benefit
            mdl_benefit = rgde_delta_r2 * baseline_mdl
            delta = -mdl_benefit + dsl_expansion_penalty
            sigma = max(baseline_mdl * 0.05, 0.3)
            per_env_deltas[source_env] = {"mu": delta, "sigma": sigma}

        # Also include other environments covered by the proposing agent
        # (from evidence) where the agent's fit differs from global best.
        for env_id, ev in proposal.evidence.items():
            if env_id == source_env:
                continue  # already handled above
            baseline = global_fit.get(env_id, {})
            baseline_mdl = baseline.get("mdl", float("inf"))
            if baseline_mdl == float("inf") or baseline_mdl <= 0:
                continue  # no baseline to compare against
            ev_mdl = ev.get("mdl", float("inf"))
            if ev_mdl == float("inf") or ev_mdl <= 0:
                continue
            delta = ev_mdl - baseline_mdl  # negative = agent did better
            sigma = max(baseline_mdl * 0.1, 0.5)
            per_env_deltas[env_id] = {"mu": delta, "sigma": sigma}

        return per_env_deltas

    def _build_theory(self, agent_outputs: list[dict]) -> Theory:
        """Build unified theory from all agent outputs.

        Compression chain (4 layers):
          L0  independent formulas
          L1  constant unification   (U1)
          L2  template extraction    (U2)
          L3  type unification       (U3)
        """
        theory = Theory()

        # ---- Collect best formula Expr per env (needed for U2) ----
        best_formulas: dict[str, Expr] = {}
        best_r2_for_formula: dict[str, float] = {}
        for output in agent_outputs:
            for env_id, formula_str in output.get("formulas", {}).items():
                if not formula_str:
                    continue
                metrics = output.get("fit_metrics", {}).get(env_id, {})
                r2 = metrics.get("r_squared", -1.0) if isinstance(metrics, dict) else -1.0
                if env_id not in best_r2_for_formula or r2 > best_r2_for_formula[env_id]:
                    try:
                        best_formulas[env_id] = from_str(formula_str)
                        best_r2_for_formula[env_id] = r2
                    except Exception:
                        pass

        # ==== U1: Constant unification ====
        agent_constants = []
        for i, output in enumerate(agent_outputs):
            constants = output.get("constants", {})
            r_squared = {}
            for env_id, metrics in output.get("fit_metrics", {}).items():
                if isinstance(metrics, dict):
                    for key in constants:
                        if key.startswith(env_id):
                            r_squared[key] = metrics.get("r_squared", 0.0)
            agent_constants.append(AgentConstants(
                agent_id=self.assignments[i].agent_id,
                constants=constants,
                r_squared=r_squared,
            ))

        if agent_constants:
            unification = unify_agent_constants(agent_constants)
            for uc in unification.unified:
                theory.add_shared_constant(
                    symbol=uc.symbol, value=uc.value,
                    uncertainty=uc.uncertainty,
                    appearances=uc.appearances,
                    chi2_consistency=uc.chi2_pvalue if uc.chi2_pvalue is not None else 0.0,
                )

        # ---- L0 + L1: compression accounting ----
        best_mdl_per_env: dict[str, float] = {}
        for output in agent_outputs:
            for env_id, metrics in output.get("fit_metrics", {}).items():
                if isinstance(metrics, dict):
                    mdl_val = metrics.get("mdl", 0.0)
                    if mdl_val != float("inf") and mdl_val > 0:
                        if env_id not in best_mdl_per_env or mdl_val < best_mdl_per_env[env_id]:
                            best_mdl_per_env[env_id] = mdl_val

        total_independent_mdl = sum(best_mdl_per_env.values())

        constant_savings = 0.0
        for sc in theory.shared_constants:
            n_apps = len(sc.get("appearances", []) if isinstance(sc, dict)
                          else getattr(sc, "appearances", []))
            if n_apps > 1:
                constant_savings += (n_apps - 1)

        n_shared = len(theory.shared_constants)
        constant_definition_cost = n_shared * 1.0
        total_after_u1 = total_independent_mdl - constant_savings + constant_definition_cost

        theory.add_compression_layer(CompressionLayer(
            level=0, total_mdl=max(total_independent_mdl, 1.0),
            label="independent formulas", delta=0.0,
        ))
        if total_independent_mdl > 0:
            theory.add_compression_layer(CompressionLayer(
                level=1, total_mdl=max(total_after_u1, 0.1),
                label="constant unification",
                delta=total_after_u1 - total_independent_mdl,
            ))

        # ==== U2: Template extraction ====
        template_savings = 0.0
        if len(best_formulas) >= 2:
            try:
                templates = extract_templates(best_formulas)
                for idx, tmpl in enumerate(templates):
                    # Cross-reference shared constants with template envs
                    shared_const_names: list[str] = []
                    for sc in theory.shared_constants:
                        apps = sc.get("appearances", []) if isinstance(sc, dict) else []
                        for env_id in tmpl.env_ids:
                            if any(a.startswith(env_id) for a in apps):
                                shared_const_names.append(
                                    sc.get("symbol", "") if isinstance(sc, dict) else "")
                                break

                    law = LawTemplate(
                        template_id=f"T{idx}",
                        template_str=to_str(tmpl.template),
                        shared_constants=list(set(shared_const_names)),
                        applies_to=tmpl.env_ids,
                        compression_savings=tmpl.savings,
                    )
                    theory.add_law_template(law)
                    template_savings += tmpl.savings

                    # Record per-env template bindings
                    for env_id, bindings in tmpl.bindings.items():
                        if env_id not in theory.experiment_bindings:
                            theory.experiment_bindings[env_id] = {}
                        theory.experiment_bindings[env_id]["template_id"] = law.template_id
                        theory.experiment_bindings[env_id]["template_bindings"] = {
                            h: to_str(expr) for h, expr in bindings.items()
                        }
            except Exception as exc:
                logger.warning("Template extraction (U2) failed: %s", exc)

        # ---- L2: template extraction savings ----
        if template_savings > 0 and len(theory.compression_chain) >= 2:
            prev_mdl = theory.compression_chain[-1].total_mdl
            total_after_u2 = prev_mdl - template_savings
            theory.add_compression_layer(CompressionLayer(
                level=2, total_mdl=max(total_after_u2, 0.1),
                label="template extraction",
                delta=-template_savings,
            ))

        # ==== U3: Type unification ====
        all_types: list[DSLType] = []
        for output in agent_outputs:
            for ext in output.get("extensions", []):
                if ext.get("type") != "new_type":
                    continue
                defn = ext.get("definition", {})
                try:
                    dsl_type = _reconstruct_dsl_type(defn)
                    if dsl_type is not None:
                        all_types.append(dsl_type)
                except Exception:
                    pass

        type_savings = 0.0
        if len(all_types) >= 2:
            try:
                type_result = unify_types(all_types)
                for ut in type_result.unified_types:
                    savings = ut.get("compression_savings", 0.0)
                    theory.add_shared_type(
                        name=ut["name"],
                        dimension=ut["dimension"],
                        constraints=ut["constraints"],
                        appears_in=ut["source_envs"],
                        compression_savings=savings,
                    )
                    type_savings += savings
            except Exception as exc:
                logger.warning("Type unification (U3) failed: %s", exc)

        # ---- L3: type unification savings ----
        if type_savings > 0:
            prev_mdl = theory.compression_chain[-1].total_mdl
            total_after_u3 = prev_mdl - type_savings
            theory.add_compression_layer(CompressionLayer(
                level=3, total_mdl=max(total_after_u3, 0.1),
                label="type unification",
                delta=-type_savings,
            ))

        # ==== Populate experiment_bindings (fit metrics + local constants) ====
        for output in agent_outputs:
            for env_id, metrics in output.get("fit_metrics", {}).items():
                if env_id not in theory.experiment_bindings:
                    theory.experiment_bindings[env_id] = {}
                if isinstance(metrics, dict):
                    existing_r2 = (theory.experiment_bindings[env_id]
                                   .get("fit_metrics", {})
                                   .get("r_squared", -1.0))
                    if metrics.get("r_squared", -1.0) > existing_r2:
                        theory.experiment_bindings[env_id]["fit_metrics"] = metrics
                # Local constants for this env
                env_constants = {
                    k: v for k, v in output.get("constants", {}).items()
                    if k.startswith(f"{env_id}:")
                }
                if env_constants:
                    existing = theory.experiment_bindings[env_id].get("local_constants", {})
                    existing.update(env_constants)
                    theory.experiment_bindings[env_id]["local_constants"] = existing

        # ==== Populate extension_lineage ====
        seen_ext_names: set[str] = set()
        for output in agent_outputs:
            for ext in output.get("extensions", []):
                ext_name = ext.get("name", "")
                if not ext_name or ext_name in seen_ext_names:
                    continue
                seen_ext_names.add(ext_name)
                lineage: dict = {
                    "name": ext_name,
                    "type": ext.get("type", "unknown"),
                    "trigger": ext.get("trigger", ""),
                    "source_env": ext.get("source_env", ""),
                }
                # Attach proposal status if this extension went through the pool
                for prop in self.proposal_pool.all_proposals():
                    if prop.extension_definition.get("name", "") == ext_name:
                        lineage["proposal_id"] = prop.proposal_id
                        lineage["status"] = prop.status.value
                        lineage["delta_mdl"] = prop.delta_total_mdl
                        break
                theory.extension_lineage.append(lineage)

        # Collect fit metrics (top-level, for backwards compat)
        for output in agent_outputs:
            for env_id, metrics in output.get("fit_metrics", {}).items():
                if env_id not in theory.fit_metrics:
                    theory.fit_metrics[env_id] = metrics

        return theory
