# ATLAS Architecture Guide

## Overview

ATLAS (Autonomous Theory Learning through Adaptive Search) is a multi-agent system that discovers physical laws and mathematical frameworks from raw experiment data. It starts with minimal mathematical knowledge (basic arithmetic + trig) and autonomously extends its symbolic language when existing tools fail to describe the data.

The system is **pure algorithmic** — no LLM, no pre-trained models, no natural language. This is by design: since we're validating the method on already-known physics (wave-particle duality), any knowledge leakage would invalidate the results.

```
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Agent Orchestrator                   │
│                   (Mode A / Mode B)                          │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    ┌──────────┐   │
│  │ Agent-1  │ │ Agent-2  │ │ Agent-3  │ .. │ Agent-N  │   │
│  │ ENV:01,  │ │ ENV:02,  │ │ ENV:03,  │    │ ENV:04,  │   │
│  │ 04,08,11 │ │ 05,07,10 │ │ 04,06,12 │    │ 07,10,11 │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘    └────┬─────┘   │
│       └─────────────┼───────────┼────────────────┘          │
│                     ▼           ▼                            │
│              ┌──────────────────────┐                        │
│              │    Proposal Pool     │ (Mode B only)          │
│              │  + Global MDL Check  │                        │
│              └──────────┬───────────┘                        │
│                         ▼                                    │
│              ┌──────────────────────┐                        │
│              │       Unifier        │                        │
│              │  U1: Constants       │                        │
│              │  U2: Templates       │                        │
│              │  U3: Types           │                        │
│              └──────────┬───────────┘                        │
│                         ▼                                    │
│              ┌──────────────────────┐                        │
│              │   Theory Output      │                        │
│              │  (compression ratio) │                        │
│              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Structure

```
atlas/
├── types.py                    # Core dataclasses: KnobSpec, DetectorSpec, EnvSchema,
│                               #   FitMetrics, FormulaRecord
│
├── dsl/                        # Symbolic expression system
│   ├── operators.py            # Op enum (ADD,SUB,MUL,...), DSL_0 (10 base operators)
│   ├── expr.py                 # AST nodes: Const, Var, BinOp, UnaryOp, NAryOp
│   ├── canonicalize.py         # Alpha-equivalence + commutativity sort +
│   │                           #   associativity flattening + identity elimination +
│   │                           #   constant folding
│   └── serialize.py            # S-expression and dict serialization
│
├── environments/               # 12 anonymized experiment simulators
│   ├── base.py                 # BaseEnvironment ABC with knob validation
│   ├── registry.py             # @register decorator, get_environment(), get_all_environments()
│   ├── normalizer.py           # normalize() / denormalize() utilities
│   ├── alt_physics.py          # altered_physics() context manager for h→2h validation
│   ├── env_01_photoelectric.py # Quantum: E = hf - W (cutoff behavior)
│   ├── env_02_compton.py       # Quantum: Δλ = (h/mc)(1-cosθ)
│   ├── env_03_electron_diffraction.py  # Quantum: λ = h/p (Bragg peaks)
│   ├── env_04_double_slit.py   # Quantum: wave-particle duality (KEY experiment)
│   ├── env_05_blackbody.py     # Quantum: Planck distribution
│   ├── env_06_hydrogen_spectrum.py  # Quantum: Rydberg formula
│   ├── env_07_stern_gerlach.py # Quantum: spin quantization (KEY experiment)
│   ├── env_08_water_wave.py    # Classical control: wave interference
│   ├── env_09_elastic_collision.py  # Classical control: particle mechanics
│   ├── env_10_spring.py        # Classical control: SHM
│   ├── env_11_freefall.py      # Distractor: gravity (should not trigger quantum)
│   └── env_12_heat_conduction.py    # Distractor: Fourier's law
│
├── data/
│   └── dataset.py              # ExperimentDataset: collect data by sweeping knobs,
│                               #   train/test split, from_env() grid/random sampling
│
├── sr/                         # Symbolic regression
│   ├── pysr_wrapper.py         # PySR integration: SRConfig, run_sr(),
│   │                           #   pysr_expr_to_atlas() recursive descent parser
│   └── formula_store.py        # FormulaStore: add/get/get_best, Pareto front,
│                               #   _extract_constants()
│
├── analysis/                   # Analysis modules
│   ├── concepts.py             # DreamCoder-style: extract_concepts() with MDL savings
│   ├── diagnostics.py          # D1 (stochasticity), D2 (discreteness),
│   │                           #   D4 (residual structure); D3/D5 deferred
│   └── pslq_unifier.py         # find_constant_relations() in log-space with sign sep,
│                               #   unify_constants() by approximate value grouping
│
├── scinet/                     # SciNet information bottleneck
│   ├── model.py                # SciNet(input_dim, bottleneck_dim, output_dim) PyTorch
│   ├── trainer.py              # train_scinet() with MSE + optional L1 sparsity
│   └── bottleneck.py           # find_optimal_k() via AIC, extract_bottleneck_vectors()
│
├── rgde/                       # RGDE: Representation-Grounded DSL Extension
│   ├── encoder_sr.py           # Step 4b: SR on encoder outputs (z_k = f_k(knobs))
│   ├── constraint_finder.py    # Step 4c: polynomial constraint discovery on z-space
│   ├── type_builder.py         # Step 4d: DSLType construction
│   ├── evaluator.py            # Step 4f: Pareto evaluation (ΔR² vs ΔMDL)
│   └── pipeline.py             # Steps 4a-4f: full RGDE orchestrator
│
├── agent/                      # Single-agent orchestration
│   ├── dsl_state.py            # DSLState: operators + concepts + extensions + snapshot
│   └── atlas_agent.py          # ATLASAgent: 5-step loop (Solve→Extract→Diagnose→
│                               #   Extend→Unify), AgentConfig, collect_data(), run()
│
├── multi_agent/                # Multi-agent consensus
│   ├── assignment.py           # generate_assignment(): random with coverage/mixing
│   ├── proposal.py             # Proposal, ProposalPool, ProposalStatus
│   ├── verifier.py             # compute_global_mdl_delta() with statistical significance
│   └── orchestrator.py         # MultiAgentOrchestrator: Mode A (independent) /
│                               #   Mode B (consensus sharing), _build_theory()
│
└── unifier/                    # Theory synthesis
    ├── theory.py               # Theory: law templates, shared constants, types,
    │                           #   compression chain, compression_ratio()
    ├── constant_unifier.py     # U1: cross-agent PSLQ with deduplication
    ├── template_extractor.py   # U2: anti_unify() + extract_templates() with MDL filter
    └── type_unifier.py         # U3: are_types_isomorphic(), unify_types()
```

### Module Dependency Graph

```
environments  ←  data  ←  sr  ←  analysis  ←  agent  ←  multi_agent
     ↑              ↑       ↑                    ↑           ↑
    dsl            dsl     dsl                 rgde       unifier
                                                ↑
                                             scinet
```

Each arrow means "depends on." No circular dependencies. `dsl` and `environments` are leaf modules.

---

## How to Run

### Prerequisites

```bash
# Python 3.11+
python --version

# Core dependencies
pip install numpy scipy

# Test dependencies
pip install pytest pytest-cov

# Symbolic regression (optional — tests skip gracefully without it)
pip install pysr
python -c "import pysr; pysr.install()"  # installs Julia backend

# Neural networks (optional — RGDE/SciNet tests skip without it)
pip install torch
```

### Running Tests

```bash
# Full suite (fast — skips PySR/PyTorch tests if not installed)
python -m pytest tests/ -v

# Only DSL tests (always fast, no optional deps)
python -m pytest tests/dsl/ -v

# Only environment tests (fast, no optional deps)
python -m pytest tests/environments/ -v

# Anti-cheating audit (verifies no physics leakage)
python -m pytest tests/environments/test_anti_cheating.py -v

# Only analysis tests
python -m pytest tests/analysis/ -v

# SciNet tests (requires PyTorch)
python -m pytest tests/scinet/ -v

# RGDE tests (requires PyTorch, optionally PySR)
python -m pytest tests/rgde/ -v

# Multi-agent tests
python -m pytest tests/multi_agent/ -v

# Full suite with coverage
python -m pytest tests/ --cov=atlas --cov-report=term-missing
```

### Running the System

```python
# Single agent on classical experiments (no optional deps needed for structure)
from atlas.agent.atlas_agent import ATLASAgent, AgentConfig

agent = ATLASAgent(
    env_ids=["ENV_10", "ENV_11", "ENV_12"],
    config=AgentConfig(
        max_epochs=5,
        n_samples_per_knob=10,
        sr_niterations=40,
        sr_maxsize=20,
    ),
)
output = agent.run()
print(output["formulas"])
print(output["constants"])
print(output["diagnostics"])

# Multi-agent Mode A (independent)
from atlas.multi_agent.orchestrator import MultiAgentOrchestrator, MultiAgentConfig, RunMode

orch = MultiAgentOrchestrator(MultiAgentConfig(
    mode=RunMode.MODE_A,
    n_agents=6,
    max_epochs=10,
    agent_sr_niterations=40,
))
result = orch.run()
print(result.theory)

# Multi-agent Mode B (consensus sharing)
orch_b = MultiAgentOrchestrator(MultiAgentConfig(
    mode=RunMode.MODE_B,
    n_agents=6,
    max_epochs=10,
    agent_sr_niterations=40,
    agent_enable_rgde=True,
))
result_b = orch_b.run()
print(f"Proposals: {result_b.proposals_submitted} submitted, {result_b.proposals_adopted} adopted")
print(f"Compression ratio: {result_b.theory['compression_ratio']}")
```

### Alternative Physics Validation

```python
from atlas.environments.alt_physics import PhysicsConfig, altered_physics
from atlas.agent.atlas_agent import ATLASAgent, AgentConfig

# Run with h → 2h
with altered_physics(PhysicsConfig(h_multiplier=2.0)):
    agent = ATLASAgent(env_ids=["ENV_01", "ENV_04", "ENV_07"],
                       config=AgentConfig(max_epochs=5))
    output_2h = agent.run()

# Run with standard h
agent = ATLASAgent(env_ids=["ENV_01", "ENV_04", "ENV_07"],
                   config=AgentConfig(max_epochs=5))
output_1h = agent.run()

# Compare: same formula structure, different constant values
print("Standard h:", output_1h["constants"])
print("2h:        ", output_2h["constants"])
```

---

## Key Design Decisions

### Anti-Cheating

Every design choice must pass: **"Would this make sense if we didn't know the answer?"**

- **No LLM** — pure algorithmic pipeline, zero knowledge source beyond DSL_0 and data
- **Anonymous interfaces** — all knobs are `knob_0..N`, all detectors are `detector_0..M`
- **Normalized inputs** — continuous knobs in [0,1] or [-1,1], no physical units exposed
- **No physics operators** — DSL_0 has only `{+,-,*,/,sin,cos,exp,log,^,neg}`
- **Alternative physics test** — if h→2h gives different constants but same structure, the system is truly data-driven
- **Automated audit** — `test_anti_cheating.py` runs against all 12 environments on every test run

### Theory = Compression

A theory's quality is measured by **compression ratio** = independent MDL / unified MDL. Better theories compress more because they capture shared structure (laws) across experiments while separating experiment-specific parameters (initial conditions).

### Global MDL for Consensus

DSL extensions are adopted based on **global MDL change** across all 12 experiments, not per-experiment voting. This correctly handles search space expansion: adding `cos²` may slightly hurt ENV-09 (which doesn't need it) but greatly help ENV-02, ENV-04, ENV-08.

---

## Next Steps

### Phase 0: SciNet→SR Bridge Validation (Critical Gate)

**This must be done before any other work.** The RGDE pipeline's core assumption is that SciNet's learned representation can be extracted into symbolic formulas via SR. If this bridge fails, the entire RGDE approach needs rethinking.

```
Validation experiment:
  1. Generate Stern-Gerlach data (ENV-07) with known Bloch sphere geometry
  2. Train SciNet, verify AIC selects K=3
  3. Run SR on encoder outputs — can it recover the Bloch parameterization?
  4. Search for constraint z₁² + z₂² + z₃² ≤ 1
  5. Run SR on decoder — can it recover the measurement formula?

Gate criteria:
  - Steps 3-4 succeed in >50% of multi-seed runs → CONTINUE
  - Steps 3-4 succeed in <30% of runs → MODIFY approach or DESCOPE RGDE
```

**How to run:**
```python
from atlas.environments.registry import get_environment
from atlas.data.dataset import ExperimentDataset
from atlas.rgde.pipeline import run_rgde, RGDEConfig

env = get_environment("ENV_07")
ds = ExperimentDataset.from_env(env, n_samples_per_knob=20, seed=42)
X = ds.knob_array()
y = ds.detector_array("detector_0")

result = run_rgde(X, y, var_names=ds.knob_names, r2_before=0.3,
                  env_id="ENV_07", config=RGDEConfig(k_range=[1,2,3,4,5]))
print(f"K selected: {result.k_selected}")
print(f"Encoder R²: {result.encoder_r2}")
print(f"Decoder R²: {result.r2_after}")
print(f"Accepted: {result.evaluation.accepted if result.evaluation else 'N/A'}")
```

### Phase 1: Single-Agent Formula Discovery

**Goal:** Verify that PySR can discover E=hf-level formulas from anonymized experiment data.

```
Steps:
  1. Install PySR + Julia
  2. Run single agent on ENV-01 (photoelectric) with generous SR budget
     → expect: meter = C₁ * max(knob_0 - C₂, 0) * knob_1
  3. Run on ENV-10 (spring) as control
     → expect: detector_0 = C₃ * cos(C₄ * knob_0) with high R²
  4. Run on ENV-04 (double slit, high intensity) 
     → expect: cos²-like pattern in detector array
  5. Verify that classical experiments do NOT trigger RGDE diagnostics

Success criteria:
  - ENV-01, ENV-10: R² > 0.95 on test set
  - ENV-04 (high intensity): R² > 0.90 on test set
  - ENV-10, ENV-11: no D1/D2 diagnostics triggered
```

### Phase 2: Constant Unification

**Goal:** Verify that PSLQ finds h across multiple quantum experiments.

```
Steps:
  1. Run single agent on ENV-01 + ENV-02 + ENV-05 (all use h internally)
  2. Extract constants from best formulas
  3. Run PSLQ — does it find a shared constant?
  4. Run with h→2h — does the shared constant change to 2h?

Success criteria:
  - PSLQ groups constants from ≥3 experiments into one universal constant
  - Alternative physics test: same formula structure, different constant value
```

### Phase 3: Multi-Agent Validation

**Goal:** Verify the multi-agent architecture produces consistent results.

```
Steps:
  1. Run Mode A with 20+ random assignment seeds
     → do agents independently converge to equivalent formulas?
  2. Run Mode B with same seeds
     → does consensus sharing accelerate convergence?
  3. Compare Mode A vs Mode B:
     - Convergence speed (epochs to R² > 0.95)
     - Compression ratio
     - Constant unification degree
  4. Overlap consistency: for experiments covered by 2+ agents,
     verify formula structures are equivalent (after canonicalization)

Success criteria:
  - >60% of seeds produce equivalent theories in Mode A
  - Mode B converges in fewer epochs than Mode A (or equal)
  - Overlapping experiment formulas match after canonicalization
```

### Phase 4: Full Pipeline — Wave-Particle Duality Discovery

**Goal:** Run the complete system and evaluate what it discovers.

```
Steps:
  1. Run Mode B, 6 agents, max 20 epochs, RGDE enabled
  2. Examine Theory output:
     - What universal constants were found?
     - What law templates were extracted?
     - Were any new DSL types discovered (BlochState)?
     - What is the compression ratio?
  3. Map structural facts to physics:
     F1: "some experiments are stochastic" → quantum probabilistic behavior
     F2: "cos² appears in both intensity and probability formulas" → Born rule
     F3: "a universal constant appears in 5+ experiments" → Planck constant
     F4: "K=3 > N-1=1 for some two-state systems" → Bloch sphere
  4. Run full validation protocol (V1-V10 from design spec)
```

### Known Limitations to Address

| Issue | Description | Priority |
|-------|-------------|----------|
| **Array output SR** | Current agent uses `np.mean(y, axis=1)` as scalar proxy for array outputs | High — need per-position SR or pattern-based SR |
| **Mode B verification** | Orchestrator auto-adopts proposals without re-running SR on all experiments | High — implement proper global MDL verification loop |
| **D3 (dimension insufficiency)** | Deferred — requires SciNet integration in diagnostics | Medium |
| **D5 (cross-experiment inconsistency)** | Deferred — requires Unifier integration in diagnostics | Medium |
| **Parallel execution** | Agents run sequentially; should use `concurrent.futures` | Low — correctness first |
| **SciNet robustness** | Encoder sparsity tuning needed for reliable SR extraction | Phase 0 will reveal |
| **PSLQ vs approximate grouping** | Current `unify_constants()` uses value grouping, not true PSLQ integer relations | Medium — `mpmath.pslq` for production |

---

## Design Documents

| Document | Location | Content |
|----------|----------|---------|
| ATLAS Proposal | `ATLAS_proposal.md` | Core system design, 5-step loop, experiments, validation |
| Anti-Cheating Audit | `anti_cheating_audit.md` | Line-by-line audit of knowledge leakage risks |
| Multi-Agent Spec | `docs/superpowers/specs/2026-03-31-multi-agent-atlas-design.md` | Multi-agent consensus, Unifier, Theory output format |
| Literature Survey | `literature_survey.md` | Related work: AI-Newton, SciNet, DreamCoder, PySR, etc. |
| Plan 1 | `docs/superpowers/plans/2026-04-01-plan1-foundation-environments.md` | Foundation + 12 environments |
| Plan 2 | `docs/superpowers/plans/2026-04-01-plan2-single-agent-core.md` | SR + concepts + diagnostics + agent loop |
| Plan 3 | `docs/superpowers/plans/2026-04-01-plan3-scinet-rgde.md` | SciNet + RGDE pipeline |
| Plan 4 | `docs/superpowers/plans/2026-04-01-plan4-multi-agent-unifier.md` | Multi-agent + Unifier |

---

## Test Summary

**352 tests** across 8 modules, running in ~67 seconds.

```
Module              Tests   Requires
─────────────────────────────────────
dsl/                  39    nothing
environments/        170    numpy
data/                   7    numpy + environments
sr/                    17    numpy (PySR optional)
analysis/              19    numpy
scinet/                13    PyTorch
rgde/                  15    PyTorch (PySR optional)
agent/                 11    all above
multi_agent/           24    all above
unifier/               19    numpy
integration/            4    all above
─────────────────────────────────────
anti-cheating           5    runs against all 12 envs
alt-physics             4    modifies constants at runtime
```

Without PySR and PyTorch: ~320 tests pass, ~30 skipped.
With all dependencies: 350 pass, 2 skipped (PySR-specific integration tests).
