# ATLAS — Autonomous Theory Learning through Adaptive Search

A multi-agent physics-discovery system that starts with **minimal mathematical
knowledge** (basic arithmetic + trig) and **autonomously extends its symbolic
language** when existing tools fail to describe the data. The validation target
is **wave–particle duality** — i.e., re-discovering quantum-style structure from
raw experiment data without ever being told quantum mechanics exists.

> **Pure algorithmic — no LLM, no pre-trained models, no natural language.**
> This is by design: since the validation goal is to rediscover already-known
> physics, any knowledge leakage from a language model would invalidate the
> result. Every design choice must pass the test:
> *“Would this make sense if we didn't know the answer?”*

---

## What problem this solves

Existing AI physics-discovery systems (PySR, AI Feynman, AI-Newton, SciNet,
HNN/LNN, …) all search for formulas **inside a fixed mathematical framework**.
They assume physical quantities are real numbers, relations are deterministic
functions, and state is determined directly by inputs. When data comes from a
domain that needs a *different* mathematical structure (probabilistic, discrete,
high-dimensional state space), these systems silently fail and have no way to
diagnose *why* or to extend their own description language.

| System | What it discovers | Framework fixed? | Self-extending? |
|---|---|---|---|
| PySR / AI Feynman | formulas `y = f(x)` | yes (ℝ + basic ops) | no |
| AI-Newton | concepts + laws | yes (differential polynomials) | no |
| SciNet | representations (bottleneck dim) | yes (NN) | no |
| HNN / LNN | parameters of H/L | yes (Hamiltonian/Lagrangian) | no |
| DreamCoder | library functions | partial (adds functions, not types) | partial |
| **ATLAS** | **geometric structures + laws + concepts** | **no — DSL grows** | **yes (RGDE)** |

**Core claim:** ATLAS is the first system that can detect when its current
mathematical framework is insufficient, **discover new mathematical description
spaces from data**, and **encode those spaces as new symbolic types** that
extend its own language.

---

## How it works — 5-step main loop

```
   Solve   →   Extract   →   Diagnose   →   Extend   →   Unify   ─┐
     ↑                                                              │
     └──────────────────────────────────────────────────────────────┘
```

1. **Solve** — try to fit the data with the current DSL via symbolic regression (PySR).
2. **Extract** — pull formulas + Pareto front + constants out of the SR results.
3. **Diagnose** — D1–D4 checks decide whether the current framework is sufficient.
4. **Extend** — if not, run the **RGDE** pipeline (encoder SR + constraint discovery
   + type builder + evaluator) to discover a new state-space geometry and add it
   to the DSL as a new type.
5. **Unify** — fold the result into the global theory: U1 (constants), U2 (templates),
   U3 (types), checked against a **global MDL criterion**.

The same loop runs in **multi-agent** mode (Mode A / Mode B), with multiple
agents proposing on overlapping environment subsets and a **proposal pool +
verifier** keeping them honest.

## Repository layout

```
atlas/
├── types.py                ← KnobSpec, DetectorSpec, EnvSchema, FitMetrics, FormulaRecord
├── dsl/                    ← symbolic expression system
│   ├── operators.py        ← DSL_0 (10 base operators)
│   ├── expr.py             ← AST nodes (Const, Var, BinOp, …)
│   ├── canonicalize.py     ← α-equivalence, commutativity, identity elimination, const folding
│   └── serialize.py        ← S-expression / dict serialization
├── environments/           ← 12 anonymized experiment simulators
│   ├── env_01_photoelectric.py … env_07_stern_gerlach.py    ← 7 quantum
│   ├── env_08_water_wave.py … env_10_spring.py              ← 3 classical control
│   └── env_11_freefall.py, env_12_heat_conduction.py        ← 2 distractors
├── data/dataset.py         ← knob sweeps, train/test split
├── sr/                     ← PySR wrapper, FormulaStore (Pareto front)
├── analysis/               ← concepts (DreamCoder-style), diagnostics D1–D4, PSLQ unifier
├── scinet/                 ← SciNet information-bottleneck (PyTorch)
├── rgde/                   ← RGDE pipeline: encoder SR → constraint discovery → type builder → evaluator
├── agent/                  ← single-agent 5-step loop (ATLASAgent)
├── multi_agent/            ← Mode A / Mode B orchestrator + proposal pool + verifier
└── unifier/                ← U1 constants / U2 templates / U3 types
docs/                       ← multi-agent design spec, anti-cheating audit
validation/                 ← Phase 0/1 validation drivers
experiments/                ← per-phase results (git-ignored)
tests/                      ← 384+ pytest tests, ~5 min fast suite
ARCHITECTURE.md             ← full architecture guide
ATLAS_proposal.md           ← original proposal / design doc
anti_cheating_audit.md      ← knowledge-leakage audit checklist
literature_survey.md
```

Dependency flow:
`environments ← data ← sr ← analysis ← agent ← multi_agent`,
with `rgde ← scinet`, and `unifier` feeding into `multi_agent`.

## Tech stack

- **Python 3.11+**
- **PySR** (Julia backend) — symbolic regression
- **PyTorch** — SciNet information bottleneck
- **mpmath** PSLQ + custom log-space search — constant identification
- **pytest** — 384+ tests, ~5 min fast suite

## Running

```bash
# Fast test suite
python -m pytest tests/ -v

# Slow integration tests (~17 min)
python -m pytest tests/agent/ tests/multi_agent/ tests/integration/ -v

# Phase 1 — single-agent classical validation
python validation/phase1_classical.py --env ENV_11   # Free fall
python validation/phase1_classical.py --env ENV_10   # Spring
python validation/phase1_classical.py                # All 5 sequential

# Phase 0 — SciNet-SR bridge re-validation
python validation/phase0_scinet_sr_bridge.py --seeds 5 --epochs 500

# Anti-cheating audit
python -m pytest tests/environments/test_anti_cheating.py -v
```

### Gate criteria

| Phase | Level | Criteria |
|---|---|---|
| Phase 1 | **Minimum** | ENV-11 + ENV-12 R² > 0.95, no false D1/D2 |
| Phase 1 | **Full**    | + ENV-10/09 R² > 0.95, ENV-08 R² > 0.90 |
| Phase 0 | **Pass**    | K ≥ 2 selected in > 50% of seeds |

## Key design documents

| Document | What's in it |
|---|---|
| [`ATLAS_proposal.md`](./ATLAS_proposal.md) | Core system design: 5-step main loop, RGDE pipeline, 12 environments, anti-cheating protocol, expected discovery pathway, implementation phases |
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Module dependency graph, how to run, key design decisions, Phase 0–4 next steps |
| [`docs/superpowers/specs/2026-03-31-multi-agent-atlas-design.md`](./docs/superpowers/specs/2026-03-31-multi-agent-atlas-design.md) | Multi-agent consensus framework: Mode A/B, proposal pool, global MDL criterion, Unifier (U1/U2/U3), V1–V10 validation protocol |
| [`anti_cheating_audit.md`](./anti_cheating_audit.md) | Audit checklist enforcing zero knowledge leakage |

## Status

Active research project. Phases 0 and 1 are running; Phases 2–3 (quantum
baselines and multi-agent runs) are upcoming. Experiment artifacts under
`experiments/` are intentionally git-ignored.
