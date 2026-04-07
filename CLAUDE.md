# ATLAS Project — Claude Code Instructions

## Project Overview

ATLAS (Autonomous Theory Learning through Adaptive Search) is a multi-agent physics discovery system that starts with minimal mathematical knowledge and autonomously extends its symbolic language to discover physical laws (targeting wave-particle duality). It is **pure algorithmic** — no LLM, no pre-trained models, no natural language. This is critical: any knowledge leakage invalidates results.

## Core Design Principle

> Every design choice must pass: **"Would this make sense if we didn't know the answer?"**

Zero tolerance for physics knowledge leakage. No shortcuts that encode quantum mechanics, no operators that presuppose the answer, no semantic naming that hints at physics.

## Key Reference Documents

| Document | Path | Content |
|----------|------|---------|
| **ATLAS Proposal** | `ATLAS_proposal.md` | Core system design: 5-step main loop (Solve/Extract/Diagnose/Extend/Unify), RGDE pipeline, 12 experiment environments, anti-cheating protocol, expected discovery pathway, implementation phases |
| **Architecture Guide** | `ARCHITECTURE.md` | Code structure, module dependency graph, how to run tests and the system, key design decisions, next steps (Phase 0-4) |
| **Multi-Agent Spec** | `docs/superpowers/specs/2026-03-31-multi-agent-atlas-design.md` | Multi-agent consensus framework: Mode A/B, proposal pool, experiment-centric verification (global MDL criterion), Unifier module (U1: constants, U2: templates, U3: types), Theory output format, validation protocol (V1-V10) |

## Tech Stack

- **Language**: Python 3.11+
- **Symbolic Regression**: PySR (Julia backend)
- **Neural Networks**: PyTorch (SciNet information bottleneck)
- **Constants**: mpmath PSLQ + custom log-space search
- **Testing**: pytest (384+ tests, ~5 min fast suite)

## Running Tests

```bash
# Full fast suite (skips PySR/PyTorch if not installed)
python -m pytest tests/ -v

# Slow integration tests (agent + orchestrator, ~17 min)
python -m pytest tests/agent/ tests/multi_agent/ tests/integration/ -v

# Anti-cheating audit
python -m pytest tests/environments/test_anti_cheating.py -v
```

## Module Structure

```
atlas/
  dsl/           # Symbolic expression AST, operators, canonicalization
  environments/  # 12 anonymized experiment simulators (7 quantum + 3 classical + 2 distractor)
  data/          # ExperimentDataset: knob sweeps, train/test split
  sr/            # PySR wrapper, FormulaStore with Pareto front
  analysis/      # Concepts (DreamCoder-style), diagnostics (D1-D4), PSLQ unifier
  scinet/        # SciNet information bottleneck (PyTorch)
  rgde/          # RGDE pipeline: encoder SR, constraint discovery, type builder, evaluator
  agent/         # Single-agent 5-step loop (ATLASAgent)
  multi_agent/   # Multi-agent orchestrator (Mode A/B), proposal pool, verifier
  unifier/       # Theory synthesis: U1 constants, U2 templates, U3 types
```
                                                     
Dependency flow: `environments <- data <- sr <- analysis <- agent <- multi_agent`, with `rgde <- scinet` and `unifier` feeding into `multi_agent`.

## Experiment Data & Results

All experiment outputs go under `experiments/` (git-ignored). Never commit raw experiment data or PySR artifacts.

```
experiments/
  phase0/                          # SciNet-SR bridge validation (ENV-07)
    phase0_results.json            # Aggregate gate results
    seed_{N}/                      # Per-seed SciNet models + bottleneck vectors
  phase1/                          # Single-agent classical validation
    ENV_{ID}/                      # Per-environment results
      result.json                  # R², formula, diagnostics, timing
      agent_output.json            # Full agent.run() output
    phase1_results.json            # Aggregate gate results
  phase2/                          # Quantum baselines (future)
  phase3/                          # Multi-agent runs (future)
```

### Experiment Reports

Each phase directory contains a `REPORT.md` with:
- Summary table (env, ground truth, R², gate status, formula quality)
- Per-environment formula analysis (discovered vs true, structure match, constants)
- Cross-cutting findings and recommended next steps
- Reports are updated after each experiment run

### Running Experiments

```bash
# Phase 1 — classical experiments (easy to hard)
python validation/phase1_classical.py --env ENV_11    # Free fall (~1-2 min)
python validation/phase1_classical.py --env ENV_12    # Heat conduction (~1-2 min)
python validation/phase1_classical.py --env ENV_10    # Spring (~3-5 min)
python validation/phase1_classical.py --env ENV_09    # Elastic collision (~3-5 min)
python validation/phase1_classical.py --env ENV_08    # Water wave (~10-15 min)
python validation/phase1_classical.py                 # All 5 sequential

# Phase 0 — SciNet-SR bridge re-validation
python validation/phase0_scinet_sr_bridge.py --seeds 5 --epochs 500
```

### Gate Criteria

| Phase | Level | Criteria |
|-------|-------|---------|
| Phase 1 | **Minimum** | ENV-11 + ENV-12 R²>0.95, no false D1/D2 |
| Phase 1 | **Full** | + ENV-10/09 R²>0.95, ENV-08 R²>0.90 |
| Phase 0 | **Pass** | K>=2 selected in >50% of seeds |
