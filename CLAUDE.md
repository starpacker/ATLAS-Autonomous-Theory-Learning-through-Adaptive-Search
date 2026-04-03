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
