# ATLAS Multi-Agent Architecture Design Spec

## A Consensus-Driven Multi-Agent Framework for Autonomous Physics Discovery

**Version 1.0 — 2026-03-31**

---

## 1. Overview

### 1.1 Motivation

ATLAS (Autonomous Theory Learning through Adaptive Search) currently operates as a single monolithic loop. This spec extends it to a **multi-agent architecture** where multiple independent ATLAS pipeline instances ("scientist agents") work in parallel, each assigned a subset of experiments, with a global consensus mechanism for sharing discoveries and a Unifier module for theory synthesis.

### 1.2 Core Design Principles

1. **Pure algorithmic agents** — no LLM, no natural language, zero knowledge leakage
2. **Experiment-centric evidence** — the fundamental unit of evidence is "extension X improves fit on experiment Y", not "agent Z supports extension X"
3. **Natural selection through consensus** — DSL extensions survive only if they improve global description length across multiple experiments
4. **Laws are compressible, initial conditions are not** — the theory captures shared structure (laws, constants, types); per-experiment parameters are explicitly separated as incompressible bindings
5. **Two independent experimental modes** — Mode A (fully independent) vs Mode B (consensus sharing), compared as controlled experiments

### 1.3 Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    ATLAS Multi-Agent System                    │
│                                                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐     │
│  │ Agent-1 │ │ Agent-2 │ │ Agent-3 │  ...  │ Agent-N │     │
│  │ ENV:    │ │ ENV:    │ │ ENV:    │       │ ENV:    │     │
│  │ 01,04,  │ │ 02,05,  │ │ 03,04,  │       │ 04,07,  │     │
│  │ 08,11   │ │ 07,10   │ │ 06,12   │       │ 10,11   │     │
│  └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘     │
│       │           │           │                  │          │
│       └───────────┴─────┬─────┴──────────────────┘          │
│                         ▼                                    │
│              ┌─────────────────────┐                         │
│              │    Proposal Pool    │                         │
│              │  (DSL extensions)   │                         │
│              └──────────┬──────────┘                         │
│                         ▼                                    │
│              ┌─────────────────────┐                         │
│              │  Experiment-Centric │                         │
│              │    Verification     │                         │
│              └──────────┬──────────┘                         │
│                         ▼                                    │
│              ┌─────────────────────┐                         │
│              │      Unifier        │                         │
│              │  U1: Constants      │                         │
│              │  U2: Templates      │                         │
│              │  U3: Types          │                         │
│              └──────────┬──────────┘                         │
│                         ▼                                    │
│              ┌─────────────────────┐                         │
│              │   Theory Output     │                         │
│              └─────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Structure

### 2.1 Single Agent Definition

Each agent is a stateless ATLAS pipeline instance. It contains no language model, no pre-trained knowledge, and no natural language processing capability.

```
Agent-i = {
  id:           unique identifier
  seed:         random seed (for reproducibility)
  assigned_envs: subset of {ENV-01, ..., ENV-12}, |assigned_envs| >= 3
  dsl:          current DSL (initialized to DSL_0)
  
  state: {
    formulas:     {env_id -> Pareto front of formulas},
    concepts:     [extracted library functions],
    extensions:   [RGDE-discovered new types],
    constants:    {env_id -> {numerical constants in formulas}},
    diagnostics:  {env_id -> [D1-D5 diagnostic results]},
    fit_metrics:  {env_id -> {formula_id -> {R2, residual_var, MDL}}},
  }
}
```

**DSL_0** (identical for all agents):
```
DSL_0 = {R, +, -, *, /, sin, cos, exp, log, ^}
```

No complex arithmetic, no physics-specific operators, no probability primitives. These must all be discovered through RGDE.

### 2.2 Agent Internal Loop

Each agent runs the standard ATLAS five-step cycle on its assigned experiments:

```
repeat until global convergence:
  Step 1 - Solve:    SR (PySR) on assigned_envs with current dsl
  Step 2 - Extract:  frequent subexpression mining -> concept candidates
  Step 3 - Diagnose: failure analysis on failed experiments (D1-D5)
  Step 4 - Extend:   RGDE pipeline -> DSL extension proposals
  Step 5 - LocalUnify: PSLQ on constants within own experiments
  
  -> Submit proposals to global pool
  -> Receive globally adopted extensions, merge into local dsl
```

### 2.3 Experiment Assignment

**Rules:**
1. Each agent is assigned >= 3 experiments
2. Each experiment is covered by >= 2 agents (ensures cross-validation)
3. Quantum / classical / distractor experiments are mixed within each agent (prevents implicit bias)
4. Assignment is generated randomly with a recorded seed (reproducible + auditable)

**Example assignment (N=6 agents, 12 experiments):**

```
Agent-1: ENV-01, ENV-04, ENV-08, ENV-11
Agent-2: ENV-02, ENV-05, ENV-07, ENV-10
Agent-3: ENV-03, ENV-04, ENV-06, ENV-12
Agent-4: ENV-01, ENV-05, ENV-09, ENV-07
Agent-5: ENV-02, ENV-03, ENV-06, ENV-08
Agent-6: ENV-04, ENV-07, ENV-10, ENV-11

Coverage matrix:
ENV:     01 02 03 04 05 06 07 08 09 10 11 12
Agents:   2  2  2  3  2  2  3  2  1  2  2  1
```

**Anti-cheating:** Random assignment prevents "deliberately grouping related experiments to guide discovery." Quantum/classical mixing prevents agents from seeing only quantum data.

---

## 3. Proposal Pool and Consensus Mechanism

### 3.1 Proposal Structure

When an agent's RGDE (Step 4) produces a DSL extension, it submits a proposal:

```
Proposal = {
  id:             "PROP-{agent_id}-{epoch}-{seq}",
  source_agent:   agent_id,
  source_env:     experiment ID that triggered this extension,
  trigger:        diagnostic signal
                  (e.g., "ENV-07: D1=stochastic, D2=discrete(N=2), D3=K>N"),
  
  extension: {
    type:         "new_type" | "new_operator" | "prob_mode",
    definition:   formal definition (see below),
  },
  
  evidence: {
    fit_before:   {env_id -> {R2, MDL}},
    fit_after:    {env_id -> {R2, MDL}},
    delta_mdl:    local MDL change on source experiments,
    n_seeds:      number of SR seeds used for evaluation,
  }
}
```

**Extension types:**

```
new_operator:
  {name: "concept_cos2", definition: "cos^2(.)"}
  -> adds a function to the SR search space

prob_mode:
  {description: "enable P(y|x) search mode for stochastic experiments"}
  -> enables probabilistic SR on experiments flagged D1=stochastic

new_type:
  {name: "State_07", dimension: 3,
   encoding: [z1 = f1(knobs), z2 = f2(knobs), z3 = f3(knobs)],
   constraints: ["z1^2 + z2^2 + z3^2 <= 1"],
   decoder_law: "P(outcome) = affine(state . measurement_vector)"}
  -> introduces a new state space type with geometric constraints
```

### 3.2 Experiment-Centric Verification

The fundamental unit of evidence is **experiment-level fit change**, not agent opinion.

```
For each proposal P in the pool:
  For each experiment ENV-j (j = 1..12):
    Executed by whichever agent(s) cover ENV-j:
    
    1. Run SR M times WITHOUT extension  -> {R2_base_1, ..., R2_base_M}
    2. Run SR M times WITH extension      -> {R2_ext_1,  ..., R2_ext_M}
    3. Compute per-run: delta_R2_m = R2_ext_m - R2_base_m
    4. Compute per-run: delta_MDL_m = MDL_ext_m - MDL_base_m
    
    Statistical test (one-sided paired t-test):
      H0: mean(delta_R2) <= 0  (extension does not help)
      H1: mean(delta_R2) > 0   (extension helps)
      -> p-value for this experiment
    
    Per-experiment result:
      mu_j     = mean(delta_MDL across M seeds)
      sigma_j  = std(delta_MDL across M seeds)
```

### 3.3 Adoption Criterion: Global MDL

Adding a new operator or type expands the search space for ALL agents, which can cause stochastic degradation on experiments that don't need the extension. Therefore, the adoption criterion is **global MDL net change**, not per-experiment zero-harm.

```
Adoption criterion:

  delta_Total_MDL = Sum_{j=1}^{12} mu_j    (mu_j = mean MDL change on ENV-j)
  
  where mu_j accounts for:
    - MDL improvement on experiments that benefit from the extension
    - MDL degradation on experiments hurt by search space expansion
    - MDL cost of the extension definition itself (amortized across experiments)
  
  Adopt if:
    delta_Total_MDL < 0                  (global description length decreases)
    AND delta_Total_MDL is statistically significant:
      |delta_Total_MDL| > 2 * sqrt(Sum_j sigma_j^2)   (exceeds pooled noise)

  Adopted -> merge into global DSL, broadcast to all agents
  Not adopted -> remain in pool (may gain new evidence in future epochs)
```

**Why global MDL instead of per-experiment voting:**
A new operator like `concept_cos2` may cause a stochastic R2 dip of -0.002 on ENV-09 (which doesn't need cos2) due to search space expansion, while providing R2 improvement of +0.15 on ENV-02, ENV-04, ENV-08. The global MDL criterion correctly adopts this extension, whereas a "zero harm" rule would incorrectly reject it.

### 3.4 Mode A vs Mode B

```
Mode A (fully independent):
  - Skip the verification + broadcast phase
  - Each agent's RGDE extensions only apply locally
  - Unifier still runs, but only for post-hoc analysis
  - Measures: do agents independently converge to equivalent theories?

Mode B (consensus sharing):
  - Full pipeline as described above
  - Adopted extensions broadcast to all agents
  - Measures: does sharing accelerate convergence? Better compression?

Comparison metrics:
  1. Convergence speed (epochs to global R2 > 0.95)
  2. Theory consistency (structural similarity of formulas across agents)
  3. Constant unification degree (can Unifier extract same base constants?)
  4. Final compression ratio
  5. Overfitting rate (agent-specific extensions that fail cross-validation)
```

---

## 4. Unifier Module

The Unifier is a pure algorithmic module (no LLM) that synthesizes individual agent outputs into a unified theory.

### 4.1 Step U1: Constant Unification (PSLQ)

```
Input: all numerical constants from all formulas across all agents
Output: minimal set of universal constants {UC_k}

Procedure:
  1. Collect all constants {C_i} from all accepted formulas
  2. De-duplicate: if multiple agents produced formulas for the same experiment,
     keep the formula with best R2 on held-out test set
  3. Sign separation:
       sign_i = sgn(C_i)
       abs_i  = |C_i|
     Signs are recorded separately and do NOT enter the PSLQ search.
  4. PSLQ search in log-space:
       Search for integer vectors (n_1, n_2, ...) such that:
         n_1 * log(abs_1) + n_2 * log(abs_2) + ... ≈ 0
       i.e., |C_1|^n_1 * |C_2|^n_2 * ... ≈ 1
  5. Extract minimal basis {UC_k} such that all |C_i| = product of UC_k^{n_k}
  6. Restore signs: C_i = sign_i * product(UC_k^{n_k})
  7. Error propagation:
       For each UC_k, compute independent estimates from each experiment:
         UC_k_estimates = [C_i / f(other_params) for each appearance]
         UC_k_value = weighted_mean(UC_k_estimates)
         UC_k_err   = weighted_std(UC_k_estimates)
       Consistency check: chi-squared test across estimates
         If chi2 p-value < 0.01 -> flag as potentially spurious unification
  8. Rewrite all formulas using {UC_k} -> free constants decrease
```

### 4.2 Step U2: Template Extraction (Anti-Unification)

**Prerequisite: AST Canonicalization**

Before comparing any two formulas, their abstract syntax trees must be canonicalized:

```
Canonicalization steps:
  1. Alpha-equivalence: rename all variables by order of first appearance
       f(knob_2, knob_0) -> f(x_0, x_1)  (knob_2 appears first -> x_0)
  2. Commutativity normalization: 
       for commutative ops (+, *), sort children by canonical ordering
       b + a -> a + b
       knob_1 * knob_0 -> knob_0 * knob_1
  3. Associativity flattening:
       (a + b) + c -> +(a, b, c)  (n-ary representation)
  4. Identity elimination:
       x * 1 -> x,  x + 0 -> x,  x^1 -> x
  5. Constant folding:
       2 * 3 -> 6,  sin(0) -> 0
```

**Anti-unification procedure:**

```
Input: two canonicalized ASTs
Output: most specific common generalization + per-formula bindings

Example:
  Formula A (ENV-01): UC_1 * max(x_0 - C_local_1, 0) * x_1
  Formula B (ENV-02): UC_1 * (1 - cos(x_0))
  -> Template: UC_1 * g(x_0, ...)   (too general, rejected by MDL)

  Formula C (ENV-04 high): I_0 * cos^2(UC_1 * f(knobs) * x)
  Formula D (ENV-04 low):  C * cos^2(UC_1 * f(knobs) * x)
  -> Template: PARAM * cos^2(UC_1 * f(knobs) * x)
  -> Binding C: PARAM = I_0  (intensity)
  -> Binding D: PARAM = C    (probability normalization)
  -> This IS a meaningful template: intensity profile = probability profile

MDL evaluation for each candidate template:
  savings = n_unified_formulas * template_size
  cost    = template_definition_length + sum(binding_lengths)
  Accept if: savings > cost
```

### 4.3 Step U3: Type Unification

**Prerequisite: Constraint equation canonicalization with alpha-equivalence**

```
When comparing constraint equations from different experiments:
  State_from_ENV07: {z in R^3 : z_1^2 + z_2^2 + z_3^2 <= 1}
  State_from_ENV03: {w in R^3 : w_1^2 + w_2^2 + w_3^2 <= 1}
  
  Step 1: Alpha-rename (z_i -> v_i, w_i -> v_i)
  Step 2: Canonicalize constraint AST
  Step 3: Compare: structurally identical -> isomorphic

For non-trivial cases (coordinate transformations):
  State_A: {z in R^3 : z_1^2 + z_2^2 + z_3^2 <= 1}
  State_B: {w in R^3 : (w_1+w_2)^2/2 + (w_1-w_2)^2/2 + w_3^2 <= 1}
  
  Step 1: Expand and simplify constraint B -> w_1^2 + w_2^2 + w_3^2 <= 1
  Step 2: Alpha-rename
  Step 3: Compare: identical -> isomorphic (related by rotation)

Tolerance for coefficient comparison:
  |a_i - b_i| / max(|a_i|, |b_i|) < tol  (tol is a hyperparameter)
  Dimension must match exactly (integer, no tolerance needed)
```

**Merge procedure:**

```
If two types are isomorphic:
  1. Merge into a single unified type with the simpler encoding
  2. Re-evaluate: does the unified type still fit both experiments?
  3. Update global compression metric
```

---

## 5. Theory Output Format

### 5.1 Structure

The final theory is a **layered compression** of all experimental data. It explicitly separates compressible structure (laws) from incompressible parameters (initial conditions).

```
Theory = {

  // ===== Compressible: Laws =====
  
  dsl_final: {
    base:       DSL_0,
    extensions: [ordered list of adopted extensions with provenance]
  },
  
  law_templates: [
    {
      id:                "LAW-1",
      template:          "E = UC_1 * x_param - W_param",
      shared_constants:  ["UC_1"],
      applies_to:        ["ENV-01", "ENV-05", ...],
      compression_savings: 156,
    },
    {
      id:                "LAW-2",
      template:          "P(y) ~ cos^2(UC_1 * f(x))",
      shared_constants:  ["UC_1"],
      applies_to:        ["ENV-04", "ENV-07", ...],
      compression_savings: 203,
    },
  ],
  
  shared_constants: [
    {
      symbol:            "UC_1",
      value:             6.626e-34,
      uncertainty:       0.003e-34,
      appears_in:        ["LAW-1", "LAW-2"],
      n_experiments:     5,
      chi2_consistency:  0.87,   // p-value of cross-experiment consistency
    }
  ],
  
  shared_types: [
    {
      name:        "State_Q",
      dimension:   3,
      constraints: ["v_1^2 + v_2^2 + v_3^2 <= 1"],
      appears_in:  ["ENV-07", ...],
      compression_savings: 83,
    }
  ],

  // ===== Incompressible: Initial Conditions =====
  
  experiment_bindings: {
    "ENV-01": {
      law: "LAW-1",
      params: {x_param: "knob_0", W_param: 0.441},
      fit: {R2: 0.997, residual_var: 2.3e-4, MDL: 47}
    },
    "ENV-04": {
      law: "LAW-2",
      params: {f: "pi * knob_1 * pos / knob_2"},
      fit: {R2: 0.993, residual_var: 1.1e-3, MDL: 62}
    },
    ...
  },
  
  // ===== Meta: Compression Accounting =====
  
  compression_chain: [
    {level: 0, total_MDL: 1247, label: "independent formulas"},
    {level: 1, total_MDL: 891,  label: "constant unification",  delta: -356},
    {level: 2, total_MDL: 724,  label: "template extraction",   delta: -167},
    {level: 3, total_MDL: 658,  label: "type unification",      delta: -66},
  ],
  compression_ratio: 1247 / 658,   // = 1.89
  
  // ===== Provenance: Discovery Lineage =====
  
  extension_lineage: [
    {
      extension:    "concept_cos2",
      epoch:        0,
      trigger:      "Extract: cos^2(.) appeared in 3 formulas",
      source_envs:  ["ENV-02", "ENV-04", "ENV-08"],
      adoption:     {delta_total_MDL: -89, n_beneficial_envs: 5},
    },
    {
      extension:    "prob_mode",
      epoch:        0,
      trigger:      "Diagnose: ENV-04(low intensity) D1=stochastic",
      source_envs:  ["ENV-04"],
      adoption:     {delta_total_MDL: -124, n_beneficial_envs: 3},
    },
    {
      extension:    "State_Q (dim=3, sphere)",
      epoch:        1,
      trigger:      "Diagnose: ENV-07 D3=K(3)>N-1(1)",
      source_envs:  ["ENV-07"],
      adoption:     {delta_total_MDL: -83, n_beneficial_envs: 2},
    },
  ]
}
```

### 5.2 Interpretation

The theory does NOT contain natural language claims like "light has wave-particle duality." Instead, it contains **structural facts** that a human physicist can interpret:

```
Structural facts the system discovers (without knowing what they mean):

  F1: "Experiments ENV-04(low), ENV-07 require prob_mode (D1=stochastic)"
      -> some experiments are fundamentally probabilistic

  F2: "LAW-2 template cos^2(UC_1 * f(x)) appears in both
       deterministic intensity formulas AND probability distributions"
      -> the shape of the wave pattern = the shape of the probability distribution

  F3: "UC_1 appears in 5+ experiments across LAW-1 and LAW-2"
      -> a single constant governs both energy relations and wave/probability patterns

  F4: "ENV-07 requires State_Q with dim=3, sphere constraint,
       but the classical prediction would be dim=1"
      -> some systems have more internal degrees of freedom than classically expected

  F1 + F2 + F3 + F4 = the operational core of wave-particle duality
  (but the system never uses these words)
```

---

## 6. Full Execution Flow

### 6.1 Main Loop

```
INITIALIZATION:
  1. Generate random experiment assignment (record seed)
  2. Initialize all agents: dsl = DSL_0, empty state
  3. Global proposal pool = empty
  4. Global adopted DSL = DSL_0

MAIN LOOP (epoch = 0, 1, 2, ...):

  ┌─ PARALLEL PHASE ────────────────────────────────────┐
  │  Each Agent-i runs independently:                    │
  │    Step 1: Solve (SR on assigned_envs with dsl_i)   │
  │    Step 2: Extract (subexpression -> concepts)       │
  │    Step 3: Diagnose (failed experiments -> D1-D5)    │
  │    Step 4: Extend (RGDE -> proposals)                │
  │    Step 5: LocalUnify (PSLQ on own constants)        │
  │                                                      │
  │  Output: proposals + formulas + fit_metrics          │
  └──────────────────────────────────────────────────────┘
                        |
                        v
  ┌─ VERIFICATION PHASE ────────────────────────────────┐
  │  For each new proposal P:                            │
  │    For each experiment ENV-j (j = 1..12):            │
  │      Run SR M times with/without extension           │
  │      Compute mu_j (mean delta_MDL), sigma_j (std)    │
  │                                                      │
  │    delta_Total_MDL = Sum_j(mu_j)                     │
  │    pooled_noise = sqrt(Sum_j(sigma_j^2))             │
  │                                                      │
  │    Adopt if:                                         │
  │      delta_Total_MDL < 0                             │
  │      AND |delta_Total_MDL| > 2 * pooled_noise        │
  │                                                      │
  │    Adopted -> merge into global DSL, broadcast       │
  │    Not adopted -> remain in pool for future epochs   │
  └──────────────────────────────────────────────────────┘
                        |
                        v
  ┌─ UNIFICATION PHASE ─────────────────────────────────┐
  │  Unifier runs:                                       │
  │    U1: PSLQ constant unification (with sign sep.)   │
  │    U2: AST canonicalization + anti-unification       │
  │    U3: Type isomorphism detection + merge            │
  │                                                      │
  │  Output: Theory snapshot for this epoch              │
  │  Record: compression_chain latest layer              │
  └──────────────────────────────────────────────────────┘
                        |
                        v
  CONVERGENCE CHECK:
    Condition 1: all 12 experiments R2 > 0.95 (held-out test set)
                 AND compression_ratio stable for 3 epochs
    Condition 2: 5 consecutive epochs with no proposals adopted
                 (pipeline saturated)
    Condition 3: compute budget exhausted (max_epochs or GPU-hours)
    
    Any condition met -> STOP
```

### 6.2 Mode A vs Mode B

```
Mode A (fully independent):
  - VERIFICATION PHASE: skip broadcast step
  - Each agent's RGDE extensions only apply locally
  - UNIFICATION PHASE: still runs, but for post-hoc comparison only
  - Each agent's dsl evolves independently

Mode B (consensus sharing):
  - Full pipeline as described above
  - Adopted extensions broadcast to all agents and merged into their DSLs
```

---

## 7. Validation Protocol

### 7.1 Inherited from ATLAS Proposal

| ID | Test | Method | Expected |
|----|------|--------|----------|
| V1 | Alternative physics (h->2h) | Modify h in all environments | Same formula structure, different constant values |
| V2 | Classical limit (h->0) | Set h to near-zero | No probabilistic extension triggered |
| V3 | Random data | Replace experiment data with noise | No meaningful formulas or types discovered |
| V4 | Classical cross-validation | Run on classical mechanics data | Discover F=ma, no RGDE triggered |
| V5 | Non-quantum sphere | Classical gyroscope (sphere state space) | Discover sphere geometry, don't claim quantum |
| V6 | Multi-seed consistency | 20+ independent seeds | >60% seeds converge to equivalent theory |

### 7.2 Multi-Agent Specific

| ID | Test | Method | Expected |
|----|------|--------|----------|
| V7 | Assignment robustness | 20+ different random assignments | Final theory independent of which agent got which experiments |
| V8 | Mode A/B comparison | Same assignment, run both modes | Quantify: does sharing help convergence and/or compression? |
| V9 | Overlap consistency | For experiments covered by 2+ agents | Independent formulas structurally equivalent, constants within error bars |
| V10 | Extension causal audit | For each adopted extension, trace lineage | Every extension has data-driven trigger, no unexplained DSL growth |

---

## 8. Anti-Cheating Checklist

Every component must pass: **"Would this design make sense if we didn't know the answer?"**

| Component | Status | Justification |
|-----------|--------|---------------|
| Agent = pure algorithm (no LLM) | CLEAN | No knowledge source beyond DSL_0 and data |
| DSL_0 = basic arithmetic + trig | CLEAN | Same as AI Feynman, standard math toolkit |
| Experiment assignment is random | CLEAN | No deliberate grouping to guide discovery |
| Environment interface: knob/detector only | CLEAN | No physical semantics, normalized, anonymized |
| Extension adoption: global MDL criterion | CLEAN | Pure information-theoretic, no semantic judgment |
| PSLQ constant unification | CLEAN | Number theory algorithm, zero physics prior |
| AST anti-unification | CLEAN | Pure syntactic operation |
| Type isomorphism detection | CLEAN | Pure mathematical structure comparison |
| Milestones defined by compression ratio | CLEAN | Not defined by known answers (E=hf, etc.) |
| Alternative physics test (V1) | CLEAN | Ultimate validation: system responds to data, not design |

**Remaining gray areas (inherited, acceptable):**

| Item | Risk | Mitigation |
|------|------|------------|
| Experiment selection (which 12?) | Medium | Mixed quantum/classical/distractor; V1 is ultimate test |
| Entity labels (experiments share entity_A) | Medium | Comparable to AI-Newton's object labels; consider removing |
| D1-D5 diagnostic categories | Low | All are generic statistical tests, not physics-specific |

---

## 9. Compute Estimate

```
Mode A single run:
  N agents * (SR cost per experiment * n_experiments * M seeds) * n_epochs
  = 6 * (1 GPU-hr * 4 * 10) * 15 epochs
  = ~3600 GPU-hours

Mode B single run (additional verification cost):
  Mode A cost + verification cost per proposal per epoch
  = ~3600 + 12 experiments * M seeds * n_proposals * n_epochs
  = ~5000 GPU-hours

Full validation suite:
  (Mode A + Mode B) * 20 assignment variants * 20 seeds
  = ~200,000 GPU-hours (rough upper bound)
  
  Can be reduced by:
    - Fewer assignment variants for initial runs
    - Early stopping on clearly converging/diverging runs
    - Sharing SR caches across agents for identical experiments
```

---

## 10. Relationship to ATLAS Proposal

This spec extends `ATLAS_proposal.md` as follows:

| ATLAS Proposal Section | This Spec |
|------------------------|-----------|
| Section 2 (Architecture) | Extended: single loop -> multi-agent with proposal pool |
| Section 3 (Five-Step Loop) | Preserved within each agent; added verification phase |
| Section 4 (Environment Layer) | Unchanged |
| Section 5 (Anti-Cheating) | Extended: V7-V10 added for multi-agent |
| Section 6 (Expected Pathway) | Unchanged (applies per-agent) |
| Section 7 (Limitations) | Unchanged |
| Section 8 (Implementation Plan) | Needs revision to incorporate multi-agent phases |

**Key additions:**
1. Experiment-centric verification with global MDL criterion
2. PSLQ sign separation for log-space search
3. AST canonicalization (alpha-equivalence + commutativity + associativity)
4. Theory as layered compression with explicit law/initial-condition separation
5. Mode A vs Mode B as controlled experimental comparison
6. Compression ratio as the single measure of theory quality

---

*ATLAS Multi-Agent Design Spec v1.0*
*2026-03-31*
