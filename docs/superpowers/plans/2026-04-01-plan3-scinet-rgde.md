# Plan 3: SciNet + RGDE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SciNet information bottleneck network and the RGDE (Representation-Grounded DSL Extension) pipeline that discovers state space geometry from data and extends the DSL with new types when symbolic regression fails.

**Architecture:** SciNet is a PyTorch encoder-decoder with an adjustable bottleneck dimension K. When the diagnostics module flags an experiment as needing more representation capacity (D3), RGDE trains SciNet, extracts symbolic structure from the learned representation via SR, discovers geometric constraints, and constructs a new DSL type. The pipeline is: SciNet train → SR on encoder → constraint discovery → type construction → SR on decoder → Pareto evaluation.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, existing atlas modules (PySR wrapper, DSL, diagnostics)

**Existing code (Plans 1-2):**
- `atlas/data/dataset.py` — ExperimentDataset
- `atlas/sr/pysr_wrapper.py` — run_sr, pysr_expr_to_atlas
- `atlas/sr/formula_store.py` — FormulaStore
- `atlas/analysis/diagnostics.py` — D1-D5 diagnostics
- `atlas/agent/dsl_state.py` — DSLState
- `atlas/agent/atlas_agent.py` — ATLASAgent main loop

---

## File Structure

```
atlas/
  scinet/
    __init__.py
    model.py              # SciNet encoder-decoder PyTorch model
    trainer.py            # Training loop with AIC-based K selection
    bottleneck.py         # Bottleneck analysis: extract z vectors, find optimal K
  rgde/
    __init__.py
    encoder_sr.py         # Step 4b: SR on encoder outputs
    constraint_finder.py  # Step 4c: discover algebraic constraints on z-space
    type_builder.py       # Step 4d: construct new DSL type from discovered structure
    decoder_sr.py         # Step 4e: SR on decoder in new type space
    pipeline.py           # Step 4a-4f: full RGDE orchestrator
    evaluator.py          # Step 4f: Pareto evaluation of proposed extension
tests/
  scinet/
    __init__.py
    test_model.py
    test_trainer.py
    test_bottleneck.py
  rgde/
    __init__.py
    test_encoder_sr.py
    test_constraint_finder.py
    test_type_builder.py
    test_pipeline.py
```

---

## Task 1: SciNet Model

**Files:**
- Create: `atlas/scinet/__init__.py`
- Create: `atlas/scinet/model.py`
- Create: `tests/scinet/__init__.py`
- Create: `tests/scinet/test_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scinet/__init__.py
```

```python
# tests/scinet/test_model.py
"""Tests for SciNet encoder-decoder model."""
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def test_model_creation():
    from atlas.scinet.model import SciNet
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    assert model.bottleneck_dim == 2


def test_forward_pass():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    x = torch.randn(10, 3)
    y_pred = model(x)
    assert y_pred.shape == (10, 1)


def test_encode():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    x = torch.randn(10, 3)
    z = model.encode(x)
    assert z.shape == (10, 2)


def test_decode():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    z = torch.randn(10, 2)
    y = model.decode(z)
    assert y.shape == (10, 1)


def test_encode_decode_roundtrip():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=3, output_dim=1,
                   encoder_hidden=[64, 32], decoder_hidden=[32, 64])
    x = torch.randn(10, 3)
    z = model.encode(x)
    assert z.shape == (10, 3)


def test_different_bottleneck_dims():
    from atlas.scinet.model import SciNet
    import torch
    for k in [1, 2, 3, 5]:
        model = SciNet(input_dim=4, bottleneck_dim=k, output_dim=1)
        x = torch.randn(5, 4)
        z = model.encode(x)
        assert z.shape == (5, k)


def test_output_dim_array():
    """Support array outputs (e.g., detector with 100 positions)."""
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=100)
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/scinet/test_model.py -v`

- [ ] **Step 3: Implement SciNet model**

```python
# atlas/scinet/__init__.py
"""SciNet: information bottleneck for representation learning."""
```

```python
# atlas/scinet/model.py
"""SciNet encoder-decoder model.

Architecture:
  encoder: input_dim -> hidden layers -> bottleneck_dim
  decoder: bottleneck_dim -> hidden layers -> output_dim

The bottleneck dimension K determines the learned representation's dimensionality.
Finding the right K is key to discovering the state space structure.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SciNet(nn.Module):
    """Information bottleneck encoder-decoder."""

    def __init__(self, input_dim: int, bottleneck_dim: int, output_dim: int,
                 encoder_hidden: list[int] | None = None,
                 decoder_hidden: list[int] | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim

        if encoder_hidden is None:
            encoder_hidden = [128, 64]
        if decoder_hidden is None:
            decoder_hidden = [64, 128]

        # Build encoder
        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in encoder_hidden:
            enc_layers.append(nn.Linear(prev_dim, h))
            enc_layers.append(nn.ReLU())
            prev_dim = h
        enc_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder
        dec_layers: list[nn.Module] = []
        prev_dim = bottleneck_dim
        for h in decoder_hidden:
            dec_layers.append(nn.Linear(prev_dim, h))
            dec_layers.append(nn.ReLU())
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to bottleneck representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck representation to output."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/scinet/test_model.py -v`

- [ ] **Step 5: Commit**

```bash
git add atlas/scinet/ tests/scinet/
git commit -m "feat: SciNet encoder-decoder model with configurable bottleneck"
```

---

## Task 2: SciNet Trainer + K Selection

**Files:**
- Create: `atlas/scinet/trainer.py`
- Create: `atlas/scinet/bottleneck.py`
- Create: `tests/scinet/test_trainer.py`
- Create: `tests/scinet/test_bottleneck.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scinet/test_trainer.py
"""Tests for SciNet trainer."""
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def test_train_config():
    from atlas.scinet.trainer import TrainConfig
    cfg = TrainConfig()
    assert cfg.epochs == 200
    assert cfg.lr == 1e-3
    assert cfg.batch_size == 64


def test_train_reduces_loss():
    from atlas.scinet.trainer import train_scinet, TrainConfig
    from atlas.scinet.model import SciNet
    import torch

    # Simple identity-like mapping: y = x0 + x1
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1).astype(np.float32)

    model = SciNet(input_dim=2, bottleneck_dim=2, output_dim=1)
    cfg = TrainConfig(epochs=50, lr=1e-3)
    result = train_scinet(model, X, y, cfg)

    assert result.final_loss < result.initial_loss
    assert result.final_loss < 0.1


def test_train_result_has_history():
    from atlas.scinet.trainer import train_scinet, TrainConfig
    from atlas.scinet.model import SciNet

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)

    model = SciNet(input_dim=2, bottleneck_dim=1, output_dim=1)
    result = train_scinet(model, X, y, TrainConfig(epochs=20))

    assert len(result.loss_history) == 20
```

```python
# tests/scinet/test_bottleneck.py
"""Tests for bottleneck dimension analysis."""
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def test_find_optimal_k_simple():
    """For y = x0, optimal K should be 1."""
    from atlas.scinet.bottleneck import find_optimal_k, KSelectionResult

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (300, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)

    result = find_optimal_k(X, y, k_range=[1, 2, 3], epochs_per_k=50)
    assert isinstance(result, KSelectionResult)
    assert result.best_k == 1


def test_find_optimal_k_higher_dim():
    """For y = f(x0, x1) with complex interaction, K should be > 1."""
    from atlas.scinet.bottleneck import find_optimal_k

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (300, 3)).astype(np.float32)
    # y depends on two independent combinations of inputs
    y1 = np.sin(X[:, 0] * 3) + X[:, 1]
    y2 = np.cos(X[:, 2] * 2)
    y = np.column_stack([y1, y2]).astype(np.float32)

    result = find_optimal_k(X, y, k_range=[1, 2, 3], epochs_per_k=80)
    assert result.best_k >= 2


def test_extract_bottleneck_vectors():
    """Extract z vectors from trained model."""
    from atlas.scinet.bottleneck import extract_bottleneck_vectors
    from atlas.scinet.model import SciNet
    from atlas.scinet.trainer import train_scinet, TrainConfig

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)

    model = SciNet(input_dim=2, bottleneck_dim=2, output_dim=1)
    train_scinet(model, X, y, TrainConfig(epochs=20))

    Z = extract_bottleneck_vectors(model, X)
    assert Z.shape == (100, 2)
    assert isinstance(Z, np.ndarray)
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement trainer.py**

```python
# atlas/scinet/trainer.py
"""SciNet training loop."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from atlas.scinet.model import SciNet


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-5
    encoder_sparsity: float = 0.0  # L1 penalty on encoder outputs


@dataclass
class TrainResult:
    final_loss: float
    initial_loss: float
    loss_history: list[float]
    model: SciNet


def train_scinet(model: SciNet, X: np.ndarray, y: np.ndarray,
                 config: TrainConfig | None = None) -> TrainResult:
    """Train SciNet on data.

    Args:
        model: SciNet model to train (modified in-place)
        X: (N, input_dim) input data
        y: (N, output_dim) target data
        config: training configuration
    """
    if config is None:
        config = TrainConfig()

    device = torch.device("cpu")
    model = model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    loss_history = []
    initial_loss = None

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Optional encoder sparsity regularization
            if config.encoder_sparsity > 0:
                z = model.encode(X_batch)
                loss = loss + config.encoder_sparsity * torch.mean(torch.abs(z))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if initial_loss is None:
            initial_loss = avg_loss

    return TrainResult(
        final_loss=loss_history[-1] if loss_history else float("inf"),
        initial_loss=initial_loss if initial_loss is not None else float("inf"),
        loss_history=loss_history,
        model=model,
    )
```

- [ ] **Step 4: Implement bottleneck.py**

```python
# atlas/scinet/bottleneck.py
"""Bottleneck dimension analysis: find optimal K and extract z vectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from atlas.scinet.model import SciNet
from atlas.scinet.trainer import train_scinet, TrainConfig


@dataclass
class KSelectionResult:
    best_k: int
    aic_scores: dict[int, float]
    losses: dict[int, float]
    models: dict[int, SciNet]


def find_optimal_k(X: np.ndarray, y: np.ndarray,
                   k_range: list[int] | None = None,
                   epochs_per_k: int = 100,
                   n_seeds: int = 3) -> KSelectionResult:
    """Find optimal bottleneck dimension K using AIC.

    Trains SciNet for each K in k_range, selects K with lowest AIC.
    AIC = N * log(MSE) + 2 * n_params, where n_params depends on K.

    Args:
        X: (N, input_dim) input data
        y: (N, output_dim) target data
        k_range: list of K values to try (default [1, 2, 3, 4, 5])
        epochs_per_k: training epochs per K value
        n_seeds: number of random seeds per K (take best)
    """
    if k_range is None:
        k_range = [1, 2, 3, 4, 5]

    input_dim = X.shape[1]
    output_dim = y.shape[1] if y.ndim > 1 else 1
    y_2d = y.reshape(-1, output_dim)
    N = X.shape[0]

    aic_scores: dict[int, float] = {}
    losses: dict[int, float] = {}
    models: dict[int, SciNet] = {}

    for k in k_range:
        best_loss_for_k = float("inf")
        best_model_for_k = None

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model = SciNet(input_dim=input_dim, bottleneck_dim=k,
                           output_dim=output_dim)
            config = TrainConfig(epochs=epochs_per_k)
            result = train_scinet(model, X, y_2d, config)

            if result.final_loss < best_loss_for_k:
                best_loss_for_k = result.final_loss
                best_model_for_k = result.model

        losses[k] = best_loss_for_k
        models[k] = best_model_for_k

        # AIC: N * log(MSE) + 2 * n_params
        n_params = sum(p.numel() for p in best_model_for_k.parameters())
        aic = N * np.log(max(best_loss_for_k, 1e-30)) + 2 * n_params
        aic_scores[k] = aic

    best_k = min(aic_scores, key=aic_scores.get)

    return KSelectionResult(
        best_k=best_k,
        aic_scores=aic_scores,
        losses=losses,
        models=models,
    )


def extract_bottleneck_vectors(model: SciNet, X: np.ndarray) -> np.ndarray:
    """Extract bottleneck (z) vectors from a trained SciNet.

    Args:
        model: trained SciNet
        X: (N, input_dim) input data

    Returns:
        (N, K) numpy array of bottleneck vectors
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        Z = model.encode(X_t).numpy()
    return Z
```

- [ ] **Step 5: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/scinet/ -v`

- [ ] **Step 6: Commit**

```bash
git add atlas/scinet/trainer.py atlas/scinet/bottleneck.py tests/scinet/test_trainer.py tests/scinet/test_bottleneck.py
git commit -m "feat: SciNet trainer with AIC-based K selection and bottleneck extraction"
```

---

## Task 3: Encoder SR (RGDE Step 4b)

**Files:**
- Create: `atlas/rgde/__init__.py`
- Create: `atlas/rgde/encoder_sr.py`
- Create: `tests/rgde/__init__.py`
- Create: `tests/rgde/test_encoder_sr.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rgde/__init__.py
```

```python
# tests/rgde/test_encoder_sr.py
"""Tests for encoder SR (RGDE Step 4b)."""
import pytest
import numpy as np
from atlas.rgde.encoder_sr import run_encoder_sr, EncoderSRResult

try:
    import pysr
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False


def test_encoder_sr_result_structure():
    result = EncoderSRResult(
        formulas={},
        r_squared_per_dim={},
        success=False,
    )
    assert not result.success


@pytest.mark.skipif(not HAS_PYSR, reason="PySR not installed")
def test_encoder_sr_on_known_mapping():
    """If z_0 = x_0 * 2, SR should recover it."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2))
    Z = np.column_stack([X[:, 0] * 2, X[:, 1] + 1])

    result = run_encoder_sr(X, Z, var_names=["knob_0", "knob_1"],
                            niterations=20, maxsize=10)
    assert result.success
    assert len(result.formulas) == 2
    assert all(r2 > 0.9 for r2 in result.r_squared_per_dim.values())


def test_encoder_sr_without_pysr():
    """Should handle missing PySR gracefully."""
    X = np.random.randn(50, 2)
    Z = np.random.randn(50, 2)
    # If PySR is not installed, should return success=False
    # If PySR IS installed, will actually try SR
    result = run_encoder_sr(X, Z, var_names=["knob_0", "knob_1"],
                            niterations=5, maxsize=5)
    assert isinstance(result, EncoderSRResult)
```

- [ ] **Step 2: Implement encoder_sr.py**

```python
# atlas/rgde/__init__.py
"""RGDE: Representation-Grounded DSL Extension."""
```

```python
# atlas/rgde/encoder_sr.py
"""RGDE Step 4b: Symbolic regression on encoder outputs.

For each bottleneck dimension k=1..K, find a symbolic formula:
  z_k = f_k(knob_settings)

This converts the neural encoder into interpretable symbolic mappings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atlas.dsl.expr import Expr
from atlas.types import FitMetrics


@dataclass
class EncoderSRResult:
    """Result of SR on encoder outputs."""
    formulas: dict[int, Expr]          # dim_index -> symbolic formula
    r_squared_per_dim: dict[int, float]
    success: bool                       # True if all dims have R² > threshold


def run_encoder_sr(X: np.ndarray, Z: np.ndarray, var_names: list[str],
                   niterations: int = 40, maxsize: int = 25,
                   r2_threshold: float = 0.8) -> EncoderSRResult:
    """Run SR on each bottleneck dimension.

    Args:
        X: (N, n_knobs) input knob values
        Z: (N, K) bottleneck vectors from trained SciNet
        var_names: knob variable names
        niterations: PySR iterations
        maxsize: max expression size
        r2_threshold: minimum R² for success

    Returns:
        EncoderSRResult with per-dimension formulas
    """
    try:
        from atlas.sr.pysr_wrapper import run_sr, SRConfig
    except ImportError:
        return EncoderSRResult(formulas={}, r_squared_per_dim={}, success=False)

    K = Z.shape[1]
    formulas: dict[int, Expr] = {}
    r2_per_dim: dict[int, float] = {}

    config = SRConfig(niterations=niterations, maxsize=maxsize)

    for k in range(K):
        z_k = Z[:, k]
        try:
            result = run_sr(X, z_k, var_names, config)
            if result.best_formula is not None:
                formulas[k] = result.best_formula
                r2_per_dim[k] = result.best_r_squared
            else:
                r2_per_dim[k] = -1.0
        except Exception:
            r2_per_dim[k] = -1.0

    success = (len(formulas) == K and
               all(r2 > r2_threshold for r2 in r2_per_dim.values()))

    return EncoderSRResult(formulas=formulas, r_squared_per_dim=r2_per_dim,
                           success=success)
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/rgde/test_encoder_sr.py -v`

- [ ] **Step 4: Commit**

```bash
git add atlas/rgde/ tests/rgde/
git commit -m "feat: encoder SR (RGDE Step 4b) — symbolic formulas for bottleneck dims"
```

---

## Task 4: Constraint Finder (RGDE Step 4c)

**Files:**
- Create: `atlas/rgde/constraint_finder.py`
- Create: `tests/rgde/test_constraint_finder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rgde/test_constraint_finder.py
"""Tests for constraint discovery (RGDE Step 4c)."""
import numpy as np
from atlas.rgde.constraint_finder import find_constraints, Constraint


def test_find_sphere_constraint():
    """Points on a unit sphere: z1^2 + z2^2 + z3^2 = 1."""
    rng = np.random.default_rng(42)
    n = 200
    # Generate points on unit sphere
    raw = rng.normal(0, 1, (n, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    Z = raw / norms  # all lie on unit sphere

    constraints = find_constraints(Z, max_degree=2)
    assert len(constraints) >= 1
    # Should find something like sum of squares ≈ constant
    c = constraints[0]
    assert isinstance(c, Constraint)
    assert c.residual < 0.05


def test_find_disk_constraint():
    """Points inside a disk: z1^2 + z2^2 <= 1."""
    rng = np.random.default_rng(42)
    n = 200
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0.5, 1.0, n)  # near boundary
    Z = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    constraints = find_constraints(Z, max_degree=2)
    # Should find sum of squares ≈ bounded
    assert len(constraints) >= 1


def test_no_constraint_for_random():
    """Random points in high-D should have no simple algebraic constraint."""
    rng = np.random.default_rng(42)
    Z = rng.uniform(-1, 1, (200, 3))
    constraints = find_constraints(Z, max_degree=2, max_residual=0.01)
    # Random points shouldn't satisfy any tight algebraic relation
    assert len(constraints) == 0


def test_constraint_structure():
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 1, (200, 2))
    Z = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    constraints = find_constraints(Z, max_degree=2)
    if constraints:
        c = constraints[0]
        assert hasattr(c, "coefficients")
        assert hasattr(c, "degree")
        assert hasattr(c, "constant")
        assert hasattr(c, "residual")
```

- [ ] **Step 2: Implement constraint_finder.py**

```python
# atlas/rgde/constraint_finder.py
"""RGDE Step 4c: Discover algebraic constraints on bottleneck vectors.

Searches for polynomial relations g(z_1, ..., z_K) ≈ constant
by fitting polynomial combinations of z to find near-constant outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np


@dataclass
class Constraint:
    """A discovered algebraic constraint on the bottleneck space."""
    coefficients: np.ndarray    # coefficients of polynomial terms
    terms: list[tuple[int, ...]]  # which variables in each term (indices)
    degree: int
    constant: float             # the near-constant value
    residual: float             # how close to constant (relative std)
    constraint_type: str        # "equality" or "inequality"


def find_constraints(Z: np.ndarray, max_degree: int = 2,
                     max_residual: float = 0.05,
                     min_samples: int = 50) -> list[Constraint]:
    """Find algebraic constraints satisfied by bottleneck vectors.

    Constructs polynomial features up to max_degree, then checks which
    polynomial combinations are approximately constant across all data points.

    Args:
        Z: (N, K) bottleneck vectors
        max_degree: maximum polynomial degree to search
        max_residual: maximum relative std to consider "constant"
        min_samples: minimum data points required
    """
    N, K = Z.shape
    if N < min_samples:
        return []

    # Generate polynomial terms up to max_degree
    terms, features = _polynomial_features(Z, K, max_degree)

    constraints = []

    # Method 1: Check individual polynomial terms for near-constancy
    for i, (term, feat) in enumerate(zip(terms, features.T)):
        if len(term) < 2:  # skip linear terms (trivially non-constant for varied data)
            continue
        mean_val = np.mean(feat)
        if abs(mean_val) < 1e-10:
            continue
        rel_std = np.std(feat) / abs(mean_val)
        if rel_std < max_residual:
            constraints.append(Constraint(
                coefficients=np.array([1.0]),
                terms=[term],
                degree=len(term),
                constant=float(mean_val),
                residual=float(rel_std),
                constraint_type="equality",
            ))

    # Method 2: Search for linear combinations of polynomial terms that are constant
    # Use SVD to find near-null-space of centered feature matrix
    if features.shape[1] >= 2:
        centered = features - np.mean(features, axis=0, keepdims=True)
        try:
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return constraints

        # Small singular values indicate near-constant linear combinations
        if len(s) > 0 and s[0] > 1e-10:
            for idx in range(len(s)):
                rel_sv = s[idx] / s[0]
                if rel_sv < max_residual:
                    coeffs = Vt[idx]
                    # This linear combination of polynomial features is near-constant
                    combo = features @ coeffs
                    mean_val = np.mean(combo)
                    if abs(mean_val) < 1e-10:
                        mean_val = 1.0  # avoid division by zero
                    rel_std = np.std(combo) / abs(mean_val)

                    # Identify which terms have significant coefficients
                    sig_mask = np.abs(coeffs) > 0.01 * np.max(np.abs(coeffs))
                    sig_terms = [terms[j] for j in range(len(terms)) if sig_mask[j]]
                    sig_coeffs = coeffs[sig_mask]

                    if len(sig_terms) >= 1:
                        constraints.append(Constraint(
                            coefficients=sig_coeffs,
                            terms=sig_terms,
                            degree=max(len(t) for t in sig_terms),
                            constant=float(np.mean(features @ coeffs)),
                            residual=float(rel_std),
                            constraint_type="equality",
                        ))

    # Sort by residual (tightest constraints first)
    constraints.sort(key=lambda c: c.residual)

    # Deduplicate: keep only constraints with sufficiently different terms
    filtered = []
    seen_term_sets: list[set] = []
    for c in constraints:
        term_set = {t for t in c.terms}
        if not any(term_set == s for s in seen_term_sets):
            filtered.append(c)
            seen_term_sets.append(term_set)

    return filtered


def _polynomial_features(Z: np.ndarray, K: int,
                         max_degree: int) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Generate polynomial features up to given degree.

    Returns (terms, features) where terms[i] is a tuple of variable indices
    and features[:, i] is the corresponding product.
    """
    terms: list[tuple[int, ...]] = []
    columns: list[np.ndarray] = []

    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(K), degree):
            term = tuple(combo)
            col = np.ones(Z.shape[0])
            for idx in term:
                col = col * Z[:, idx]
            terms.append(term)
            columns.append(col)

    features = np.column_stack(columns) if columns else np.empty((Z.shape[0], 0))
    return terms, features
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/rgde/test_constraint_finder.py -v`

- [ ] **Step 4: Commit**

```bash
git add atlas/rgde/constraint_finder.py tests/rgde/test_constraint_finder.py
git commit -m "feat: constraint finder (RGDE Step 4c) — polynomial relation discovery"
```

---

## Task 5: Type Builder + Evaluator (RGDE Steps 4d, 4f)

**Files:**
- Create: `atlas/rgde/type_builder.py`
- Create: `atlas/rgde/evaluator.py`
- Create: `tests/rgde/test_type_builder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rgde/test_type_builder.py
"""Tests for type builder and Pareto evaluator."""
from atlas.rgde.type_builder import DSLType, build_type
from atlas.rgde.evaluator import evaluate_extension
from atlas.rgde.constraint_finder import Constraint
from atlas.dsl.expr import Var, BinOp, Const
from atlas.dsl.operators import Op
import numpy as np


def test_build_type_basic():
    encoder_formulas = {
        0: BinOp(Op.MUL, Var("knob_0"), Const(2.0)),
        1: BinOp(Op.ADD, Var("knob_1"), Const(0.5)),
    }
    constraints = [
        Constraint(
            coefficients=np.array([1.0, 1.0]),
            terms=[(0, 0), (1, 1)],
            degree=2,
            constant=1.0,
            residual=0.01,
            constraint_type="equality",
        )
    ]
    dsl_type = build_type("ENV_07", encoder_formulas, constraints)
    assert isinstance(dsl_type, DSLType)
    assert dsl_type.name == "State_ENV_07"
    assert dsl_type.dimension == 2
    assert len(dsl_type.constraints) == 1
    assert len(dsl_type.encoding) == 2


def test_build_type_no_constraints():
    encoder_formulas = {0: Var("knob_0")}
    dsl_type = build_type("ENV_01", encoder_formulas, [])
    assert dsl_type.dimension == 1
    assert len(dsl_type.constraints) == 0


def test_evaluate_extension_accepts():
    """Extension that improves R² significantly should be accepted."""
    result = evaluate_extension(
        r2_before=0.3,
        r2_after=0.95,
        mdl_before=5.0,
        mdl_after=12.0,  # DSL got bigger
        type_mdl_cost=7.0,
    )
    assert result.accepted
    assert result.delta_r2 > 0


def test_evaluate_extension_rejects_no_improvement():
    """Extension that doesn't improve R² should be rejected."""
    result = evaluate_extension(
        r2_before=0.95,
        r2_after=0.96,
        mdl_before=5.0,
        mdl_after=20.0,
        type_mdl_cost=15.0,
    )
    assert not result.accepted  # tiny R² gain doesn't justify 3x MDL increase


def test_dsl_type_mdl_cost():
    encoder_formulas = {
        0: BinOp(Op.MUL, Var("knob_0"), Const(2.0)),
    }
    dsl_type = build_type("ENV_01", encoder_formulas, [])
    cost = dsl_type.mdl_cost()
    assert cost > 0
```

- [ ] **Step 2: Implement type_builder.py and evaluator.py**

```python
# atlas/rgde/type_builder.py
"""RGDE Step 4d: Construct new DSL type from discovered structure."""
from __future__ import annotations

from dataclasses import dataclass, field

from atlas.dsl.expr import Expr
from atlas.dsl.serialize import to_str
from atlas.rgde.constraint_finder import Constraint


@dataclass
class DSLType:
    """A discovered state space type."""
    name: str
    dimension: int
    encoding: dict[int, Expr]           # dim_index -> symbolic formula from knobs
    constraints: list[Constraint]
    source_env: str

    def mdl_cost(self) -> float:
        """Description length cost of this type definition."""
        cost = 1.0  # base cost for the type itself
        for expr in self.encoding.values():
            cost += expr.size()
        cost += len(self.constraints) * 3.0  # each constraint adds cost
        return cost

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "encoding": {k: to_str(v) for k, v in self.encoding.items()},
            "constraints": [
                {
                    "terms": c.terms,
                    "constant": c.constant,
                    "residual": c.residual,
                    "type": c.constraint_type,
                }
                for c in self.constraints
            ],
            "source_env": self.source_env,
        }


def build_type(env_id: str, encoder_formulas: dict[int, Expr],
               constraints: list[Constraint]) -> DSLType:
    """Build a DSL type from encoder SR results and discovered constraints.

    Args:
        env_id: source experiment ID
        encoder_formulas: symbolic formulas for each bottleneck dimension
        constraints: algebraic constraints on the bottleneck space
    """
    return DSLType(
        name=f"State_{env_id}",
        dimension=len(encoder_formulas),
        encoding=dict(encoder_formulas),
        constraints=list(constraints),
        source_env=env_id,
    )
```

```python
# atlas/rgde/evaluator.py
"""RGDE Step 4f: Pareto evaluation of proposed DSL extension."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of Pareto evaluation for a proposed extension."""
    accepted: bool
    delta_r2: float         # R² improvement
    delta_mdl: float        # MDL cost increase
    pareto_efficient: bool  # is (delta_r2, delta_mdl) on Pareto front?
    reason: str


def evaluate_extension(r2_before: float, r2_after: float,
                       mdl_before: float, mdl_after: float,
                       type_mdl_cost: float,
                       min_r2_improvement: float = 0.1) -> EvaluationResult:
    """Evaluate whether a proposed DSL extension is worth accepting.

    Uses a simple criterion: the R² improvement must be substantial enough
    to justify the MDL cost increase.

    Args:
        r2_before: best R² before extension
        r2_after: R² with the new type
        mdl_before: formula MDL before extension
        mdl_after: formula MDL after extension (includes new type cost)
        type_mdl_cost: MDL cost of the type definition itself
        min_r2_improvement: minimum R² improvement to consider
    """
    delta_r2 = r2_after - r2_before
    total_mdl_after = mdl_after + type_mdl_cost
    delta_mdl = total_mdl_after - mdl_before

    # Acceptance criteria:
    # 1. R² must improve by at least min_r2_improvement
    # 2. R² improvement per MDL unit must be positive
    if delta_r2 < min_r2_improvement:
        return EvaluationResult(
            accepted=False, delta_r2=delta_r2, delta_mdl=delta_mdl,
            pareto_efficient=False,
            reason=f"R² improvement {delta_r2:.4f} < threshold {min_r2_improvement}",
        )

    # Pareto efficiency: is this a good tradeoff?
    efficiency = delta_r2 / max(delta_mdl, 0.1)

    if efficiency > 0.01:  # at least 0.01 R² per MDL unit
        return EvaluationResult(
            accepted=True, delta_r2=delta_r2, delta_mdl=delta_mdl,
            pareto_efficient=True,
            reason=f"Accepted: R² +{delta_r2:.4f}, efficiency {efficiency:.4f}",
        )

    return EvaluationResult(
        accepted=False, delta_r2=delta_r2, delta_mdl=delta_mdl,
        pareto_efficient=False,
        reason=f"Rejected: efficiency {efficiency:.4f} too low",
    )
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/rgde/test_type_builder.py -v`

- [ ] **Step 4: Commit**

```bash
git add atlas/rgde/type_builder.py atlas/rgde/evaluator.py tests/rgde/test_type_builder.py
git commit -m "feat: type builder + Pareto evaluator (RGDE Steps 4d, 4f)"
```

---

## Task 6: RGDE Pipeline Orchestrator

**Files:**
- Create: `atlas/rgde/pipeline.py`
- Create: `tests/rgde/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rgde/test_pipeline.py
"""Tests for the full RGDE pipeline."""
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from atlas.rgde.pipeline import RGDEConfig, RGDEResult, run_rgde


def test_rgde_config_defaults():
    cfg = RGDEConfig()
    assert cfg.k_range == [1, 2, 3, 4, 5]
    assert cfg.scinet_epochs == 200


def test_rgde_result_structure():
    result = RGDEResult(
        success=False,
        dsl_type=None,
        decoder_formula=None,
        r2_before=0.3,
        r2_after=-1.0,
        evaluation=None,
        k_selected=None,
    )
    assert not result.success


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_rgde_on_sphere_data():
    """Synthetic test: data lives on a sphere, RGDE should find K=3 and sphere constraint."""
    rng = np.random.default_rng(42)
    n = 300

    # Generate points on 2-sphere embedded in 3D knob space
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([
        theta / np.pi,        # knob_0: normalized theta
        phi / (2 * np.pi),    # knob_1: normalized phi
        rng.uniform(0, 1, n), # knob_2: irrelevant knob
    ]).astype(np.float32)

    # Output depends on position on sphere
    y = (np.sin(theta) * np.cos(phi)).reshape(-1, 1).astype(np.float32)

    config = RGDEConfig(
        k_range=[1, 2, 3],
        scinet_epochs=100,
        sr_niterations=10,
        sr_maxsize=10,
    )
    result = run_rgde(X, y, var_names=["knob_0", "knob_1", "knob_2"],
                      r2_before=0.2, env_id="ENV_TEST", config=config)

    assert isinstance(result, RGDEResult)
    assert result.k_selected is not None
    assert result.k_selected >= 2  # needs at least 2 dims for sphere data


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_rgde_simple_1d():
    """Simple 1D case: y = f(x), should find K=1."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    y = (X[:, 0] ** 2).reshape(-1, 1).astype(np.float32)

    config = RGDEConfig(k_range=[1, 2], scinet_epochs=50,
                        sr_niterations=5, sr_maxsize=8)
    result = run_rgde(X, y, var_names=["knob_0", "knob_1"],
                      r2_before=0.1, env_id="ENV_T2", config=config)
    assert result.k_selected is not None
```

- [ ] **Step 2: Implement pipeline.py**

```python
# atlas/rgde/pipeline.py
"""RGDE Pipeline: full orchestration of Steps 4a-4f.

Step 4a: Train SciNet with optimal K
Step 4b: SR on encoder (knobs -> z)
Step 4c: Find constraints on z-space
Step 4d: Build DSL type
Step 4e: SR on decoder (z -> output)  [uses z directly, not knobs]
Step 4f: Pareto evaluation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from atlas.dsl.expr import Expr
from atlas.rgde.type_builder import DSLType, build_type
from atlas.rgde.constraint_finder import find_constraints
from atlas.rgde.evaluator import evaluate_extension, EvaluationResult
from atlas.types import FitMetrics

logger = logging.getLogger(__name__)


@dataclass
class RGDEConfig:
    k_range: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    scinet_epochs: int = 200
    scinet_lr: float = 1e-3
    scinet_seeds: int = 3
    encoder_sparsity: float = 0.01
    sr_niterations: int = 40
    sr_maxsize: int = 25
    constraint_max_degree: int = 2
    constraint_max_residual: float = 0.05
    min_r2_improvement: float = 0.1


@dataclass
class RGDEResult:
    success: bool
    dsl_type: DSLType | None
    decoder_formula: Expr | None
    r2_before: float
    r2_after: float
    evaluation: EvaluationResult | None
    k_selected: int | None
    encoder_r2: dict[int, float] | None = None


def run_rgde(X: np.ndarray, y: np.ndarray, var_names: list[str],
             r2_before: float, env_id: str,
             config: RGDEConfig | None = None) -> RGDEResult:
    """Run the full RGDE pipeline.

    Args:
        X: (N, n_knobs) input data
        y: (N, output_dim) target data
        var_names: knob names
        r2_before: best R² achieved by SR without extension
        env_id: experiment identifier
        config: RGDE configuration
    """
    if config is None:
        config = RGDEConfig()

    try:
        import torch
        from atlas.scinet.model import SciNet
        from atlas.scinet.trainer import train_scinet, TrainConfig
        from atlas.scinet.bottleneck import find_optimal_k, extract_bottleneck_vectors
    except ImportError:
        logger.warning("PyTorch not installed, RGDE unavailable")
        return RGDEResult(success=False, dsl_type=None, decoder_formula=None,
                          r2_before=r2_before, r2_after=-1.0,
                          evaluation=None, k_selected=None)

    X_f = X.astype(np.float32)
    y_f = y.reshape(-1, 1).astype(np.float32) if y.ndim == 1 else y.astype(np.float32)

    # Step 4a: Find optimal K and train SciNet
    logger.info(f"RGDE Step 4a: Finding optimal K for {env_id}")
    k_result = find_optimal_k(
        X_f, y_f,
        k_range=config.k_range,
        epochs_per_k=config.scinet_epochs,
        n_seeds=config.scinet_seeds,
    )
    K = k_result.best_k
    model = k_result.models[K]
    logger.info(f"RGDE: Selected K={K} for {env_id}")

    # Extract bottleneck vectors
    Z = extract_bottleneck_vectors(model, X_f)

    # Step 4b: SR on encoder
    logger.info(f"RGDE Step 4b: SR on encoder ({K} dims)")
    from atlas.rgde.encoder_sr import run_encoder_sr
    encoder_result = run_encoder_sr(
        X, Z, var_names,
        niterations=config.sr_niterations,
        maxsize=config.sr_maxsize,
    )

    # Step 4c: Find constraints
    logger.info("RGDE Step 4c: Finding constraints")
    constraints = find_constraints(
        Z,
        max_degree=config.constraint_max_degree,
        max_residual=config.constraint_max_residual,
    )
    logger.info(f"RGDE: Found {len(constraints)} constraints")

    # Step 4d: Build type
    dsl_type = build_type(env_id, encoder_result.formulas, constraints)

    # Step 4e: SR on decoder (z -> y)
    logger.info("RGDE Step 4e: SR on decoder")
    z_var_names = [f"z_{k}" for k in range(K)]
    decoder_formula = None
    r2_after = -1.0

    try:
        from atlas.sr.pysr_wrapper import run_sr, SRConfig
        sr_config = SRConfig(niterations=config.sr_niterations,
                             maxsize=config.sr_maxsize)
        y_flat = y_f.ravel() if y_f.shape[1] == 1 else np.mean(y_f, axis=1)
        sr_result = run_sr(Z, y_flat, z_var_names, sr_config)
        if sr_result.best_formula is not None:
            decoder_formula = sr_result.best_formula
            r2_after = sr_result.best_r_squared
    except ImportError:
        logger.warning("PySR not installed, skipping decoder SR")
    except Exception as e:
        logger.warning(f"Decoder SR failed: {e}")

    # Step 4f: Pareto evaluation
    mdl_before = 10.0  # approximate MDL of best formula before
    mdl_after = decoder_formula.size() if decoder_formula else float("inf")
    type_cost = dsl_type.mdl_cost()

    evaluation = evaluate_extension(
        r2_before=r2_before,
        r2_after=r2_after,
        mdl_before=mdl_before,
        mdl_after=mdl_after,
        type_mdl_cost=type_cost,
        min_r2_improvement=config.min_r2_improvement,
    )

    success = evaluation.accepted and decoder_formula is not None

    return RGDEResult(
        success=success,
        dsl_type=dsl_type if success else None,
        decoder_formula=decoder_formula if success else None,
        r2_before=r2_before,
        r2_after=r2_after,
        evaluation=evaluation,
        k_selected=K,
        encoder_r2=encoder_result.r_squared_per_dim,
    )
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/rgde/test_pipeline.py -v`

- [ ] **Step 4: Commit**

```bash
git add atlas/rgde/pipeline.py tests/rgde/test_pipeline.py
git commit -m "feat: RGDE pipeline orchestrator (Steps 4a-4f)"
```

---

## Task 7: Integrate RGDE into ATLAS Agent

**Files:**
- Modify: `atlas/agent/atlas_agent.py`
- Create: `tests/agent/test_atlas_agent_rgde.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_atlas_agent_rgde.py
"""Tests for ATLAS agent with RGDE integration."""
import pytest
from atlas.agent.atlas_agent import ATLASAgent, AgentConfig

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pysr
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False


def test_agent_config_has_rgde_fields():
    cfg = AgentConfig()
    assert hasattr(cfg, "enable_rgde")
    assert hasattr(cfg, "rgde_k_range")


def test_agent_rgde_disabled_by_default():
    """RGDE should be disabled by default for backward compatibility."""
    cfg = AgentConfig()
    assert cfg.enable_rgde is False


def test_agent_output_has_rgde_section():
    agent = ATLASAgent(
        env_ids=["ENV_10"],
        config=AgentConfig(max_epochs=1, n_samples_per_knob=5,
                           sr_niterations=5, sr_maxsize=8, enable_rgde=False),
    )
    output = agent.run()
    assert "extensions" in output


@pytest.mark.skipif(not (HAS_TORCH and HAS_PYSR), reason="Requires PyTorch and PySR")
def test_agent_with_rgde_enabled():
    agent = ATLASAgent(
        env_ids=["ENV_10"],
        config=AgentConfig(
            max_epochs=1, n_samples_per_knob=5,
            sr_niterations=5, sr_maxsize=8,
            enable_rgde=True,
            rgde_k_range=[1, 2],
            rgde_scinet_epochs=30,
            rgde_sr_niterations=5,
        ),
    )
    output = agent.run()
    assert "extensions" in output
```

- [ ] **Step 2: Modify atlas_agent.py to add RGDE fields and Step 4**

Add to `AgentConfig`:
```python
    enable_rgde: bool = False
    rgde_k_range: list[int] = field(default_factory=lambda: [1, 2, 3])
    rgde_scinet_epochs: int = 200
    rgde_sr_niterations: int = 40
    rgde_sr_maxsize: int = 25
```

Add Step 4 to `run_epoch()` between Step 3 (Diagnose) and Step 5 (Unify):
```python
        # Step 4: Extend — RGDE on failed experiments (if enabled)
        extensions_found = []
        if self.config.enable_rgde:
            for env_id in failed_envs:
                ds = self.datasets.get(env_id)
                if ds is None or len(ds) < 50:
                    continue
                # Check if any diagnostic triggered D3-like signals
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
                    rgde_result = run_rgde(
                        X, y, ds.knob_names,
                        r2_before=best_r2, env_id=env_id,
                        config=rgde_config,
                    )
                    if rgde_result.success and rgde_result.dsl_type is not None:
                        self.dsl_state.add_extension(
                            name=rgde_result.dsl_type.name,
                            ext_type="new_type",
                            definition=rgde_result.dsl_type.to_dict(),
                            trigger=f"RGDE on {env_id}, K={rgde_result.k_selected}",
                        )
                        extensions_found.append(rgde_result.dsl_type.name)
                except ImportError:
                    logger.warning("RGDE unavailable (missing PyTorch or PySR)")
                except Exception as e:
                    logger.warning(f"RGDE failed for {env_id}: {e}")
```

Add `"extensions"` to the output dict in `run()`:
```python
        return {
            ...existing fields...,
            "extensions": [e for e in self.dsl_state.extensions],
        }
```

Update `EpochResult` to include `extensions_found`:
```python
@dataclass
class EpochResult:
    ...existing fields...
    extensions_found: list[str] = field(default_factory=list)
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/agent/ -v`
Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/ -v --tb=short`

- [ ] **Step 4: Commit**

```bash
git add atlas/agent/atlas_agent.py tests/agent/test_atlas_agent_rgde.py
git commit -m "feat: integrate RGDE (Step 4) into ATLAS agent main loop"
```

---

## Summary

After completing all 7 tasks, the project has:

- **SciNet**: PyTorch encoder-decoder with configurable bottleneck, AIC-based K selection
- **RGDE Pipeline**: Full 6-step process (4a train → 4b encoder SR → 4c constraints → 4d type → 4e decoder SR → 4f Pareto eval)
- **Integration**: RGDE plugged into ATLAS agent as Step 4, enabled via config flag

This completes the single-agent ATLAS system. **Plan 4** will add multi-agent consensus + Unifier.
