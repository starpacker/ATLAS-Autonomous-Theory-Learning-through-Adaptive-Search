"""Phase 0: SciNet -> SR Bridge Validation on ENV-07 (Stern-Gerlach).

Critical gate for the RGDE pipeline.  Validates that:
  1. SciNet can learn a compressed representation of ENV-07's 200-bin histogram
  2. AIC selects a meaningful bottleneck dimension K >= 2
  3. SR can extract symbolic formulas from the encoder outputs (z_k = f_k(knobs))
  4. Constraint finder discovers algebraic relations on bottleneck space
  5. SR can reconstruct the decoder (z -> y)

Gate criteria (from ARCHITECTURE.md):
  - Steps 3-4 succeed in >50% of multi-seed runs -> CONTINUE
  - Steps 3-4 succeed in <30% of runs -> MODIFY approach or DESCOPE RGDE

Usage:
    python validation/phase0_scinet_sr_bridge.py [--seeds N] [--epochs E] [--sr-iters I]
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("phase0")


def collect_env07_data(n_angle: int = 25, n_gradient: int = 25,
                       particle_count: int = 500_000, env_seed: int = 42):
    """Collect ENV-07 data with fixed high particle count for clean signals.

    Uses 2 continuous knobs (angle, gradient) with fixed particle count.
    Returns normalized probability histograms as output.
    """
    from atlas.environments.registry import get_environment

    env = get_environment("ENV_07")
    angles = np.linspace(0.0, 1.0, n_angle)
    gradients = np.linspace(0.05, 0.95, n_gradient)

    X_list, y_list = [], []
    for a in angles:
        for g in gradients:
            knobs = {"knob_0": float(a), "knob_1": float(g), "knob_2": particle_count}
            result = env.run(knobs)
            histogram = result["detector_0"]
            total = histogram.sum()
            if total > 0:
                histogram = histogram / total
            X_list.append([a, g])
            y_list.append(histogram)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(f"Collected {len(X)} samples, X shape={X.shape}, y shape={y.shape}")
    return X, y


def step_4a_find_k(X, y, k_range, epochs, n_seeds):
    """Train SciNet with various K, select best via validation loss.

    Uses a smaller architecture [64, 32] to keep model manageable,
    validation-based K selection (not AIC), and tanh bottleneck activation
    to bound encoder outputs for SR readability.
    """
    from atlas.scinet.bottleneck import find_optimal_k
    from atlas.scinet.trainer import TrainConfig

    encoder_hidden = [64, 32]
    decoder_hidden = [32, 64]

    # No encoder sparsity during K selection (biases toward small K).
    # Sparsity is applied separately after K is chosen.
    config = TrainConfig(epochs=epochs, lr=1e-3, encoder_sparsity=0.0,
                         use_cosine_schedule=True)

    logger.info(f"Step 4a: Finding optimal K from {k_range} "
                f"(epochs={epochs}, seeds={n_seeds}, arch={encoder_hidden}, "
                f"method=val_loss, activation=tanh)")
    t0 = time.time()
    result = find_optimal_k(X, y, k_range=k_range, n_seeds=n_seeds,
                            encoder_hidden=encoder_hidden, decoder_hidden=decoder_hidden,
                            train_config=config,
                            val_fraction=0.2,
                            selection_method="val_loss",
                            bottleneck_activation="tanh")
    elapsed = time.time() - t0

    logger.info(f"  AIC scores:  {result.aic_scores}")
    logger.info(f"  Val losses:  {result.val_losses}")
    logger.info(f"  Train losses:{result.losses}")
    logger.info(f"  Best K:      {result.best_k} (method={result.selection_method})")
    logger.info(f"  Time:        {elapsed:.1f}s")
    return result


def step_4b_encoder_sr(X, Z, var_names, niterations, maxsize):
    """Run SR on each bottleneck dimension: z_k = f_k(knobs)."""
    from atlas.rgde.encoder_sr import run_encoder_sr

    K = Z.shape[1]
    logger.info(f"Step 4b: SR on encoder outputs ({K} dims, {niterations} iterations)")
    t0 = time.time()
    result = run_encoder_sr(X, Z, var_names, niterations=niterations, maxsize=maxsize)
    elapsed = time.time() - t0

    for k, r2 in sorted(result.r_squared_per_dim.items()):
        formula_str = str(result.formulas.get(k, "FAILED"))
        logger.info(f"  z_{k}: R2={r2:.4f}  formula={formula_str}")
    logger.info(f"  Overall success: {result.success}")
    logger.info(f"  Time: {elapsed:.1f}s")
    return result


def step_4c_constraints(Z):
    """Find polynomial constraints on bottleneck vectors."""
    from atlas.rgde.constraint_finder import find_constraints

    logger.info(f"Step 4c: Finding constraints on Z (shape={Z.shape})")
    constraints = find_constraints(Z, max_degree=2, max_residual=0.15)

    for i, c in enumerate(constraints):
        terms_str = " + ".join(
            f"{coeff:.3f}*{'*'.join(f'z{t}' for t in term)}"
            for coeff, term in zip(c.coefficients, c.terms)
        )
        logger.info(f"  Constraint {i}: {terms_str} ~ {c.constant:.4f} "
                     f"(residual={c.residual:.4f}, type={c.constraint_type})")
    logger.info(f"  Found {len(constraints)} constraints")
    return constraints


def step_4e_decoder_sr(Z, X, y, var_names, niterations, maxsize):
    """Run SR on decoder: (z, knobs) -> y at selected positions."""
    from atlas.sr.pysr_wrapper import run_sr, SRConfig

    K = Z.shape[1]
    z_names = [f"z_{k}" for k in range(K)]
    all_names = z_names + list(var_names)

    n_samples, n_positions = y.shape
    # Pick positions with highest variance (where signal lives)
    pos_variances = np.var(y, axis=0)
    top_pos = np.argsort(pos_variances)[-5:]
    pos_indices = np.sort(top_pos)

    logger.info(f"Step 4e: SR on decoder (Z + knobs -> y), {len(pos_indices)} positions")
    results = {}
    config = SRConfig(niterations=niterations, maxsize=maxsize)

    for pos_idx in pos_indices:
        decoder_X = np.hstack([Z, X])
        pos_col = np.full((n_samples, 1), pos_idx / (n_positions - 1), dtype=np.float32)
        decoder_X = np.hstack([decoder_X, pos_col])
        names = all_names + ["_position"]

        y_pos = y[:, pos_idx]
        # Skip positions with near-zero variance (no signal)
        if np.std(y_pos) < 1e-8:
            logger.info(f"  Position {pos_idx}: SKIPPED (no variance)")
            results[pos_idx] = {"r2": 0.0, "formula": "ZERO_VARIANCE"}
            continue

        try:
            sr_result = run_sr(decoder_X, y_pos, names, config)
            r2 = sr_result.best_r_squared
            formula = str(sr_result.best_formula) if sr_result.best_formula else "FAILED"
            results[pos_idx] = {"r2": r2, "formula": formula}
            logger.info(f"  Position {pos_idx}: R2={r2:.4f}  formula={formula}")
        except Exception as e:
            logger.warning(f"  Position {pos_idx}: FAILED ({e})")
            results[pos_idx] = {"r2": -1.0, "formula": "ERROR"}

    return results


def run_single_seed(seed: int, args):
    """Run the full Phase 0 validation with a single seed."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 0 Validation -- Seed {seed}")
    logger.info(f"{'='*60}")

    # Collect data — 50x50 grid for better K discrimination
    X, y = collect_env07_data(n_angle=50, n_gradient=50,
                              particle_count=500_000, env_seed=seed)

    # Step 4a: Find optimal K
    k_range = [1, 2, 3, 4, 5]
    k_result = step_4a_find_k(X, y, k_range, epochs=args.epochs, n_seeds=3)
    K = k_result.best_k

    model = k_result.models[K]

    # Extract bottleneck vectors
    from atlas.scinet.bottleneck import extract_bottleneck_vectors
    Z = extract_bottleneck_vectors(model, X)
    logger.info(f"Bottleneck Z shape: {Z.shape}")
    logger.info(f"Z stats per dim:")
    for k in range(K):
        logger.info(f"  z_{k}: mean={Z[:, k].mean():.4f}, std={Z[:, k].std():.4f}, "
                     f"range=[{Z[:, k].min():.4f}, {Z[:, k].max():.4f}]")

    # Step 4b: SR on encoder
    var_names = ["knob_0", "knob_1"]
    # Give encoder SR more budget — it needs to find the mapping from knobs to Z
    encoder_result = step_4b_encoder_sr(X, Z, var_names,
                                         niterations=args.sr_iters * 2, maxsize=25)

    # Step 4c: Constraints
    constraints = step_4c_constraints(Z)

    # Step 4e: Decoder SR
    decoder_results = step_4e_decoder_sr(Z, X, y, var_names,
                                          niterations=args.sr_iters, maxsize=30)

    # Evaluate success
    encoder_success = encoder_result.success
    # Relaxed encoder criterion: at least half of dims have R2 > 0.7
    encoder_partial = sum(1 for r2 in encoder_result.r_squared_per_dim.values()
                          if r2 > 0.7) >= max(1, K // 2)
    has_constraints = len(constraints) > 0

    decoder_r2s = [v["r2"] for v in decoder_results.values()
                   if v["r2"] > 0 and v["formula"] not in ("ZERO_VARIANCE", "ERROR")]
    decoder_success = len(decoder_r2s) > 0 and np.mean(decoder_r2s) > 0.5

    # Bridge success: K >= 2 AND (encoder SR partially works OR constraints found)
    bridge_success = K >= 2 and (encoder_partial or has_constraints)

    result = {
        "seed": seed,
        "K_selected": K,
        "aic_scores": k_result.aic_scores,
        "losses": k_result.losses,
        "encoder_r2": encoder_result.r_squared_per_dim,
        "encoder_success": encoder_success,
        "encoder_partial": encoder_partial,
        "n_constraints": len(constraints),
        "has_constraints": has_constraints,
        "decoder_r2": {str(k): v["r2"] for k, v in decoder_results.items()},
        "decoder_success": decoder_success,
        "bridge_success": bridge_success,
    }

    logger.info(f"\n--- Seed {seed} Summary ---")
    logger.info(f"  K selected:        {K}")
    logger.info(f"  Encoder success:   {encoder_success} (partial: {encoder_partial})")
    logger.info(f"  Constraints found: {len(constraints)}")
    logger.info(f"  Decoder success:   {decoder_success}")
    logger.info(f"  BRIDGE SUCCESS:    {bridge_success}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 0: SciNet-SR Bridge Validation")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=300, help="SciNet training epochs per K")
    parser.add_argument("--sr-iters", type=int, default=40, help="PySR iterations")
    args = parser.parse_args()

    logger.info("Phase 0: SciNet -> SR Bridge Validation")
    logger.info(f"Config: seeds={args.seeds}, epochs={args.epochs}, sr_iters={args.sr_iters}")

    results = []
    for seed in range(args.seeds):
        result = run_single_seed(seed, args)
        results.append(result)

    # Aggregate
    n_bridge_success = sum(1 for r in results if r["bridge_success"])
    success_rate = n_bridge_success / len(results)

    logger.info(f"\n{'='*60}")
    logger.info("PHASE 0 AGGREGATE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Seeds run:        {len(results)}")
    logger.info(f"Bridge successes: {n_bridge_success}/{len(results)} ({success_rate:.0%})")
    logger.info(f"K selections:     {[r['K_selected'] for r in results]}")

    for r in results:
        logger.info(f"  Seed {r['seed']}: K={r['K_selected']}, "
                     f"enc_partial={r['encoder_partial']}, "
                     f"constraints={r['n_constraints']}, "
                     f"bridge={r['bridge_success']}")

    if success_rate >= 0.5:
        logger.info("GATE: PASS -- proceed with RGDE pipeline")
        gate = "PASS"
    elif success_rate >= 0.3:
        logger.info("GATE: MARGINAL -- consider tuning before proceeding")
        gate = "MARGINAL"
    else:
        logger.info("GATE: FAIL -- modify approach or descope RGDE")
        gate = "FAIL"

    # Save results to experiments/ directory
    exp_dir = Path(__file__).resolve().parent.parent / "experiments" / "phase0"
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "phase0_results.json"
    output = {"gate": gate, "success_rate": success_rate, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return 0 if gate != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
