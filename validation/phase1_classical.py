"""Phase 1: Single-Agent Formula Discovery on Classical Experiments.

Validates that the ATLAS agent can discover known formulas from anonymized
experiment data using PySR symbolic regression.

Experiments tested (in order of difficulty):
  ENV-11  Free fall        y = v0*t - 0.5*g*t^2           (trivial)
  ENV-12  Heat conduction  Q = k*A*dT/L                   (trivial)
  ENV-10  Spring           x = A*cos(sqrt(k)*t)           (moderate)
  ENV-09  Elastic collision  v1_f = f(m_ratio, v1, v2)    (moderate)
  ENV-08  Water wave       I = cos^2(pi*d*x/(lambda*L))   (hard, array)

Also validates diagnostics:
  - D1 (stochasticity) must NOT trigger on any classical env
  - D2 (discreteness) must NOT trigger on any classical env

Usage:
    python validation/phase1_classical.py [--env ENV_ID] [--sr-iters N]

Gate criteria:
    PASS:     ENV-11, ENV-12 R^2 > 0.95 AND no false D1/D2 triggers
    FULL:     + ENV-10, ENV-09 R^2 > 0.95, ENV-08 R^2 > 0.90
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("phase1")

# Experiment configs: env_id -> (n_samples_per_knob, sr_niterations, sr_maxsize, r2_target)
EXPERIMENTS = {
    "ENV_11": {"n_samples": 15, "sr_iters": 40, "sr_maxsize": 20, "r2_target": 0.95,
               "label": "Free Fall (quadratic)"},
    "ENV_12": {"n_samples": 10, "sr_iters": 40, "sr_maxsize": 20, "r2_target": 0.95,
               "label": "Heat Conduction (rational)"},
    "ENV_10": {"n_samples": 12, "sr_iters": 60, "sr_maxsize": 30, "r2_target": 0.95,
               "label": "Spring (cos+sqrt)"},
    "ENV_09": {"n_samples": 12, "sr_iters": 60, "sr_maxsize": 25, "r2_target": 0.95,
               "label": "Elastic Collision (rational)"},
    "ENV_08": {"n_samples": 8, "sr_iters": 80, "sr_maxsize": 30, "r2_target": 0.90,
               "label": "Water Wave (array, cos^2)"},
}


def run_single_env(env_id: str, cfg: dict, seed: int = 42) -> dict:
    """Run single-agent on one environment and return results."""
    from atlas.agent.atlas_agent import ATLASAgent, AgentConfig

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {env_id}: {cfg['label']}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    agent = ATLASAgent(
        env_ids=[env_id],
        config=AgentConfig(
            max_epochs=3,
            n_samples_per_knob=cfg["n_samples"],
            sr_niterations=cfg["sr_iters"],
            sr_maxsize=cfg["sr_maxsize"],
            sr_timeout=300,
            seed=seed,
        ),
    )

    output = agent.run()
    elapsed = time.time() - t0

    # Extract results
    formula_str = output["formulas"].get(env_id, "NONE")
    fit = output["fit_metrics"].get(env_id, {})
    r2 = fit.get("r_squared", -1.0)
    diagnostics = output["diagnostics"].get(env_id, [])

    # Check D1/D2 false triggers
    d1_triggered = any(d.get("diagnostic_id") == "D1" and d.get("triggered")
                       for d in diagnostics)
    d2_triggered = any(d.get("diagnostic_id") == "D2" and d.get("triggered")
                       for d in diagnostics)

    result = {
        "env_id": env_id,
        "label": cfg["label"],
        "r2": r2,
        "r2_target": cfg["r2_target"],
        "r2_pass": r2 >= cfg["r2_target"],
        "formula": formula_str,
        "d1_triggered": d1_triggered,
        "d2_triggered": d2_triggered,
        "diagnostics_pass": not d1_triggered and not d2_triggered,
        "elapsed_s": elapsed,
    }

    logger.info(f"  R^2:       {r2:.4f} (target: {cfg['r2_target']})")
    logger.info(f"  Formula:   {formula_str}")
    logger.info(f"  D1 false:  {d1_triggered}")
    logger.info(f"  D2 false:  {d2_triggered}")
    logger.info(f"  Time:      {elapsed:.1f}s")
    logger.info(f"  PASS:      {result['r2_pass'] and result['diagnostics_pass']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Classical Experiment Validation")
    parser.add_argument("--env", type=str, default=None,
                        help="Run only this environment (e.g. ENV_11)")
    parser.add_argument("--sr-iters", type=int, default=None,
                        help="Override SR iterations for all environments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    logger.info("Phase 1: Single-Agent Classical Experiment Validation")

    envs_to_run = EXPERIMENTS
    if args.env:
        if args.env not in EXPERIMENTS:
            logger.error(f"Unknown environment: {args.env}")
            logger.error(f"Available: {list(EXPERIMENTS.keys())}")
            return 1
        envs_to_run = {args.env: EXPERIMENTS[args.env]}

    if args.sr_iters:
        for cfg in envs_to_run.values():
            cfg["sr_iters"] = args.sr_iters

    results = []
    for env_id, cfg in envs_to_run.items():
        try:
            result = run_single_env(env_id, cfg, seed=args.seed)
            results.append(result)
        except Exception as e:
            logger.error(f"  CRASH: {e}")
            results.append({
                "env_id": env_id, "label": cfg["label"],
                "r2": -1.0, "r2_target": cfg["r2_target"],
                "r2_pass": False, "formula": f"CRASH: {e}",
                "d1_triggered": False, "d2_triggered": False,
                "diagnostics_pass": True, "elapsed_s": 0,
            })

    # Aggregate
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1 AGGREGATE RESULTS")
    logger.info(f"{'='*60}")

    gate_envs = {"ENV_11", "ENV_12"}  # minimum gate
    full_envs = {"ENV_10", "ENV_09", "ENV_08"}  # full success

    gate_pass = all(
        r["r2_pass"] and r["diagnostics_pass"]
        for r in results if r["env_id"] in gate_envs
    )
    full_pass = gate_pass and all(
        r["r2_pass"] for r in results if r["env_id"] in full_envs
    )
    diag_pass = all(r["diagnostics_pass"] for r in results)

    for r in results:
        status = "PASS" if r["r2_pass"] and r["diagnostics_pass"] else "FAIL"
        logger.info(f"  {r['env_id']} ({r['label']}): "
                     f"R^2={r['r2']:.4f} [{status}]  "
                     f"D1={r['d1_triggered']} D2={r['d2_triggered']}  "
                     f"({r['elapsed_s']:.0f}s)")

    logger.info(f"\n  Gate (ENV-11, ENV-12):  {'PASS' if gate_pass else 'FAIL'}")
    logger.info(f"  Diagnostics:           {'PASS' if diag_pass else 'FAIL'}")
    logger.info(f"  Full (all 5 envs):     {'PASS' if full_pass else 'FAIL'}")

    # Save results to experiments/ directory
    exp_dir = Path(__file__).resolve().parent.parent / "experiments" / "phase1"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save per-environment results
    for r in results:
        env_dir = exp_dir / r["env_id"]
        env_dir.mkdir(parents=True, exist_ok=True)
        with open(env_dir / "result.json", "w") as f:
            json.dump(r, f, indent=2, default=str)

    # Save aggregate results
    output_path = exp_dir / "phase1_results.json"
    output = {
        "gate_pass": gate_pass,
        "full_pass": full_pass,
        "diagnostics_pass": diag_pass,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {exp_dir}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
