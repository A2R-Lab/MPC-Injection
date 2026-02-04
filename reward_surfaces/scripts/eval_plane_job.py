"""Evaluate a single point on the reward surface plane."""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os

# Suppress JAX warnings
os.environ["JAX_PLATFORMS"] = "cpu"  # Force CPU for individual eval jobs
import warnings
warnings.filterwarnings("ignore")

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
reward_surfaces_root = script_dir.parent
mpc_rl_root = reward_surfaces_root.parent
sys.path.insert(0, str(reward_surfaces_root))
sys.path.insert(0, str(mpc_rl_root))

# Import SBX adapter
if __name__ == "__main__" or __package__ is None:
    from scripts.sbx_adapter import SBXRewardSurfaceEvaluator
    from reward_surfaces.utils import readz
else:
    from .sbx_adapter import SBXRewardSurfaceEvaluator
    from ..utils import readz


def main():
    parser = argparse.ArgumentParser(description='Evaluate single point on reward surface')
    parser.add_argument('job_dir', type=str, help='Job directory with info.json and directions')
    parser.add_argument('--offset1', type=float, required=True, help='Offset along direction 1')
    parser.add_argument('--offset2', type=float, required=True, help='Offset along direction 2')
    
    args = parser.parse_args()
    
    job_dir = Path(args.job_dir)
    
    # Load metadata
    with open(job_dir / "info.json", 'r') as f:
        info = json.load(f)
    
    # Load model and create evaluator
    checkpoint_path = info['checkpoint_path']
    domain = info['domain']
    task = info['task']
    vecnormalize_path = info.get('vecnormalize_path')
    num_episodes = info['num_episodes']
    grid_size = info['grid_size']
    magnitude = info['magnitude']
    
    print(f"Loading model from {checkpoint_path}")
    evaluator = SBXRewardSurfaceEvaluator(
        model_path=checkpoint_path,
        domain=domain,
        task=task,
        vecnormalize_path=vecnormalize_path,
        seed=42,
    )
    
    # Get base weights
    base_weights = evaluator.get_weights()
    
    # Load direction vectors using readz to maintain order
    dir1_vec = readz(str(job_dir / "dir1.npz"))
    dir2_vec = readz(str(job_dir / "dir2.npz"))
    
    # Verify shapes match
    if len(base_weights) != len(dir1_vec) or len(base_weights) != len(dir2_vec):
        raise ValueError(f"Shape mismatch: base has {len(base_weights)} params, "
                        f"dir1 has {len(dir1_vec)}, dir2 has {len(dir2_vec)}")
    
    # Apply offsets with magnitude scaling
    offset1_scalar = args.offset1 / (grid_size // 2)
    offset2_scalar = args.offset2 / (grid_size // 2)
    
    perturbed_weights = []
    for base, d1, d2 in zip(base_weights, dir1_vec, dir2_vec):
        # Multiply by magnitude to scale the perturbation
        perturbed = base + magnitude * offset1_scalar * d1 + magnitude * offset2_scalar * d2
        perturbed_weights.append(perturbed)
    
    # Set perturbed weights
    evaluator.set_weights(perturbed_weights)
    
    # Evaluate
    print(f"Evaluating at offset ({args.offset1}, {args.offset2}) for {num_episodes} episodes...")
    results = evaluator.evaluate(num_episodes)
    
    # Add offset information to results
    results.update({
        "dim0": offset1_scalar,
        "dim1": offset2_scalar,
        "offset1": args.offset1,
        "offset2": args.offset2,
        "magnitude": magnitude,
        "grid_size": grid_size,
    })
    
    # Save results
    results_dir = job_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    job_name = f"{args.offset1},{args.offset2}.json"
    output_path = results_dir / job_name
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Mean reward: {results['episode_rewards']:.2f} ± {results['episode_std_rewards']:.2f}")


if __name__ == "__main__":
    main()
