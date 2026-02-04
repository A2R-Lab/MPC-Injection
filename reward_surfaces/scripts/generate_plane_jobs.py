"""Generate reward surface evaluation jobs from a trained checkpoint."""

import argparse
import json
from pathlib import Path
import sys

# Add parent directories to path for imports
script_dir = Path(__file__).parent
reward_surfaces_root = script_dir.parent
mpc_rl_root = reward_surfaces_root.parent
sys.path.insert(0, str(reward_surfaces_root))
sys.path.insert(0, str(mpc_rl_root))

from reward_surfaces.core import filter_normalized_params, generate_plane_data
from reward_surfaces.utils import readz
from scripts.sbx_adapter import SBXRewardSurfaceEvaluator


def main():
    parser = argparse.ArgumentParser(description='Generate reward surface jobs')
    parser.add_argument('checkpoint_path', type=str,
                       help='Path to checkpoint (.zip file or directory containing best_model.zip)')
    parser.add_argument('output_dir', type=str,
                       help='Output directory for surface evaluation')
    parser.add_argument('--grid-size', type=int, default=31,
                       help='Grid size (must be odd, e.g., 31 for 31x31 grid)')
    parser.add_argument('--magnitude', type=float, default=1.0,
                       help='Magnitude scale for directions')
    parser.add_argument('--num-episodes', type=int, default=50,
                       help='Number of episodes per evaluation point')
    parser.add_argument('--dir1', type=str, help='Override dir1 with .npz file')
    parser.add_argument('--dir2', type=str, help='Override dir2 with .npz file')
    parser.add_argument('--env-name', type=str,
                       help='Environment name (domain-task format, e.g., walker-walk)')
    parser.add_argument('--vecnormalize', type=str,
                       help='Path to vec_normalize.pkl file')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    
    # Handle checkpoint path - could be .zip file or directory
    if checkpoint_path.is_dir():
        # Look for best_model.zip in directory
        model_zip = checkpoint_path / "best_model.zip"
        if not model_zip.exists():
            raise FileNotFoundError(f"No best_model.zip found in {checkpoint_path}")
        checkpoint_path = model_zip
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load config from parent directory
    config_path = checkpoint_path.parent.parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        env_name = config.get('env_name', args.env_name)
        domain = config.get('domain')
        task = config.get('task')
        print(f"Loaded config from {config_path}")
        print(f"Environment: {env_name} ({domain}/{task})")
    else:
        if not args.env_name:
            raise ValueError("No config.json found. Please specify --env-name")
        env_name = args.env_name
        parts = env_name.split('-', 1)
        domain = parts[0]
        task = parts[1] if len(parts) > 1 else parts[0]
        config = {'env_name': env_name, 'domain': domain, 'task': task}
    
    # Find vecnormalize file if not specified
    vecnormalize_path = None
    if args.vecnormalize:
        vecnormalize_path = Path(args.vecnormalize)
    else:
        # Look in common locations
        possible_paths = [
            checkpoint_path.parent.parent / "vec_normalize.pkl",
            checkpoint_path.parent / "vec_normalize.pkl",
        ]
        for p in possible_paths:
            if p.exists():
                vecnormalize_path = p
                break
    
    if vecnormalize_path:
        print(f"Using VecNormalize: {vecnormalize_path}")
    else:
        print("Warning: No VecNormalize file found. Evaluation may not match training.")
    
    # Create evaluator
    print(f"Loading model from {checkpoint_path}...")
    evaluator = SBXRewardSurfaceEvaluator(
        model_path=str(checkpoint_path),
        domain=domain,
        task=task,
        vecnormalize_path=str(vecnormalize_path) if vecnormalize_path else None,
        seed=42,
    )
    
    # Generate or load direction vectors
    if args.dir1 and args.dir2:
        print(f"Loading directions from {args.dir1} and {args.dir2}")
        dir1_vec = readz(args.dir1)
        dir2_vec = readz(args.dir2)
    else:
        print("Generating filter-normalized random directions...")
        dir1_vec, dir2_vec = filter_normalized_params(evaluator)
    
    # Create metadata
    metadata = {
        'checkpoint_path': str(checkpoint_path),
        'env_name': env_name,
        'domain': domain,
        'task': task,
        'vecnormalize_path': str(vecnormalize_path) if vecnormalize_path else None,
    }
    
    # Generate plane data
    print(f"Generating plane data with grid size {args.grid_size}x{args.grid_size}...")
    generate_plane_data(
        output_path=output_dir,
        dir1_vec=dir1_vec,
        dir2_vec=dir2_vec,
        magnitude=args.magnitude,
        metadata=metadata,
        grid_size=args.grid_size,
        num_episodes=args.num_episodes,
    )
    
    print(f"\nSetup complete! Run evaluation jobs with:")
    print(f"  bash {output_dir / 'jobs.sh'}")
    print(f"\nOr in parallel:")
    print(f"  python -m reward_surfaces.scripts.run_jobs_multiproc --num-cpus=8 {output_dir / 'jobs.sh'}")


if __name__ == "__main__":
    main()
