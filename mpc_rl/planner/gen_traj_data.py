import numpy as np
from pathlib import Path
import sys
from itertools import product

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mpc_rl.planner.mpc_planner import MPCPlanner


def generate_filename(qpos, qvel, rollout_horizon):
    """
    Generate filename from initial state and rollout horizon.
    Format: qpos_[x.xx,x.xx]_qvel_[x.xx,x.xx]_rh_XXXXX.npz
    Values are rounded to 2 decimal places for ease of comparison.
    """
    qpos_str = ",".join([f"{q:.2f}" for q in qpos])
    qvel_str = ",".join([f"{v:.2f}" for v in qvel])
    filename = f"qpos_[{qpos_str}]_qvel_[{qvel_str}]_rh_{rollout_horizon}.npz"
    return filename


def generate_trajectories(
    qpos0_range=(-1.0, 1.0),
    qpos1_range=None,  # Will default to (pi-1, pi+1)
    qvel0_range=(-1.0, 1.0),
    qvel1_range=(-1.0, 1.0),
    interval=0.10,
    rollout_horizon=10000,
    opt_steps=10,
    weights=None,
    task_params=None,
    output_dir=None,
    verbose=1,
    partition_idx=None,
    num_partitions=None
    ):
    """
    Generate trajectories by scanning over initial states.
    
    Args:
        qpos0_range: Tuple (min, max) for qpos[0] (cart position)
        qpos1_range: Tuple (min, max) for qpos[1] (pole angle), defaults to (pi-1, pi+1)
        qvel0_range: Tuple (min, max) for qvel[0] (cart velocity)
        qvel1_range: Tuple (min, max) for qvel[1] (pole velocity)
        interval: Step size for scanning over the ranges
        rollout_horizon: Total length of each trajectory
        opt_steps: Number of optimization steps for MPC
        weights: Cost weights dictionary
        task_params: Task parameters dictionary
        output_dir: Directory to save trajectories (defaults to ../../data/)
        verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
    """
    # Set defaults
    if qpos1_range is None:
        qpos1_range = (np.pi - 1.0, np.pi + 1.0)
    
    if weights is None:
        weights = {
            "Vertical": 10.0,
            "Centered": 10.0,
            "Velocity": 0.1,
            "Control": 0.1
        }
    
    if task_params is None:
        task_params = {"Goal": 0.0}
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create planner instance
    planner = MPCPlanner(
        rollout_horizon=rollout_horizon,
        opt_steps=opt_steps,
        weights=weights,
        task_params=task_params,
        init_state_noise_flag=False,
        verbose=0  # Let this function handle verbosity
    )
    
    # Generate ranges for each state variable
    qpos0_vals = np.arange(qpos0_range[0], qpos0_range[1] + interval/2, interval)
    qpos1_vals = np.arange(qpos1_range[0], qpos1_range[1] + interval/2, interval)
    qvel0_vals = np.arange(qvel0_range[0], qvel0_range[1] + interval/2, interval)
    qvel1_vals = np.arange(qvel1_range[0], qvel1_range[1] + interval/2, interval)
    
    # Calculate total number of trajectories
    total_trajs = len(qpos0_vals) * len(qpos1_vals) * len(qvel0_vals) * len(qvel1_vals)
    
    if verbose > 0:
        print(f"Generating {total_trajs} trajectories:")
        print(f"  qpos[0]: {len(qpos0_vals)} values from {qpos0_range[0]:.2f} to {qpos0_range[1]:.2f}")
        print(f"  qpos[1]: {len(qpos1_vals)} values from {qpos1_range[0]:.2f} to {qpos1_range[1]:.2f}")
        print(f"  qvel[0]: {len(qvel0_vals)} values from {qvel0_range[0]:.2f} to {qvel0_range[1]:.2f}")
        print(f"  qvel[1]: {len(qvel1_vals)} values from {qvel1_range[0]:.2f} to {qvel1_range[1]:.2f}")
        print(f"  Interval: {interval:.2f}")
        print(f"  Rollout horizon: {rollout_horizon}")
        print(f"  Output directory: {output_dir}")
        print()
    
    # Generate all combinations
    all_combinations = list(product(qpos0_vals, qpos1_vals, qvel0_vals, qvel1_vals))
    
    # Apply partitioning if specified
    if partition_idx is not None and num_partitions is not None:
        if partition_idx >= num_partitions:
            raise ValueError(f"partition_idx ({partition_idx}) must be less than num_partitions ({num_partitions})")
        
        # Split combinations into partitions
        partition_size = len(all_combinations) // num_partitions
        start_idx = partition_idx * partition_size
        
        # Last partition gets any remainder
        if partition_idx == num_partitions - 1:
            end_idx = len(all_combinations)
        else:
            end_idx = start_idx + partition_size
        
        all_combinations = all_combinations[start_idx:end_idx]
        
        if verbose > 0:
            print(f"Running partition {partition_idx + 1}/{num_partitions}")
            print(f"Processing trajectories {start_idx + 1} to {end_idx} of {total_trajs}")
            print()
    
    # Generate trajectories for assigned combinations
    traj_count = 0
    for qp0, qp1, qv0, qv1 in all_combinations:
        traj_count += 1
        
        # Create initial state
        init_qpos = np.array([qp0, qp1])
        init_qvel = np.array([qv0, qv1])
        
        if verbose > 0:
            total_in_partition = len(all_combinations)
            print(f"[{traj_count}/{total_in_partition}] Planning trajectory with qpos={init_qpos}, qvel={init_qvel}")
        
        # Plan trajectory
        planner.plan(keyframe="home", init_qpos=init_qpos, init_qvel=init_qvel)
        
        # Get trajectories
        qpos, qvel, ctrl, time = planner.get_trajectories()
        cost_total, cost_terms = planner.get_costs()
        
        # Generate filename and save
        filename = generate_filename(init_qpos, init_qvel, rollout_horizon)
        filepath = output_dir / filename
        
        np.savez_compressed(
            filepath,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            time=time,
            cost_total=cost_total,
            cost_terms=cost_terms,
            init_qpos=init_qpos,
            init_qvel=init_qvel
        )
        
        if verbose > 1:
            print(f"  Saved to: {filepath}")
            print(f"  Final qpos: {qpos[:, -1]}")
            print(f"  Total cost: {cost_total.sum():.2f}")
        
    if verbose > 0:
        print(f"\nCompleted! Generated {traj_count} trajectories in {output_dir}")


def main():
    """
    Main function with default parameters.
    Modify these parameters as needed or use command-line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MPC trajectories with various initial conditions")
    parser.add_argument("--interval", type=float, default=0.10, 
                        help="Step size for scanning state space (default: 0.10)")
    parser.add_argument("--rollout-horizon", type=int, default=10000,
                        help="Trajectory length (default: 10000)")
    parser.add_argument("--opt-steps", type=int, default=10,
                        help="MPC optimization steps (default: 10)")
    parser.add_argument("--partition-idx", type=int, default=None,
                        help="Partition index (0-based) for parallel execution")
    parser.add_argument("--num-partitions", type=int, default=None,
                        help="Total number of partitions for parallel execution")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ../../data/)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level (0=quiet, 1=progress, 2=detailed)")
    
    args = parser.parse_args()
    
    # Validate partition arguments
    if (args.partition_idx is not None) != (args.num_partitions is not None):
        parser.error("--partition-idx and --num-partitions must be used together")
    
    generate_trajectories(
        qpos0_range=(-1.0, 1.0),
        qpos1_range=(np.pi - 1.0, np.pi + 1.0),
        qvel0_range=(-1.0, 1.0),
        qvel1_range=(-1.0, 1.0),
        interval=args.interval,
        rollout_horizon=args.rollout_horizon,
        opt_steps=args.opt_steps,
        output_dir=args.output_dir,
        verbose=args.verbose,
        partition_idx=args.partition_idx,
        num_partitions=args.num_partitions
    )


if __name__ == "__main__":
    """
    How to Run:

    Running in Parallel - Examples Split into 4 partitions (4 terminals)
    Terminal 1:
    python mpc_rl/planner/gen_traj_data.py --interval 0.5 --partition-idx 0 --num-partitions 4

    Terminal 2:
    python mpc_rl/planner/gen_traj_data.py --interval 0.5 --partition-idx 0 --num-partitions 4
    
    Terminal 3:
    python mpc_rl/planner/gen_traj_data.py --interval 0.5 --partition-idx 0 --num-partitions 4

    Terminal 4:
    python mpc_rl/planner/gen_traj_data.py --interval 0.5 --partition-idx 0 --num-partitions 4

    Each terminal will process ~156 trajectories (625 ÷ 4) in parallel.

    NOTE: 
    """
    main()