"""
Generate MPC trajectories for the three-legged cheetah task.

This is a cheetah3-focused variant of `mpc_rl/planner/gen_traj_data.py`. It
uses the same `MPCPlanner` interface as the walker generator, but fixes the
model path and task id to the three-legged cheetah.
"""

import argparse
import contextlib
import io
from pathlib import Path
import sys

import mujoco
import numpy as np

# Add parent directory to path to import from mpc_rl when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mpc_rl.planner.mpc_planner import MPCPlanner
from mpc_rl.envs.cheetah3_common import sample_valid_cheetah3_initial_state


CHEETAH3_MODEL_PATH = Path(__file__).parent.parent / "tasks/cheetah/task.xml"
CHEETAH3_TASK_ID = "Three-Legged Cheetah"
DEFAULT_WEIGHTS = {
    "Speed": 1.0,
    "Height": 10.0,
    "Rotation": 3.0,
    "Control": 0.1,
}
DEFAULT_TASK_PARAMS = {
    "Speed Goal": 3.0,
    "Height Goal": 0.7,
}


def generate_filename(seed, rollout_horizon):
    """Generate a stable filename for one cheetah3 rollout."""
    if seed is None:
        return f"cheetah3_home_rh_{rollout_horizon}.npz"
    return f"cheetah3_seed_{seed:06d}_rh_{rollout_horizon}.npz"


def _agent_timestep(model):
    """Read the MPC agent timestep from the task XML custom numeric section."""
    for i in range(model.nnumeric):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, i)
        if name == "agent_timestep":
            return float(model.numeric_data[model.numeric_adr[i]])
    return float(model.opt.timestep)


def _walker_like_initial_state(model, seed):
    """
    Create an initial state with the same coverage pattern as gen_traj_walker.

    The root slide joints stay at their XML defaults, the unlimited root hinge
    is sampled in [-pi, pi], every limited leg hinge is sampled uniformly inside
    its XML limits, and all velocities start at zero.
    """
    rng = np.random.default_rng(seed)
    return sample_valid_cheetah3_initial_state(model, rng)


def _dm_control_initial_state(model, seed, stabilize_steps=200):
    """
    Create an initial state like dm_control.suite.cheetah.Cheetah.

    DM Control randomizes all limited one-DoF joints uniformly inside their XML
    limits, stabilizes for 200 physics steps, then starts the episode at time 0.
    """
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    if model.nq != model.njnt:
        raise ValueError("cheetah3 initialization assumes one qpos per joint.")

    limited = model.jnt_limited == 1
    lower = model.jnt_range[limited, 0]
    upper = model.jnt_range[limited, 1]
    data.qpos[limited] = rng.uniform(lower, upper)

    mujoco.mj_forward(model, data)
    for _ in range(stabilize_steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    data.time = 0.0
    return data.qpos.copy(), data.qvel.copy()


def _home_initial_state(model):
    """Return the XML default state."""
    return model.qpos0.copy(), np.zeros(model.nv)


def _trajectory_success(model, qpos, qvel, min_forward_distance, min_final_height):
    """Compute simple locomotion sanity metrics for filtering or metadata."""
    rootx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rootx")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    rootx_adr = model.jnt_qposadr[rootx_id]

    forward_distance = float(qpos[rootx_adr, -1] - qpos[rootx_adr, 0])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[:, -1]
    data.qvel[:] = qvel[:, -1]
    mujoco.mj_forward(model, data)
    final_height = float(data.xpos[torso_id, 2])
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    success = (
        finite
        and forward_distance >= min_forward_distance
        and final_height >= min_final_height
    )
    return success, forward_distance, final_height


def gen_traj_cheetah3(
    num_trajectories=1,
    seed_start=0,
    init_mode="walker_like",
    stabilize_steps=200,
    rollout_horizon=1000,
    opt_steps=1,
    weights=None,
    task_params=None,
    output_dir=None,
    verbose=1,
    partition_idx=None,
    num_partitions=None,
    min_forward_distance=0.0,
    min_final_height=0.25,
    require_success=False,
):
    """
    Generate three-legged cheetah MPC rollouts.

    Saved qpos/qvel/control arrays are at the MuJoCo physics timestep. For the
    current cheetah3 XML this is 0.01 s. The MPC agent timestep is 0.02 s, so
    `MPCPlanner` holds each chosen control for two physics steps.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if task_params is None:
        task_params = DEFAULT_TASK_PARAMS

    model = mujoco.MjModel.from_xml_path(str(CHEETAH3_MODEL_PATH))
    physics_timestep = float(model.opt.timestep)
    agent_timestep = _agent_timestep(model)
    steps_per_agent_update = int(round(agent_timestep / physics_timestep))

    if output_dir is None:
        output_dir = (
            Path(__file__).parent.parent.parent
            / f"data/cheetah3_{physics_timestep:.3f}dt".replace(".", "_")
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if init_mode == "home":
        seeds = [None]
    else:
        seeds = list(range(seed_start, seed_start + num_trajectories))

    if partition_idx is not None or num_partitions is not None:
        if partition_idx is None or num_partitions is None:
            raise ValueError("partition_idx and num_partitions must be used together.")
        if partition_idx < 0 or num_partitions <= 0:
            raise ValueError("partition_idx must be >= 0 and num_partitions must be > 0.")
        if partition_idx >= num_partitions:
            raise ValueError("partition_idx must be less than num_partitions.")
        partition_size = len(seeds) // num_partitions
        start_idx = partition_idx * partition_size
        end_idx = len(seeds) if partition_idx == num_partitions - 1 else start_idx + partition_size
        seeds = seeds[start_idx:end_idx]

    if verbose > 0:
        print("Generating cheetah3 trajectories:")
        print(f"  Model path: {CHEETAH3_MODEL_PATH}")
        print(f"  Task id: {CHEETAH3_TASK_ID}")
        print(f"  Physics timestep: {physics_timestep:.6f}s")
        print(f"  Agent timestep: {agent_timestep:.6f}s")
        print(f"  Steps per agent update: {steps_per_agent_update}")
        print(f"  Init mode: {init_mode}")
        print(f"  Rollout horizon: {rollout_horizon}")
        print(f"  Output directory: {output_dir}")
        print(f"  Trajectories in this run: {len(seeds)}")
        print()

    planner = MPCPlanner(
        model_path=CHEETAH3_MODEL_PATH,
        task_id=CHEETAH3_TASK_ID,
        rollout_horizon=rollout_horizon,
        opt_steps=opt_steps,
        weights=weights,
        task_params=task_params,
        init_state_noise_flag=False,
        verbose=0,
    )

    generated = 0
    saved = 0
    try:
        for idx, seed in enumerate(seeds, start=1):
            generated += 1
            if init_mode == "home":
                init_qpos, init_qvel = _home_initial_state(model)
            elif init_mode == "walker_like":
                init_qpos, init_qvel = _walker_like_initial_state(model, seed=seed)
            elif init_mode == "dm_control":
                init_qpos, init_qvel = _dm_control_initial_state(
                    model, seed=seed, stabilize_steps=stabilize_steps
                )
            else:
                raise ValueError(f"Unsupported init_mode: {init_mode}")

            if verbose > 0:
                label = "home" if seed is None else f"seed {seed}"
                print(f"[{idx}/{len(seeds)}] Planning trajectory from {label}")

            # cheetah3 has no keyframes; MPCPlanner still accepts explicit init state.
            with contextlib.redirect_stdout(io.StringIO()):
                planner.plan(keyframe="home", init_qpos=init_qpos, init_qvel=init_qvel)

            qpos, qvel, ctrl, time = planner.get_trajectories()
            cost_total, cost_terms = planner.get_costs()
            success, forward_distance, final_height = _trajectory_success(
                model,
                qpos,
                qvel,
                min_forward_distance=min_forward_distance,
                min_final_height=min_final_height,
            )

            if require_success and not success:
                if verbose > 0:
                    print(
                        "  Skipped unsuccessful trajectory: "
                        f"forward={forward_distance:.3f}, final_height={final_height:.3f}"
                    )
                continue

            filepath = output_dir / generate_filename(seed, rollout_horizon)
            np.savez_compressed(
                filepath,
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                time=time,
                cost_total=cost_total,
                cost_terms=cost_terms,
                init_qpos=init_qpos,
                init_qvel=init_qvel,
                init_seed=-1 if seed is None else seed,
                init_mode=init_mode,
                physics_timestep=physics_timestep,
                agent_timestep=agent_timestep,
                steps_per_agent_update=steps_per_agent_update,
                task_id=CHEETAH3_TASK_ID,
                model_path=str(CHEETAH3_MODEL_PATH),
                weights=weights,
                task_params=task_params,
                success=success,
                forward_distance=forward_distance,
                final_height=final_height,
            )
            saved += 1

            if verbose > 0:
                print(
                    f"  Saved {filepath.name}: success={success}, "
                    f"forward={forward_distance:.3f}, final_height={final_height:.3f}"
                )
    finally:
        planner.agent.close()

    if verbose > 0:
        print(f"\nCompleted: generated {generated}, saved {saved} in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPC trajectories for the three-legged cheetah task"
    )
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--init-mode",
        choices=["walker_like", "dm_control", "home"],
        default="walker_like",
        help="Initial-state distribution (default: walker_like)",
    )
    parser.add_argument(
        "--stabilize-steps",
        type=int,
        default=200,
        help="Zero-control stabilization steps for dm_control initialization",
    )
    parser.add_argument("--rollout-horizon", type=int, default=1000)
    parser.add_argument("--opt-steps", type=int, default=1)
    parser.add_argument("--partition-idx", type=int, default=None)
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-forward-distance", type=float, default=0.0)
    parser.add_argument("--min-final-height", type=float, default=0.25)
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Only save trajectories satisfying the simple locomotion checks",
    )
    args = parser.parse_args()

    gen_traj_cheetah3(
        num_trajectories=args.num_trajectories,
        seed_start=args.seed_start,
        init_mode=args.init_mode,
        stabilize_steps=args.stabilize_steps,
        rollout_horizon=args.rollout_horizon,
        opt_steps=args.opt_steps,
        output_dir=args.output_dir,
        verbose=args.verbose,
        partition_idx=args.partition_idx,
        num_partitions=args.num_partitions,
        min_forward_distance=args.min_forward_distance,
        min_final_height=args.min_final_height,
        require_success=args.require_success,
    )


if __name__ == "__main__":
    main()
