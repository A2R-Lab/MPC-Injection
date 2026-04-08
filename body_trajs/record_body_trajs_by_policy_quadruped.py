#!/usr/bin/env python3
"""Record quadruped policy rollouts for later Viser visualization.

This mirrors the walker trajectory recording flow, but uses the quadruped RL
environment directly so we can capture the same rollout tensors used by the
MPC-injection pipeline:

- full MuJoCo state (`qpos`, `qvel`)
- policy actions
- PD joint targets
- applied torques at every simulation substep
- velocity commands
- observations / next observations

The replay environment always disables domain randomization so the saved
trajectories come from the nominal training plant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3 import TD3 as SB3_TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path so local imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

import mpc_rl.envs  # noqa: F401  Registers QuadrupedVelocityTracking-v0.
from mpc_rl.asym_policies import AsymmetricSACPolicy, AsymmetricTD3Policy  # noqa: F401
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.sac_mpc.sb3_sac_mpc import SB3_SAC_MPC
from mpc_rl.td3_mpc.sb3_td3_mpc import SB3_TD3_MPC


DEFAULT_RUN_ROOT = Path("logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd")
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "model_traj_data_quadruped"
DEFAULT_START_CHECKPOINT = 100_000
DEFAULT_END_CHECKPOINT = 1_000_000
DEFAULT_CHECKPOINT_STEP = 100_000
DEFAULT_MAX_CONTROL_STEPS = 1_000
DEFAULT_RANDOM_SEED = 100
DEFAULT_ROBOT = "go2"
DEFAULT_COMMAND_VX = 0.5
DEFAULT_COMMAND_VY = 0.0
DEFAULT_COMMAND_WZ = 0.0
QPOS_ROOT_NAMES = (
    "base_x",
    "base_y",
    "base_z",
    "base_qw",
    "base_qx",
    "base_qy",
    "base_qz",
)


def load_config(run_dir: Path) -> dict:
    """Load a saved training config.json."""
    config_path = run_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_dir(run_path: Path) -> Path:
    """Resolve a concrete run directory.

    The default path points at a directory containing multiple runs, so if the
    provided path is not itself a run directory we pick the lexicographically
    latest child that contains a config.
    """
    run_path = run_path.expanduser().resolve()
    if (run_path / "config.json").exists():
        return run_path

    candidates = sorted(
        child for child in run_path.iterdir()
        if child.is_dir() and (child / "config.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directory with config.json found under: {run_path}"
        )
    return candidates[-1]


def detect_algorithm(run_dir: Path, config: dict) -> str:
    """Infer the algorithm name used for the run."""
    if "algorithm" in config:
        return str(config["algorithm"])

    run_name = run_dir.name.upper()
    if "SAC-MPC" in run_name:
        return "SAC-MPC"
    if "TD3-MPC" in run_name:
        return "TD3-MPC"
    if "SAC" in run_name:
        return "SAC"
    if "TD3" in run_name:
        return "TD3"
    raise ValueError(f"Could not detect algorithm for run: {run_dir}")


def build_quadruped_env(*, robot: str, simple_reward: bool):
    """Create the nominal non-domain-randomized quadruped env."""
    return gym.make(
        "QuadrupedVelocityTracking-v0",
        robot=robot,
        render_mode=None,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
        simple_reward=simple_reward,
    )


def make_model_env(*, robot: str, simple_reward: bool):
    """Create the VecEnv used only for model loading / observation normalization."""
    return DummyVecEnv(
        [lambda: build_quadruped_env(robot=robot, simple_reward=simple_reward)]
    )


def load_model_and_vecnormalize(
    run_dir: Path,
    checkpoint_step: int,
    *,
    algorithm: str,
    robot: str,
):
    """Load a saved quadruped policy checkpoint plus VecNormalize stats."""
    simple_reward = algorithm in {"SAC-MPC", "TD3-MPC"}
    vec_env = make_model_env(robot=robot, simple_reward=simple_reward)

    vecnormalize_path = run_dir / "checkpoints" / f"model_vecnormalize_{checkpoint_step}_steps.pkl"
    model_path = run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip"

    if not vecnormalize_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnormalize_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    algo_class = {
        "SAC": SB3_SAC,
        "TD3": SB3_TD3,
        "SAC-MPC": SB3_SAC_MPC,
        "TD3-MPC": SB3_TD3_MPC,
    }[algorithm]
    model = algo_class.load(model_path, env=vec_env)
    return model, vec_env


def normalize_obs_for_model(vec_env: VecNormalize, obs_raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize a single raw Dict observation for SB3 policy inference."""
    obs_batched = {key: value[None, ...] for key, value in obs_raw.items()}
    return vec_env.normalize_obs(obs_batched)


def get_actuated_joint_names(env) -> list[str]:
    """Return actuated joint names in policy / URDF order."""
    return [
        mujoco.mj_id2name(env.mjModel, mujoco.mjtObj.mjOBJ_JOINT, env.mjModel.actuator_trnid[i, 0])
        for i in range(env.num_joints)
    ]


def get_all_body_names(env) -> list[str]:
    """Return all non-world body names."""
    body_names: list[str] = []
    for body_id in range(1, env.mjModel.nbody):
        name = mujoco.mj_id2name(env.mjModel, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_names.append(name if name is not None else f"body_{body_id}")
    return body_names


def compute_robot_com(env) -> np.ndarray:
    """Return the subtree COM for the whole robot rooted at the base body."""
    base_body_id = env._base_body_id
    return np.array(env.mjData.subtree_com[base_body_id], copy=True)


def snapshot_control_state(env, foot_names: list[str], body_names: list[str]) -> dict[str, np.ndarray]:
    """Capture the current control-rate state used for visualization."""
    foot_positions = env._get_foot_positions()
    foot_contacts = env._get_foot_contacts()
    body_positions = np.stack(
        [env.mjData.xpos[mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_BODY, name)] for name in body_names],
        axis=0,
    )
    body_orientations = np.stack(
        [env.mjData.xquat[mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_BODY, name)] for name in body_names],
        axis=0,
    )
    return {
        "qpos": np.array(env.mjData.qpos, copy=True),
        "qvel": np.array(env.mjData.qvel, copy=True),
        "base_pos": np.array(env.mjData.qpos[:3], copy=True),
        "base_quat": np.array(env.mjData.qpos[3:7], copy=True),
        "com_pos": compute_robot_com(env),
        "foot_positions": np.array(foot_positions, copy=True),
        "foot_contacts": np.array(foot_contacts, copy=True),
        "feet_air_time": np.array(env._feet_air_time, copy=True),
        "feet_contact_time": np.array(env._feet_contact_time, copy=True),
        "swing_peak": np.array(env._swing_peak, copy=True),
        "body_positions": np.array(body_positions, copy=True),
        "body_orientations": np.array(body_orientations, copy=True),
        "foot_names": np.array(foot_names, dtype="<U16"),
        "body_names": np.array(body_names, dtype="<U64"),
    }


def rollout_one_policy_episode(
    model,
    vec_env: VecNormalize,
    *,
    seed: int,
    max_control_steps: int,
    robot: str,
    algorithm: str,
    command_vx: float,
    command_vy: float,
    command_wz: float,
) -> dict:
    """Run one rollout with exact manual stepping and rich trajectory capture."""
    simple_reward = algorithm in {"SAC-MPC", "TD3-MPC"}
    rollout_env = build_quadruped_env(robot=robot, simple_reward=simple_reward).unwrapped
    rollout_env.reset(seed=seed)
    rollout_env.set_commands(vx=command_vx, vy=command_vy, wz=command_wz)
    rollout_env.assert_generation_contact_friction_matches()

    actuated_joint_names = get_actuated_joint_names(rollout_env)
    qpos_names = list(QPOS_ROOT_NAMES) + actuated_joint_names
    qvel_names = [
        "base_vx",
        "base_vy",
        "base_vz",
        "base_wx",
        "base_wy",
        "base_wz",
    ] + actuated_joint_names
    foot_names = list(rollout_env._foot_geom_ids.keys())
    body_names = get_all_body_names(rollout_env)

    obs_raw = rollout_env._get_obs()

    qpos_sim = [np.array(rollout_env.mjData.qpos, copy=True)]
    qvel_sim = [np.array(rollout_env.mjData.qvel, copy=True)]
    tau_applied_sim: list[np.ndarray] = []
    q_des_sim: list[np.ndarray] = []
    commands_sim: list[np.ndarray] = []

    actions_ctrl: list[np.ndarray] = []
    q_targets_ctrl: list[np.ndarray] = []
    rewards_ctrl: list[float] = []
    commands_ctrl: list[np.ndarray] = []
    next_commands_ctrl: list[np.ndarray] = []
    terminated_ctrl: list[bool] = []
    base_lin_vel_body_ctrl: list[np.ndarray] = []
    base_ang_vel_body_ctrl: list[np.ndarray] = []
    base_height_ctrl: list[float] = []
    normalized_policy_obs_ctrl: list[np.ndarray] = []
    normalized_privileged_obs_ctrl: list[np.ndarray] = []
    policy_obs_ctrl: list[np.ndarray] = []
    privileged_obs_ctrl: list[np.ndarray] = []
    next_policy_obs_ctrl: list[np.ndarray] = []
    next_privileged_obs_ctrl: list[np.ndarray] = []

    control_snapshots = [snapshot_control_state(rollout_env, foot_names, body_names)]

    terminated = False
    control_steps = 0

    while control_steps < max_control_steps and not terminated:
        obs_norm = normalize_obs_for_model(vec_env, obs_raw)
        action, _ = model.predict(obs_norm, deterministic=True)
        action = np.asarray(action[0], dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        commands_before = np.array(rollout_env._commands, copy=True)
        q_target = rollout_env.default_joint_pos + rollout_env.action_scale * action

        policy_obs_ctrl.append(np.array(obs_raw["policy"], copy=True))
        privileged_obs_ctrl.append(np.array(obs_raw["privileged"], copy=True))
        normalized_policy_obs_ctrl.append(np.array(obs_norm["policy"][0], copy=True))
        normalized_privileged_obs_ctrl.append(np.array(obs_norm["privileged"][0], copy=True))
        actions_ctrl.append(np.array(action, copy=True))
        q_targets_ctrl.append(np.array(q_target, copy=True))
        commands_ctrl.append(commands_before)

        rollout_env._prev_last_action = rollout_env._last_action.copy()
        rollout_env._last_action = action.copy()

        for _ in range(rollout_env.decimation):
            q_current = rollout_env.mjData.qpos[7:].copy()
            dq_current = rollout_env.mjData.qvel[6:].copy()
            torques = rollout_env.kp * (q_target - q_current) + rollout_env.kd * (0.0 - dq_current)
            torques = np.clip(
                torques,
                rollout_env.torque_limits[:, 0],
                rollout_env.torque_limits[:, 1],
            )
            rollout_env._applied_torques = torques
            rollout_env.mjData.ctrl[:] = torques
            mujoco.mj_step(rollout_env.mjModel, rollout_env.mjData)

            tau_applied_sim.append(np.array(torques, copy=True))
            q_des_sim.append(np.array(q_target, copy=True))
            commands_sim.append(commands_before.copy())
            qpos_sim.append(np.array(rollout_env.mjData.qpos, copy=True))
            qvel_sim.append(np.array(rollout_env.mjData.qvel, copy=True))

        rollout_env._step_count += 1
        rollout_env._steps_since_command_resample += 1
        rollout_env._maybe_push_robot()
        rollout_env._update_feet_air_time()

        joint_vel_current = rollout_env.mjData.qvel[6:].copy()
        rollout_env._joint_acc = (
            joint_vel_current - rollout_env._last_joint_vel
        ) / rollout_env.control_dt
        rollout_env._last_joint_vel = joint_vel_current

        if (
            not rollout_env._fixed_commands
            and rollout_env._steps_since_command_resample >= rollout_env.command_resample_interval
        ):
            rollout_env._sample_commands()

        next_obs_raw = rollout_env._get_obs()
        terminated = rollout_env._check_termination()
        reward = float(rollout_env._compute_reward(action, terminated))
        info = rollout_env._get_info()
        rollout_env._swing_peak *= ~rollout_env._current_contacts

        rewards_ctrl.append(reward)
        next_commands_ctrl.append(np.array(info["commands"], copy=True))
        terminated_ctrl.append(bool(terminated))
        base_lin_vel_body_ctrl.append(np.array(info["base_lin_vel_body"], copy=True))
        base_ang_vel_body_ctrl.append(np.array(info["base_ang_vel_body"], copy=True))
        base_height_ctrl.append(float(info["base_height"]))
        next_policy_obs_ctrl.append(np.array(next_obs_raw["policy"], copy=True))
        next_privileged_obs_ctrl.append(np.array(next_obs_raw["privileged"], copy=True))
        control_snapshots.append(snapshot_control_state(rollout_env, foot_names, body_names))

        obs_raw = next_obs_raw
        control_steps += 1

    rollout_env.close()

    qpos_sim_arr = np.stack(qpos_sim, axis=0)
    qvel_sim_arr = np.stack(qvel_sim, axis=0)
    tau_applied_arr = np.stack(tau_applied_sim, axis=0) if tau_applied_sim else np.zeros((0, len(actuated_joint_names)))
    q_des_arr = np.stack(q_des_sim, axis=0) if q_des_sim else np.zeros((0, len(actuated_joint_names)))
    commands_sim_arr = np.stack(commands_sim, axis=0) if commands_sim else np.zeros((0, 3))

    qpos_ctrl_arr = np.stack([snap["qpos"] for snap in control_snapshots], axis=0)
    qvel_ctrl_arr = np.stack([snap["qvel"] for snap in control_snapshots], axis=0)
    base_pos_arr = np.stack([snap["base_pos"] for snap in control_snapshots], axis=0)
    base_quat_arr = np.stack([snap["base_quat"] for snap in control_snapshots], axis=0)
    com_pos_arr = np.stack([snap["com_pos"] for snap in control_snapshots], axis=0)
    foot_positions_arr = np.stack([snap["foot_positions"] for snap in control_snapshots], axis=0)
    foot_contacts_arr = np.stack([snap["foot_contacts"] for snap in control_snapshots], axis=0)
    feet_air_time_arr = np.stack([snap["feet_air_time"] for snap in control_snapshots], axis=0)
    feet_contact_time_arr = np.stack([snap["feet_contact_time"] for snap in control_snapshots], axis=0)
    swing_peak_arr = np.stack([snap["swing_peak"] for snap in control_snapshots], axis=0)
    body_positions_arr = np.stack([snap["body_positions"] for snap in control_snapshots], axis=0)
    body_orientations_arr = np.stack([snap["body_orientations"] for snap in control_snapshots], axis=0)

    return {
        "timesteps": qpos_ctrl_arr.shape[0],
        "num_control_steps": control_steps,
        "num_sim_steps": tau_applied_arr.shape[0],
        "done": bool(terminated),
        "fell": bool(terminated),
        "seed": seed,
        "robot": robot,
        "algorithm": algorithm,
        "fixed_command": np.array([command_vx, command_vy, command_wz], dtype=np.float64),
        "sim_dt": float(rollout_env.sim_dt),
        "control_dt": float(rollout_env.control_dt),
        "frame_dt": float(rollout_env.control_dt),
        "decimation": int(rollout_env.decimation),
        "action_scale": float(rollout_env.action_scale),
        "default_joint_pos": np.array(rollout_env.default_joint_pos, copy=True),
        "actuated_joint_names": np.array(actuated_joint_names, dtype="<U32"),
        "qpos_names": np.array(qpos_names, dtype="<U32"),
        "qvel_names": np.array(qvel_names, dtype="<U32"),
        "foot_names": np.array(foot_names, dtype="<U16"),
        "body_names": np.array(body_names, dtype="<U64"),
        "qpos": qpos_sim_arr.T,
        "qvel": qvel_sim_arr.T,
        "qpos_ctrl": qpos_ctrl_arr,
        "qvel_ctrl": qvel_ctrl_arr,
        "joint_positions_ctrl": qpos_ctrl_arr[:, 7:],
        "joint_velocities_ctrl": qvel_ctrl_arr[:, 6:],
        "actions": np.stack(actions_ctrl, axis=0) if actions_ctrl else np.zeros((0, len(actuated_joint_names))),
        "q_targets_ctrl": np.stack(q_targets_ctrl, axis=0) if q_targets_ctrl else np.zeros((0, len(actuated_joint_names))),
        "tau_applied": tau_applied_arr.T,
        "q_des": q_des_arr.T,
        "commands": commands_sim_arr.T,
        "commands_ctrl": np.stack(commands_ctrl, axis=0) if commands_ctrl else np.zeros((0, 3)),
        "next_commands_ctrl": np.stack(next_commands_ctrl, axis=0) if next_commands_ctrl else np.zeros((0, 3)),
        "rewards": np.array(rewards_ctrl, dtype=np.float64),
        "terminated_ctrl": np.array(terminated_ctrl, dtype=bool),
        "policy_obs": np.stack(policy_obs_ctrl, axis=0) if policy_obs_ctrl else np.zeros((0, rollout_env.policy_obs_dim)),
        "privileged_obs": np.stack(privileged_obs_ctrl, axis=0) if privileged_obs_ctrl else np.zeros((0, rollout_env.privileged_obs_dim)),
        "next_policy_obs": np.stack(next_policy_obs_ctrl, axis=0) if next_policy_obs_ctrl else np.zeros((0, rollout_env.policy_obs_dim)),
        "next_privileged_obs": np.stack(next_privileged_obs_ctrl, axis=0) if next_privileged_obs_ctrl else np.zeros((0, rollout_env.privileged_obs_dim)),
        "normalized_policy_obs": np.stack(normalized_policy_obs_ctrl, axis=0) if normalized_policy_obs_ctrl else np.zeros((0, rollout_env.policy_obs_dim)),
        "normalized_privileged_obs": np.stack(normalized_privileged_obs_ctrl, axis=0) if normalized_privileged_obs_ctrl else np.zeros((0, rollout_env.privileged_obs_dim)),
        "base_lin_vel_body_ctrl": np.stack(base_lin_vel_body_ctrl, axis=0) if base_lin_vel_body_ctrl else np.zeros((0, 3)),
        "base_ang_vel_body_ctrl": np.stack(base_ang_vel_body_ctrl, axis=0) if base_ang_vel_body_ctrl else np.zeros((0, 3)),
        "base_height_ctrl": np.array(base_height_ctrl, dtype=np.float64),
        "base_positions": base_pos_arr,
        "base_quaternions": base_quat_arr,
        "com_positions": com_pos_arr,
        "foot_positions": foot_positions_arr,
        "foot_contacts": foot_contacts_arr.astype(bool),
        "feet_air_time": feet_air_time_arr,
        "feet_contact_time": feet_contact_time_arr,
        "swing_peak": swing_peak_arr,
        "body_positions": body_positions_arr,
        "body_orientations": body_orientations_arr,
        "total_reward": float(np.sum(rewards_ctrl)) if rewards_ctrl else 0.0,
        "mean_reward": float(np.mean(rewards_ctrl)) if rewards_ctrl else 0.0,
    }


def save_trajectory_data(
    trajectory_data: dict,
    *,
    output_dir: Path,
    checkpoint_step: int,
    run_dir: Path,
) -> Path:
    """Save one trajectory bundle to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"trajectories_step_{checkpoint_step}.npz"
    np.savez_compressed(
        output_file,
        checkpoint_step=checkpoint_step,
        run_name=run_dir.name,
        run_dir=str(run_dir),
        **trajectory_data,
    )
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record nominal quadruped policy trajectories for Viser playback."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=(
            "Specific run directory, or a directory containing multiple runs. "
            f"Default: {DEFAULT_RUN_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--start-checkpoint",
        type=int,
        default=DEFAULT_START_CHECKPOINT,
        help=f"First checkpoint step to process. Default: {DEFAULT_START_CHECKPOINT}",
    )
    parser.add_argument(
        "--end-checkpoint",
        type=int,
        default=DEFAULT_END_CHECKPOINT,
        help=f"Last checkpoint step to process. Default: {DEFAULT_END_CHECKPOINT}",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=DEFAULT_CHECKPOINT_STEP,
        help=f"Checkpoint interval. Default: {DEFAULT_CHECKPOINT_STEP}",
    )
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=DEFAULT_MAX_CONTROL_STEPS,
        help=f"Max control steps per rollout. Default: {DEFAULT_MAX_CONTROL_STEPS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Rollout seed. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=DEFAULT_ROBOT,
        help=f"Quadruped robot model. Default: {DEFAULT_ROBOT}",
    )
    parser.add_argument(
        "--command-vx",
        type=float,
        default=DEFAULT_COMMAND_VX,
        help=f"Fixed commanded linear x velocity in m/s. Default: {DEFAULT_COMMAND_VX}",
    )
    parser.add_argument(
        "--command-vy",
        type=float,
        default=DEFAULT_COMMAND_VY,
        help=f"Fixed commanded linear y velocity in m/s. Default: {DEFAULT_COMMAND_VY}",
    )
    parser.add_argument(
        "--command-wz",
        type=float,
        default=DEFAULT_COMMAND_WZ,
        help=f"Fixed commanded yaw rate in rad/s. Default: {DEFAULT_COMMAND_WZ}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    config = load_config(run_dir)
    algorithm = detect_algorithm(run_dir, config)
    checkpoints = list(
        range(args.start_checkpoint, args.end_checkpoint + 1, args.checkpoint_step)
    )

    print("=" * 80)
    print("Recording Quadruped Policy Trajectories")
    print("=" * 80)
    print(f"Resolved run dir: {run_dir}")
    print(
        "Fixed commands: "
        f"vx={args.command_vx:.3f} m/s, vy={args.command_vy:.3f} m/s, wz={args.command_wz:.3f} rad/s"
    )
    print(f"Algorithm: {algorithm}")
    print(f"Robot: {args.robot}")
    print(f"Nominal domain randomization: disabled")
    print(f"Checkpoints: {checkpoints[0]} -> {checkpoints[-1]} (step {args.checkpoint_step})")
    print(f"Max control steps: {args.max_control_steps}")
    print(f"Seed: {args.seed}")

    output_subdir = args.output_dir / run_dir.name
    print(f"Output directory: {output_subdir}")

    for idx, checkpoint_step in enumerate(checkpoints, start=1):
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(checkpoints)}] Checkpoint {checkpoint_step}")
        print("-" * 80)
        try:
            model, vec_env = load_model_and_vecnormalize(
                run_dir,
                checkpoint_step,
                algorithm=algorithm,
                robot=args.robot,
            )

            trajectory_data = rollout_one_policy_episode(
                model,
                vec_env,
                seed=args.seed,
                max_control_steps=args.max_control_steps,
                robot=args.robot,
                algorithm=algorithm,
                command_vx=args.command_vx,
                command_vy=args.command_vy,
                command_wz=args.command_wz,
            )
            saved_path = save_trajectory_data(
                trajectory_data,
                output_dir=output_subdir,
                checkpoint_step=checkpoint_step,
                run_dir=run_dir,
            )

            print(f"Frames saved: {trajectory_data['timesteps']}")
            print(f"Control steps: {trajectory_data['num_control_steps']}")
            print(f"Sim steps: {trajectory_data['num_sim_steps']}")
            print(f"Total reward: {trajectory_data['total_reward']:.3f}")
            print(f"Terminated early: {trajectory_data['done']}")
            print(f"Saved: {saved_path}")
            vec_env.close()
        except FileNotFoundError as exc:
            print(f"Skipping checkpoint {checkpoint_step}: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - debugging path
            print(f"Failed to record checkpoint {checkpoint_step}: {exc}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()
