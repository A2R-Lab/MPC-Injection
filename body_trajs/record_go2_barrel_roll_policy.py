#!/usr/bin/env python3
"""Record one selected Go2 barrel-roll policy rollout for Viser playback.

The recorder executes the normal Gymnasium environment step so the saved
rollout includes the same action clipping, 5 Hz target filter, PD control, and
termination behavior used during policy evaluation.  It records control-rate
state and visualization metadata; it does not reconstruct physics-substep
states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from body_trajs.record_body_trajs_by_policy_quadruped import (  # noqa: E402
    QPOS_ROOT_NAMES,
    get_actuated_joint_names,
    get_all_body_names,
    normalize_obs_for_model,
    resolve_run_dir,
    snapshot_control_state,
)
from mpc_rl.envs.barrel_roll_common import CONTROL_STEPS, SCHEMA_VERSION  # noqa: E402
from mpc_rl.planner.barrel_roll_dataset import sha256_file  # noqa: E402
from mpc_rl.train import (  # noqa: E402
    load_saved_model_for_video_eval,
    make_quadruped_env,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "model_traj_data_quadruped"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def resolve_policy_artifacts(
    run_dir: Path,
    *,
    checkpoint_step: int | None,
    final_model: bool,
) -> dict[str, Any]:
    """Resolve one model and its matching observation-normalization state."""
    if checkpoint_step is not None:
        model = run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip"
        vecnormalize = (
            run_dir
            / "checkpoints"
            / f"model_vecnormalize_{checkpoint_step}_steps.pkl"
        )
        source = "checkpoint"
        selected_step = checkpoint_step
    elif final_model:
        model = run_dir / "final_model.zip"
        vecnormalize = run_dir / "vec_normalize.pkl"
        source = "final_model"
        selected_step = None
    else:
        best_dir = run_dir / "best_model"
        selection_path = best_dir / "selection.json"
        selection = _read_json(selection_path)
        model = best_dir / "best_model.zip"
        vecnormalize = best_dir / "vec_normalize.pkl"
        source = "best_model"
        selected_step = int(selection["timesteps"])

    missing = [str(path) for path in (model, vecnormalize) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing policy artifacts: " + ", ".join(missing))
    return {
        "source": source,
        "selected_step": selected_step,
        "model": model,
        "vecnormalize": vecnormalize,
    }


def _stack(values: list[np.ndarray], width: int) -> np.ndarray:
    if not values:
        return np.zeros((0, width), dtype=np.float64)
    return np.stack(values, axis=0)


def record_rollout(
    *,
    model: Any,
    model_env: Any,
    seed: int,
) -> dict[str, Any]:
    """Execute and capture one deterministic nominal barrel-roll episode."""
    gym_env = make_quadruped_env(
        robot="go2",
        task="barrel_roll",
        domain_rand_cfg=None,
        simple_reward=False,
        use_go2_sysid=True,
    )
    env = gym_env.unwrapped
    try:
        obs, reset_info = env.reset(seed=seed)
        env.assert_generation_contact_friction_matches()

        actuated_joint_names = get_actuated_joint_names(env)
        qpos_names = list(QPOS_ROOT_NAMES) + actuated_joint_names
        qvel_names = [
            "base_vx",
            "base_vy",
            "base_vz",
            "base_wx",
            "base_wy",
            "base_wz",
        ] + actuated_joint_names
        foot_names = list(env._foot_geom_ids.keys())
        body_names = get_all_body_names(env)

        snapshots = [snapshot_control_state(env, foot_names, body_names)]
        policy_obs: list[np.ndarray] = []
        privileged_obs: list[np.ndarray] = []
        normalized_policy_obs: list[np.ndarray] = []
        normalized_privileged_obs: list[np.ndarray] = []
        next_policy_obs: list[np.ndarray] = []
        next_privileged_obs: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        raw_q_targets: list[np.ndarray] = []
        filtered_q_targets: list[np.ndarray] = []
        applied_torques: list[np.ndarray] = []
        rewards: list[float] = []
        terminated_ctrl: list[bool] = []
        truncated_ctrl: list[bool] = []
        roll_progress: list[float] = []
        desired_roll: list[float] = []
        body_up_tilt: list[float] = []
        reward_components: dict[str, list[float]] = {
            "roll_tracking": [],
            "rate_tracking": [],
            "action_change": [],
            "terminal_outcome": [],
        }

        terminated = False
        truncated = False
        terminal_info: dict[str, Any] = dict(reset_info)
        while not (terminated or truncated) and len(actions) < CONTROL_STEPS:
            normalized = normalize_obs_for_model(model_env, obs)
            action, _ = model.predict(normalized, deterministic=True)
            action = np.clip(np.asarray(action[0], dtype=np.float64), -1.0, 1.0)

            policy_obs.append(np.array(obs["policy"], copy=True))
            privileged_obs.append(np.array(obs["privileged"], copy=True))
            normalized_policy_obs.append(
                np.array(normalized["policy"][0], copy=True)
            )
            normalized_privileged_obs.append(
                np.array(normalized["privileged"][0], copy=True)
            )
            actions.append(action.copy())

            next_obs, reward, terminated, truncated, info = env.step(action)
            terminal_info = dict(info)
            next_policy_obs.append(np.array(next_obs["policy"], copy=True))
            next_privileged_obs.append(
                np.array(next_obs["privileged"], copy=True)
            )
            raw_q_targets.append(np.array(info["raw_q_target"], copy=True))
            filtered_q_targets.append(
                np.array(info["filtered_q_target"], copy=True)
            )
            applied_torques.append(np.array(info["applied_torques"], copy=True))
            rewards.append(float(reward))
            terminated_ctrl.append(bool(terminated))
            truncated_ctrl.append(bool(truncated))
            roll_progress.append(float(info["roll_progress"]))
            desired_roll.append(float(info["desired_roll"]))
            body_up_tilt.append(float(info["body_up_tilt"]))
            step_reward_components = info.get("reward_components", {})
            for name in reward_components:
                value = step_reward_components.get(name, np.nan)
                reward_components[name].append(float(value))
            snapshots.append(snapshot_control_state(env, foot_names, body_names))
            obs = next_obs

        qpos_ctrl = np.stack([item["qpos"] for item in snapshots], axis=0)
        qvel_ctrl = np.stack([item["qvel"] for item in snapshots], axis=0)
        control_steps = len(actions)
        return {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "task_id": np.asarray("go2_barrel_roll"),
            "seed": np.asarray(seed),
            "timesteps": np.asarray(len(snapshots)),
            "num_control_steps": np.asarray(control_steps),
            "num_sim_steps": np.asarray(control_steps * env.decimation),
            "done": np.asarray(terminated or truncated),
            "terminated": np.asarray(terminated),
            "truncated": np.asarray(truncated),
            "terminated_early": np.asarray(control_steps < CONTROL_STEPS),
            "is_success": np.asarray(bool(terminal_info.get("is_success", False))),
            "failure_reason": np.asarray(
                str(terminal_info.get("failure_reason") or "")
            ),
            "robot": np.asarray("go2"),
            "algorithm": np.asarray("SAC-MPC"),
            "sim_dt": np.asarray(float(env.sim_dt)),
            "control_dt": np.asarray(float(env.control_dt)),
            "frame_dt": np.asarray(float(env.control_dt)),
            "decimation": np.asarray(int(env.decimation)),
            "action_scale": np.asarray(float(env.action_scale)),
            "action_lpf_cutoff_hz": np.asarray(float(env.action_lpf_cutoff_hz)),
            "action_lpf_alpha": np.asarray(float(env.action_lpf_alpha)),
            "default_joint_pos": np.array(env.default_joint_pos, copy=True),
            "actuated_joint_names": np.asarray(actuated_joint_names, dtype="<U32"),
            "qpos_names": np.asarray(qpos_names, dtype="<U32"),
            "qvel_names": np.asarray(qvel_names, dtype="<U32"),
            "foot_names": np.asarray(foot_names, dtype="<U16"),
            "body_names": np.asarray(body_names, dtype="<U64"),
            "qpos_ctrl": qpos_ctrl,
            "qvel_ctrl": qvel_ctrl,
            "joint_positions_ctrl": qpos_ctrl[:, 7:],
            "joint_velocities_ctrl": qvel_ctrl[:, 6:],
            "actions": _stack(actions, env.num_joints),
            "raw_q_targets_ctrl": _stack(raw_q_targets, env.num_joints),
            "filtered_q_targets_ctrl": _stack(
                filtered_q_targets, env.num_joints
            ),
            "applied_torques_ctrl": _stack(applied_torques, env.num_joints),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "terminated_ctrl": np.asarray(terminated_ctrl, dtype=bool),
            "truncated_ctrl": np.asarray(truncated_ctrl, dtype=bool),
            "policy_obs": _stack(policy_obs, env.policy_obs_dim),
            "privileged_obs": _stack(privileged_obs, env.privileged_obs_dim),
            "next_policy_obs": _stack(next_policy_obs, env.policy_obs_dim),
            "next_privileged_obs": _stack(
                next_privileged_obs, env.privileged_obs_dim
            ),
            "normalized_policy_obs": _stack(
                normalized_policy_obs, env.policy_obs_dim
            ),
            "normalized_privileged_obs": _stack(
                normalized_privileged_obs, env.privileged_obs_dim
            ),
            "roll_progress_ctrl": np.asarray(roll_progress, dtype=np.float64),
            "desired_roll_ctrl": np.asarray(desired_roll, dtype=np.float64),
            "body_up_tilt_ctrl": np.asarray(body_up_tilt, dtype=np.float64),
            "reward_roll_tracking": np.asarray(
                reward_components["roll_tracking"], dtype=np.float64
            ),
            "reward_rate_tracking": np.asarray(
                reward_components["rate_tracking"], dtype=np.float64
            ),
            "reward_action_change": np.asarray(
                reward_components["action_change"], dtype=np.float64
            ),
            "reward_terminal_outcome": np.asarray(
                reward_components["terminal_outcome"], dtype=np.float64
            ),
            "base_positions": np.stack(
                [item["base_pos"] for item in snapshots], axis=0
            ),
            "base_quaternions": np.stack(
                [item["base_quat"] for item in snapshots], axis=0
            ),
            "com_positions": np.stack(
                [item["com_pos"] for item in snapshots], axis=0
            ),
            "foot_positions": np.stack(
                [item["foot_positions"] for item in snapshots], axis=0
            ),
            "foot_contacts": np.stack(
                [item["foot_contacts"] for item in snapshots], axis=0
            ).astype(bool),
            "feet_air_time": np.stack(
                [item["feet_air_time"] for item in snapshots], axis=0
            ),
            "feet_contact_time": np.stack(
                [item["feet_contact_time"] for item in snapshots], axis=0
            ),
            "swing_peak": np.stack(
                [item["swing_peak"] for item in snapshots], axis=0
            ),
            "body_positions": np.stack(
                [item["body_positions"] for item in snapshots], axis=0
            ),
            "body_orientations": np.stack(
                [item["body_orientations"] for item in snapshots], axis=0
            ),
            "total_reward": np.asarray(float(np.sum(rewards))),
            "mean_reward": np.asarray(float(np.mean(rewards))),
            "terminal_roll_progress": np.asarray(
                float(terminal_info.get("roll_progress", np.nan))
            ),
            "terminal_base_height": np.asarray(
                float(terminal_info.get("base_height", np.nan))
            ),
            "terminal_body_up_tilt": np.asarray(
                float(terminal_info.get("body_up_tilt", np.nan))
            ),
        }
    finally:
        gym_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Concrete training run, or a seed directory containing one run.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint-step",
        type=int,
        help="Record a checkpoint pair instead of the selected best model.",
    )
    source.add_argument(
        "--final-model",
        action="store_true",
        help="Record final_model.zip instead of the selected best model.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Rollout reset seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Fail without writing if the rollout does not satisfy the classifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    config = _read_json(run_dir / "config.json")
    if config.get("task") != "barrel_roll" or config.get("algorithm") != "SAC-MPC":
        raise ValueError(f"not a Go2 SAC-MPC barrel-roll run: {run_dir}")
    if config.get("use_go2_sysid") is not True:
        raise ValueError("barrel-roll replay requires use_go2_sysid=True")

    artifacts = resolve_policy_artifacts(
        run_dir,
        checkpoint_step=args.checkpoint_step,
        final_model=args.final_model,
    )
    model, model_env = load_saved_model_for_video_eval(
        algorithm="SAC-MPC",
        model_path=artifacts["model"],
        vecnormalize_path=artifacts["vecnormalize"],
        domain="quadruped",
        task="barrel_roll",
        is_quadruped=True,
        robot="go2",
        simple_reward=False,
        use_go2_sysid=True,
    )
    try:
        trajectory = record_rollout(model=model, model_env=model_env, seed=args.seed)
    finally:
        model_env.close()

    if args.require_success and not bool(trajectory["is_success"]):
        reason = str(trajectory["failure_reason"].item() or "unknown")
        raise RuntimeError(f"rollout seed {args.seed} failed: {reason}")

    training_seed = int(config["seed"])
    selected_step = artifacts["selected_step"]
    step_label = "final" if selected_step is None else f"step{selected_step}"
    output_dir = args.output_dir / run_dir.name
    output_path = output_dir / (
        f"training_seed{training_seed}_{artifacts['source']}_{step_label}"
        f"_rollout_seed{args.seed}.npz"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite trajectory: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        run_dir=np.asarray(str(run_dir)),
        training_seed=np.asarray(training_seed),
        policy_source=np.asarray(artifacts["source"]),
        selected_timesteps=np.asarray(
            -1 if selected_step is None else int(selected_step)
        ),
        model_path=np.asarray(str(artifacts["model"])),
        model_sha256=np.asarray(sha256_file(artifacts["model"])),
        vecnormalize_path=np.asarray(str(artifacts["vecnormalize"])),
        vecnormalize_sha256=np.asarray(sha256_file(artifacts["vecnormalize"])),
        **trajectory,
    )
    result = {
        "output": str(output_path.resolve()),
        "training_seed": training_seed,
        "policy_source": artifacts["source"],
        "selected_timesteps": selected_step,
        "rollout_seed": args.seed,
        "control_steps": int(trajectory["num_control_steps"]),
        "frames": int(trajectory["timesteps"]),
        "success": bool(trajectory["is_success"]),
        "failure_reason": str(trajectory["failure_reason"].item()) or None,
        "total_reward": float(trajectory["total_reward"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
