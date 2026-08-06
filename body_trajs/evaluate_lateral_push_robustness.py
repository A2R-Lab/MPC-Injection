#!/usr/bin/env python3
"""Deterministic lateral-push robustness evaluation in native MuJoCo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3 import TD3 as SB3_TD3
from stable_baselines3.common.vec_env import VecNormalize

from record_body_trajs_by_policy_quadruped import (
    build_quadruped_env,
    detect_algorithm,
    load_config,
    make_model_env,
    normalize_obs_for_model,
    resolve_use_go2_sysid,
)

from mpc_rl.sac_mpc.sb3_sac_mpc import SB3_SAC_MPC
from mpc_rl.td3_mpc.sb3_td3_mpc import SB3_TD3_MPC


POLICY_NAME = "25% MPC-Injection"
SIMULATOR_NAME = "MuJoCo"
COMMAND = np.array([0.5, 0.0, 0.0], dtype=np.float64)
PUSH_TIME_S = 3.0
ROLLOUT_DURATION_S = 6.0
CONTACT_THRESHOLD_N = 1.0
MAGNITUDES_MPS = (0.5, 1.0, 1.5, 2.0)
SIDE_TO_SIGN = {"left": -1.0, "right": 1.0}
CSV_FIELDS = (
    "policy",
    "simulator",
    "checkpoint",
    "policy_path",
    "normalization_path",
    "run_config_path",
    "algorithm",
    "robot",
    "use_go2_sysid",
    "command_vx_mps",
    "command_vy_mps",
    "command_wz_radps",
    "push_source_side",
    "delta_vy_body_mps",
    "push_magnitude_mps",
    "push_time_s",
    "rollout_duration_s",
    "post_push_window_s",
    "simulator_timestep_s",
    "control_timestep_s",
    "decimation",
    "push_control_step",
    "base_vx_body_pre_push_mps",
    "base_vy_body_pre_push_mps",
    "base_vx_body_post_push_mps",
    "base_vy_body_post_push_mps",
    "actual_delta_vx_body_mps",
    "actual_delta_vy_body_mps",
    "vertical_velocity_delta_mps",
    "velocity_delta_tolerance_mps",
    "base_contact",
    "base_contact_time_s",
    "max_base_contact_force_n",
    "passed",
    "pre_push_failure",
    "non_base_contact",
    "max_non_base_contact_force_n",
    "minimum_base_height_m",
    "peak_abs_roll_rad",
    "peak_abs_pitch_rad",
    "native_termination_after_push",
    "domain_randomization_enabled",
    "observation_noise_enabled",
    "native_reset_randomization_enabled",
    "nominal_initial_state",
    "velocity_frame",
    "pass_fail_basis",
    "mujoco_version",
)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _quat_rotation(env: Any) -> Rotation:
    quat_wxyz = np.asarray(env.mjData.qpos[3:7], dtype=np.float64)
    return Rotation.from_quat(np.roll(quat_wxyz, -1))


def _yaw_rotation(env: Any) -> Rotation:
    yaw = _quat_rotation(env).as_euler("xyz")[2]
    return Rotation.from_euler("z", yaw)


def _yaw_frame_velocity(env: Any) -> np.ndarray:
    return _yaw_rotation(env).inv().apply(
        np.asarray(env.mjData.qvel[:3], dtype=np.float64)
    )


def _roll_pitch(env: Any) -> tuple[float, float]:
    roll, pitch, _ = _quat_rotation(env).as_euler("xyz")
    return float(roll), float(pitch)


def _ground_contact_forces(env: Any) -> tuple[float, float]:
    """Return maximum base and thigh/calf ground-contact force norms."""
    maximum_base_force = 0.0
    maximum_non_base_force = 0.0
    force = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.mjData.ncon):
        contact = env.mjData.contact[contact_index]
        body1 = int(env.mjModel.geom_bodyid[contact.geom1])
        body2 = int(env.mjModel.geom_bodyid[contact.geom2])
        if body1 != 0 and body2 != 0:
            continue
        other_body = body2 if body1 == 0 else body1
        mujoco.mj_contactForce(env.mjModel, env.mjData, contact_index, force)
        force_norm = float(np.linalg.norm(force[:3]))
        if other_body == env._base_body_id:
            maximum_base_force = max(maximum_base_force, force_norm)
            continue
        body_name = mujoco.mj_id2name(
            env.mjModel, mujoco.mjtObj.mjOBJ_BODY, other_body
        )
        if body_name and ("thigh" in body_name.lower() or "calf" in body_name.lower()):
            maximum_non_base_force = max(maximum_non_base_force, force_norm)
    return maximum_base_force, maximum_non_base_force


def _force_nominal_state(env: Any, seed: int) -> None:
    """Reset wrapper buffers, then replace all randomized plant state exactly."""
    env.reset(seed=seed)
    env.mjData.qpos[:] = env.default_qpos
    env.mjData.qvel[:] = 0.0
    env.mjData.qacc[:] = 0.0
    env.mjData.qacc_warmstart[:] = 0.0
    env.mjData.ctrl[:] = 0.0
    env._last_action[:] = 0.0
    env._prev_last_action[:] = 0.0
    env._raw_q_target[:] = env.default_joint_pos
    env._filtered_q_target[:] = env.default_joint_pos
    env._applied_torques[:] = 0.0
    env._last_joint_vel[:] = 0.0
    env._joint_acc[:] = 0.0
    env._feet_air_time[:] = 0.0
    env._feet_contact_time[:] = 0.0
    env._last_foot_contacts[:] = False
    env._swing_peak[:] = 0.0
    env._first_contact[:] = False
    env._current_contacts[:] = False
    env._step_count = 0
    env._steps_since_command_resample = 0
    env._steps_since_last_push = 0
    env.set_commands(vx=COMMAND[0], vy=COMMAND[1], wz=COMMAND[2])
    mujoco.mj_forward(env.mjModel, env.mjData)

    if not np.array_equal(env.mjData.qpos, env.default_qpos):
        raise AssertionError("failed to force exact default MuJoCo qpos")
    if not np.array_equal(env.mjData.qvel, np.zeros_like(env.mjData.qvel)):
        raise AssertionError("failed to force zero MuJoCo qvel")
    if not np.array_equal(env._commands, COMMAND):
        raise AssertionError(f"fixed command mismatch: {env._commands.tolist()}")


def _load_policy(
    *,
    run_dir: Path,
    policy_path: Path,
    normalization_path: Path,
    algorithm: str,
    robot: str,
    use_go2_sysid: bool,
):
    simple_reward = algorithm in {"SAC-MPC", "TD3-MPC"}
    vec_env = make_model_env(
        robot=robot,
        simple_reward=simple_reward,
        use_go2_sysid=use_go2_sysid,
    )
    vec_env = VecNormalize.load(normalization_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    algorithm_class = {
        "SAC": SB3_SAC,
        "TD3": SB3_TD3,
        "SAC-MPC": SB3_SAC_MPC,
        "TD3-MPC": SB3_TD3_MPC,
    }.get(algorithm)
    if algorithm_class is None:
        vec_env.close()
        raise ValueError(f"unsupported saved algorithm: {algorithm}")
    model = algorithm_class.load(policy_path, env=vec_env)
    return model, vec_env


def _apply_push(env: Any, signed_magnitude_mps: float) -> tuple[np.ndarray, ...]:
    pre_body = _yaw_frame_velocity(env)
    delta_body = np.array([0.0, signed_magnitude_mps, 0.0], dtype=np.float64)
    delta_world = _yaw_rotation(env).apply(delta_body)
    if delta_world[2] != 0.0:
        raise AssertionError(
            f"yaw-only push introduced vertical delta {delta_world[2]:.17g} m/s"
        )
    env.mjData.qvel[:3] += delta_world
    mujoco.mj_forward(env.mjModel, env.mjData)
    post_body = _yaw_frame_velocity(env)
    return pre_body, post_body, delta_body, delta_world


def _rollout_condition(
    model: Any,
    vec_env: VecNormalize,
    env: Any,
    *,
    source_side: str,
    magnitude_mps: float,
    seed: int,
    velocity_tolerance_mps: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if source_side not in SIDE_TO_SIGN:
        raise ValueError(f"unsupported push source side: {source_side}")
    signed_magnitude = SIDE_TO_SIGN[source_side] * magnitude_mps
    _force_nominal_state(env, seed)
    env.assert_generation_contact_friction_matches()

    control_steps_float = ROLLOUT_DURATION_S / env.control_dt
    push_step_float = PUSH_TIME_S / env.control_dt
    if not control_steps_float.is_integer() or not push_step_float.is_integer():
        raise AssertionError("rollout and push times must be exact control boundaries")
    control_steps = int(control_steps_float)
    push_step = int(push_step_float)

    push_applied = False
    base_contact = False
    base_contact_time_s: float | None = None
    pre_push_failure = False
    native_termination_after_push = False
    maximum_base_force = 0.0
    maximum_non_base_force = 0.0
    minimum_base_height = float(env.mjData.qpos[2])
    peak_abs_roll = 0.0
    peak_abs_pitch = 0.0
    pre_body: np.ndarray | None = None
    post_body: np.ndarray | None = None
    delta_body: np.ndarray | None = None
    delta_world: np.ndarray | None = None
    command_trace: list[np.ndarray] = []

    initial_base_force, initial_non_base_force = _ground_contact_forces(env)
    maximum_base_force = max(maximum_base_force, initial_base_force)
    maximum_non_base_force = max(maximum_non_base_force, initial_non_base_force)
    if initial_base_force > CONTACT_THRESHOLD_N:
        raise RuntimeError(
            f"pre-push base contact at t=0: {initial_base_force:.6f} N"
        )

    for control_step in range(control_steps):
        control_time_s = control_step * env.control_dt
        if control_step == push_step:
            pre_body, post_body, delta_body, delta_world = _apply_push(
                env, signed_magnitude
            )
            actual_delta = post_body - pre_body
            if abs(actual_delta[0]) > velocity_tolerance_mps:
                raise AssertionError(
                    f"push changed yaw-frame vx by {actual_delta[0]:.9g} m/s"
                )
            if abs(actual_delta[1] - signed_magnitude) > velocity_tolerance_mps:
                raise AssertionError(
                    "lateral push mismatch: requested "
                    f"{signed_magnitude:.9g}, measured {actual_delta[1]:.9g} m/s"
                )
            if abs(actual_delta[2]) > velocity_tolerance_mps:
                raise AssertionError(
                    f"push changed yaw-frame vz by {actual_delta[2]:.9g} m/s"
                )
            push_applied = True

        if not np.array_equal(env._commands, COMMAND):
            raise AssertionError(
                f"command changed at t={control_time_s:.9f}: {env._commands.tolist()}"
            )
        command_trace.append(np.array(env._commands, copy=True))
        obs_raw = env._get_obs()
        obs_normalized = normalize_obs_for_model(vec_env, obs_raw)
        action, _ = model.predict(obs_normalized, deterministic=True)
        action = np.clip(np.asarray(action[0], dtype=np.float64), -1.0, 1.0)

        env._prev_last_action = env._last_action.copy()
        env._last_action = action.copy()
        env._before_control_step()
        env._raw_q_target = env.default_joint_pos + env.action_scale * action
        q_target = env._apply_action_lpf(env._raw_q_target)

        stop_condition = False
        for substep in range(env.decimation):
            q_current = env.mjData.qpos[7:]
            dq_current = env.mjData.qvel[6:]
            torques = env.kp * (q_target - q_current) - env.kd * dq_current
            torques = np.clip(
                torques, env.torque_limits[:, 0], env.torque_limits[:, 1]
            )
            env._applied_torques = torques
            env.mjData.ctrl[:] = torques
            mujoco.mj_step(env.mjModel, env.mjData)
            env._after_physics_substep()

            sample_time_s = control_time_s + (substep + 1) * env.sim_dt
            base_force, non_base_force = _ground_contact_forces(env)
            maximum_base_force = max(maximum_base_force, base_force)
            maximum_non_base_force = max(maximum_non_base_force, non_base_force)
            minimum_base_height = min(minimum_base_height, float(env.mjData.qpos[2]))
            roll, pitch = _roll_pitch(env)
            peak_abs_roll = max(peak_abs_roll, abs(roll))
            peak_abs_pitch = max(peak_abs_pitch, abs(pitch))

            if base_force > CONTACT_THRESHOLD_N:
                if sample_time_s <= PUSH_TIME_S and not push_applied:
                    pre_push_failure = True
                else:
                    base_contact = True
                    base_contact_time_s = sample_time_s
                stop_condition = True
                break

        if stop_condition:
            break

        env._step_count += 1
        env._steps_since_command_resample += 1
        env._maybe_push_robot()
        env._update_feet_air_time()
        joint_velocity = env.mjData.qvel[6:].copy()
        env._joint_acc = (joint_velocity - env._last_joint_vel) / env.control_dt
        env._last_joint_vel = joint_velocity
        native_termination = bool(env._check_termination())
        if native_termination and not push_applied:
            pre_push_failure = True
            break
        if native_termination:
            native_termination_after_push = True
        env._swing_peak *= ~env._current_contacts

    if pre_push_failure:
        raise RuntimeError(
            f"invalid pre-push failure for side={source_side}, magnitude={magnitude_mps}"
        )
    if not push_applied or pre_body is None or post_body is None:
        raise AssertionError("push was not applied at the required control boundary")
    expected_command_trace = np.repeat(COMMAND[None, :], len(command_trace), axis=0)
    if not np.array_equal(np.stack(command_trace), expected_command_trace):
        raise AssertionError("command trace did not remain exactly [0.5, 0.0, 0.0]")

    actual_delta = post_body - pre_body
    assert delta_body is not None and delta_world is not None
    row: dict[str, Any] = {
        **metadata,
        "command_vx_mps": COMMAND[0],
        "command_vy_mps": COMMAND[1],
        "command_wz_radps": COMMAND[2],
        "push_source_side": source_side,
        "delta_vy_body_mps": signed_magnitude,
        "push_magnitude_mps": magnitude_mps,
        "push_time_s": PUSH_TIME_S,
        "rollout_duration_s": ROLLOUT_DURATION_S,
        "post_push_window_s": ROLLOUT_DURATION_S - PUSH_TIME_S,
        "simulator_timestep_s": env.sim_dt,
        "control_timestep_s": env.control_dt,
        "decimation": env.decimation,
        "push_control_step": push_step,
        "base_vx_body_pre_push_mps": pre_body[0],
        "base_vy_body_pre_push_mps": pre_body[1],
        "base_vx_body_post_push_mps": post_body[0],
        "base_vy_body_post_push_mps": post_body[1],
        "actual_delta_vx_body_mps": actual_delta[0],
        "actual_delta_vy_body_mps": actual_delta[1],
        "vertical_velocity_delta_mps": delta_world[2],
        "velocity_delta_tolerance_mps": velocity_tolerance_mps,
        "base_contact": _bool(base_contact),
        "base_contact_time_s": "" if base_contact_time_s is None else base_contact_time_s,
        "max_base_contact_force_n": maximum_base_force,
        "passed": _bool(not base_contact),
        "pre_push_failure": "false",
        "non_base_contact": _bool(maximum_non_base_force > CONTACT_THRESHOLD_N),
        "max_non_base_contact_force_n": maximum_non_base_force,
        "minimum_base_height_m": minimum_base_height,
        "peak_abs_roll_rad": peak_abs_roll,
        "peak_abs_pitch_rad": peak_abs_pitch,
        "native_termination_after_push": _bool(native_termination_after_push),
        "domain_randomization_enabled": "false",
        "observation_noise_enabled": "false",
        "native_reset_randomization_enabled": "false",
        "nominal_initial_state": "default joints; level base; zero base/joint velocity",
        "velocity_frame": "yaw-only horizontal body frame (+x forward, +y left)",
        "pass_fail_basis": "base-ground contact force > 1 N after push only",
        "mujoco_version": mujoco.__version__,
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    conditions = {
        (row["push_source_side"], float(row["push_magnitude_mps"])) for row in rows
    }
    expected = {(side, magnitude) for side in SIDE_TO_SIGN for magnitude in MAGNITUDES_MPS}
    if len(rows) != 8 or conditions != expected:
        raise AssertionError(
            f"expected exactly 8 unique MuJoCo conditions, got {len(rows)} rows: {conditions}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to write MuJoCo results: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--vec-normalize", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robot", default="go2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--velocity-tolerance-mps", type=float, default=1e-9)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one unreported zero-delta rollout and do not write a CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_path = args.policy.expanduser().resolve()
    normalization_path = args.vec_normalize.expanduser().resolve()
    for path in (run_dir / "config.json", policy_path, normalization_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if policy_path.parent != run_dir or normalization_path.parent != run_dir:
        raise ValueError("policy and VecNormalize state must come from the explicit run directory")
    if args.velocity_tolerance_mps <= 0.0:
        raise ValueError("--velocity-tolerance-mps must be positive")
    if not args.smoke and args.output is None:
        raise ValueError("--output is required unless --smoke is used")

    config = load_config(run_dir)
    algorithm = detect_algorithm(run_dir, config)
    use_go2_sysid = resolve_use_go2_sysid(config, None)
    if not use_go2_sysid:
        raise ValueError("the saved run configuration does not enable Go2 sysID")
    if config.get("percentage") != 25:
        raise ValueError(f"expected 25% MPC injection, got {config.get('percentage')!r}")

    model, vec_env = _load_policy(
        run_dir=run_dir,
        policy_path=policy_path,
        normalization_path=normalization_path,
        algorithm=algorithm,
        robot=args.robot,
        use_go2_sysid=use_go2_sysid,
    )
    env = build_quadruped_env(
        robot=args.robot,
        simple_reward=algorithm in {"SAC-MPC", "TD3-MPC"},
        use_go2_sysid=use_go2_sysid,
    ).unwrapped
    metadata = {
        "policy": POLICY_NAME,
        "simulator": SIMULATOR_NAME,
        "checkpoint": policy_path.name,
        "policy_path": str(policy_path),
        "normalization_path": str(normalization_path),
        "run_config_path": str(run_dir / "config.json"),
        "algorithm": algorithm,
        "robot": args.robot,
        "use_go2_sysid": _bool(use_go2_sysid),
    }
    try:
        if args.smoke:
            row = _rollout_condition(
                model,
                vec_env,
                env,
                source_side="right",
                magnitude_mps=0.0,
                seed=args.seed,
                velocity_tolerance_mps=args.velocity_tolerance_mps,
                metadata=metadata,
            )
            print("MUJOCO_ZERO_DELTA_SMOKE=" + json.dumps(row, sort_keys=True))
            return

        rows = []
        for source_side in SIDE_TO_SIGN:
            for magnitude_mps in MAGNITUDES_MPS:
                row = _rollout_condition(
                    model,
                    vec_env,
                    env,
                    source_side=source_side,
                    magnitude_mps=magnitude_mps,
                    seed=args.seed,
                    velocity_tolerance_mps=args.velocity_tolerance_mps,
                    metadata=metadata,
                )
                rows.append(row)
                print(
                    f"{POLICY_NAME} source={source_side} magnitude={magnitude_mps:.1f} "
                    f"result={'PASS' if row['passed'] == 'true' else 'FAIL'} "
                    f"contact_time={row['base_contact_time_s'] or '-'}"
                )
        assert args.output is not None
        output_path = args.output.expanduser().resolve()
        _write_csv(output_path, rows)
        print(f"mujoco_results_csv={output_path}")
    finally:
        env.close()
        vec_env.close()


if __name__ == "__main__":
    main()
