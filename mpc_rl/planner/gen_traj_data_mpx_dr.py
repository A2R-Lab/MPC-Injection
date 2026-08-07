import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from timeit import default_timer as timer
from types import SimpleNamespace

import jax
import mujoco
import numpy as np

# JAX configuration (must be before other JAX imports)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

import mpx.config.config_go2 as config
import mpx.utils.mpc_wrapper as mpc_wrapper
from mpc_rl.envs.action_interfaces import (
    ACTION_INTERFACES,
    DEFAULT_ACTION_INTERFACE_ID,
    ENV_STEP_LPF_INVERSE_MODE,
    INFERRED_ACTION_DIRECT_TORQUE_MODE,
    MPX_BOUND_ACTION_INTERFACE_ID,
    MPX_BOUND_ENV_STEP_MODE,
    action_interface_metadata,
    resolve_action_interface,
)
from mpc_rl.envs.domain_randomization import (
    STARTUP_DOMAIN_RAND_PRESET_NAMES,
    resolve_startup_domain_rand_config,
)
from mpc_rl.envs.go2_sysid import GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.planner.gen_traj_data_mpx import COMMAND_THRESHOLD, sample_commands
from mpc_rl.planner.mpx_bounding_data import (
    BOUNDING_GAITS,
    BOUNDING_STAGE_SEED_RANGES,
    build_command_schedule,
    command_schedule_id,
    make_commissioning_acceptance_declaration,
    sha256_file,
    validate_acceptance_declaration,
    validate_bounding_trajectory_data,
)


try:
    gpu_device = jax.devices("gpu")[0]
except RuntimeError:
    gpu_device = jax.devices("cpu")[0]
jax.default_device(gpu_device)


DEFAULT_MPX_TRAJECTORY_DOMAIN_RAND_PRESET = "sysid_dyn20_mjlab"
ACTION_CONVERSION_MODES = (
    INFERRED_ACTION_DIRECT_TORQUE_MODE,
    ENV_STEP_LPF_INVERSE_MODE,
    MPX_BOUND_ENV_STEP_MODE,
)
# Keep the historical behavior as the default until the conditional mapping
# has passed its predeclared actual-MPX parity gate.
DEFAULT_ACTION_CONVERSION_MODE = INFERRED_ACTION_DIRECT_TORQUE_MODE


def _controller_config_with_overrides(
    mpx_qrot_pitch_cost, mpx_qomega_pitch_cost, mpx_robot_height_m
):
    """Return the stock MPX config or an instance-local reference/cost variant."""
    if (
        mpx_qrot_pitch_cost is None
        and mpx_qomega_pitch_cost is None
        and mpx_robot_height_m is None
    ):
        return config
    pitch_cost = (
        float(np.asarray(config.Qrot)[1, 1])
        if mpx_qrot_pitch_cost is None
        else float(mpx_qrot_pitch_cost)
    )
    robot_height = (
        float(config.robot_height)
        if mpx_robot_height_m is None
        else float(mpx_robot_height_m)
    )
    pitch_rate_cost = (
        float(np.asarray(config.Qomega)[1, 1])
        if mpx_qomega_pitch_cost is None
        else float(mpx_qomega_pitch_cost)
    )
    if not np.isfinite(pitch_cost) or pitch_cost < 0.0:
        raise ValueError("mpx_qrot_pitch_cost must be finite and non-negative")
    if not np.isfinite(robot_height) or robot_height <= 0.0:
        raise ValueError("mpx_robot_height_m must be finite and positive")
    if not np.isfinite(pitch_rate_cost) or pitch_rate_cost < 0.0:
        raise ValueError(
            "mpx_qomega_pitch_cost must be finite and non-negative"
        )
    qrot = config.Qrot.astype(config.W.dtype).at[1, 1].set(pitch_cost)
    qomega = (
        config.Qomega.astype(config.W.dtype)
        .at[1, 1]
        .set(pitch_rate_cost)
    )
    weights = jax.scipy.linalg.block_diag(
        config.Qp,
        qrot,
        config.Qq,
        config.Qdp,
        qomega,
        config.Qdq,
        config.Qleg,
        config.Qtau,
        config.Q_grf,
    )
    attributes = {
        name: getattr(config, name)
        for name in dir(config)
        if not name.startswith("__")
    }
    p0 = config.p0.astype(config.W.dtype).at[2].set(robot_height)
    attributes.update(
        Qrot=qrot,
        Qomega=qomega,
        W=weights,
        initial_height=robot_height,
        p0=p0,
        robot_height=robot_height,
    )
    return SimpleNamespace(**attributes)


def _resolve_action_configuration(
    action_interface_id: str,
    action_conversion_mode: str | None,
):
    """Resolve and validate the coupled environment/generator action contract."""
    action_interface = resolve_action_interface(action_interface_id)
    if action_conversion_mode is None:
        action_conversion_mode = (
            action_interface.required_mpx_conversion_mode
            or DEFAULT_ACTION_CONVERSION_MODE
        )
    if action_conversion_mode not in ACTION_CONVERSION_MODES:
        raise ValueError(
            f"action_conversion_mode must be one of {ACTION_CONVERSION_MODES}, "
            f"got {action_conversion_mode!r}"
        )
    required_mode = action_interface.required_mpx_conversion_mode
    if required_mode is not None and action_conversion_mode != required_mode:
        raise ValueError(
            f"action interface {action_interface.interface_id!r} requires "
            f"action_conversion_mode={required_mode!r}, got "
            f"{action_conversion_mode!r}"
        )
    if (
        action_conversion_mode == MPX_BOUND_ENV_STEP_MODE
        and action_interface.interface_id != MPX_BOUND_ACTION_INTERFACE_ID
    ):
        raise ValueError(
            f"action_conversion_mode={MPX_BOUND_ENV_STEP_MODE!r} requires "
            f"action_interface_id={MPX_BOUND_ACTION_INTERFACE_ID!r}"
        )
    return action_interface, action_conversion_mode


def _actuated_joint_names(env: QuadrupedVelocityTrackingEnv) -> np.ndarray:
    """Return actuator joint names in the 12D action order."""
    names = []
    for actuator_index in range(env.num_joints):
        joint_id = int(env.mjModel.actuator_trnid[actuator_index, 0])
        joint_name = mujoco.mj_id2name(
            env.mjModel, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        names.append(joint_name or f"joint_{joint_id}")
    return np.asarray(names, dtype="<U64")


def _controller_gait_metadata(mpc) -> dict:
    """Return pickle-free effective gait metadata when the controller exposes it."""
    gait_parameters = getattr(mpc, "gait_parameters", None)
    if gait_parameters is None:
        return {
            "gait_name": "unspecified",
            "gait_initial_phase": np.asarray(config.timer_t, dtype=np.float64),
            "gait_duty_factor": float(getattr(mpc, "duty_factor", config.duty_factor)),
            "gait_step_frequency_hz": float(config.step_freq),
            "gait_step_height_m": float(config.step_height),
            "contact_order": np.asarray(config.contact_frame, dtype="<U16"),
        }
    return {
        "gait_name": str(gait_parameters["gait_name"]),
        "gait_initial_phase": np.asarray(
            gait_parameters["initial_phase"], dtype=np.float64
        ),
        "gait_duty_factor": float(gait_parameters["duty_factor"]),
        "gait_step_frequency_hz": float(
            gait_parameters["step_frequency_hz"]
        ),
        "gait_step_height_m": float(gait_parameters["step_height_m"]),
        "contact_order": np.asarray(config.contact_frame, dtype="<U16"),
    }


def _go2_sysid_signature_vector() -> np.ndarray:
    """Return a deterministic vector snapshot of the canonical Go2 sysID table."""
    values = []
    for joint_name in sorted(GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS):
        dynamics = GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS[joint_name]
        values.extend(
            [
                float(dynamics["armature"]),
                float(dynamics["damping"]),
                float(dynamics["frictionloss"]),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def _configure_mpc_duty_factor(
    commands: np.ndarray,
    mpc,
    *,
    moving_duty_factor: float = 0.5,
    move_on_any_nonzero_command: bool = False,
) -> float:
    """Match the nominal MPX standing-vs-trotting duty-factor heuristic."""
    total_command = np.linalg.norm(commands[:2]) + abs(commands[2])
    standing = (
        total_command == 0.0
        if move_on_any_nonzero_command
        else total_command < COMMAND_THRESHOLD
    )
    if standing:
        mpc.duty_factor = 1.0
    else:
        mpc.duty_factor = float(moving_duty_factor)
    return float(total_command)


def _validate_requested_gait(
    mpc,
    *,
    gait: str | None,
    duty_factor: float | None,
    step_frequency_hz: float | None,
    step_height_m: float | None,
) -> None:
    """Ensure a caller-supplied shared controller matches explicit settings."""
    if all(
        value is None
        for value in (gait, duty_factor, step_frequency_hz, step_height_m)
    ):
        return
    actual = getattr(mpc, "gait_parameters", None)
    if actual is None:
        raise ValueError(
            "explicit gait settings require a controller with gait_parameters"
        )
    expected = {
        "gait_name": gait,
        "duty_factor": duty_factor,
        "step_frequency_hz": step_frequency_hz,
        "step_height_m": step_height_m,
    }
    for key, value in expected.items():
        if value is None:
            continue
        if actual[key] != value:
            raise ValueError(
                f"shared controller {key}={actual[key]!r} does not match "
                f"requested value {value!r}"
            )


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    mpx_root = root / "deps" / "mpx"
    ilqr_root = mpx_root / "mpx" / "primal_dual_ilqr"
    return {
        "generator_root_commit": _git_commit(root),
        "generator_mpx_commit": _git_commit(mpx_root),
        "generator_primal_dual_ilqr_commit": _git_commit(ilqr_root),
    }


def _controller_configuration_arrays(env, mpc) -> dict:
    controller_config = getattr(mpc, "config", config)
    weights = np.asarray(getattr(controller_config, "W", config.W), dtype=np.float64)
    signature = hashlib.sha256(weights.tobytes(order="C")).hexdigest()
    return {
        "controller_weight_matrix": weights,
        "controller_weight_matrix_sha256": signature,
        "controller_qrot_pitch_cost": float(
            np.asarray(controller_config.Qrot, dtype=np.float64)[1, 1]
        ),
        "controller_qomega_pitch_cost": float(
            np.asarray(controller_config.Qomega, dtype=np.float64)[1, 1]
        ),
        "controller_robot_height_m": float(mpc.robot_height),
        "environment_kp": np.asarray(env.kp, dtype=np.float64).copy(),
        "environment_kd": np.asarray(env.kd, dtype=np.float64).copy(),
        "environment_torque_limits": np.asarray(
            env.torque_limits, dtype=np.float64
        ).copy(),
        "environment_max_pitch_rad": float(env.max_pitch),
        "environment_max_roll_rad": float(env.max_roll),
        "environment_min_base_height_m": float(env.min_base_height),
    }


def _inverse_pd_raw_residual_action(
    env: QuadrupedVelocityTrackingEnv, tau_applied_first: np.ndarray
) -> np.ndarray:
    """Recover the unclipped inverse-PD residual action."""
    q_current = env.mjData.qpos[7 : 7 + env.num_joints].copy()
    dq_current = env.mjData.qvel[6 : 6 + env.num_joints].copy()
    q_target = q_current + (tau_applied_first + env.kd * dq_current) / env.kp
    return ((q_target - env.default_joint_pos) / env.action_scale).astype(
        np.float64
    )


def _inverse_pd_residual_action(
    env: QuadrupedVelocityTrackingEnv, tau_applied_first: np.ndarray
) -> np.ndarray:
    """Recover the RL residual action whose PD torques match the first applied torque."""
    raw_action = _inverse_pd_raw_residual_action(env, tau_applied_first)
    return np.clip(raw_action, -1.0, 1.0).astype(np.float64)


def _mpx_to_env_step_raw_action(
    env: QuadrupedVelocityTrackingEnv,
    tau_ff: np.ndarray,
    q_des: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map MPX feedforward/position output through the environment action API.

    The desired filtered target preserves the MPX feedforward term under the
    environment's realized proportional gains. The raw target then inverts the
    first-order target LPF for one control step. Action and torque clipping are
    deliberately left to ``env.step`` and its diagnostics.
    """
    tau_ff = np.asarray(tau_ff, dtype=np.float64)
    q_des = np.asarray(q_des, dtype=np.float64)
    expected_shape = (env.num_joints,)
    if tau_ff.shape != expected_shape or q_des.shape != expected_shape:
        raise ValueError(
            "MPX tau_ff and q_des must both have shape "
            f"{expected_shape}, got {tau_ff.shape} and {q_des.shape}"
        )
    if not np.all(np.isfinite(tau_ff)) or not np.all(np.isfinite(q_des)):
        raise ValueError("MPX tau_ff and q_des must contain only finite values")

    kp = np.asarray(env.kp, dtype=np.float64)
    alpha = float(env.action_lpf_alpha)
    action_scale = float(env.action_scale)
    if kp.shape != expected_shape or not np.all(np.isfinite(kp)):
        raise ValueError("environment realized Kp must be a finite per-joint vector")
    if np.any(kp == 0.0):
        raise ValueError("environment realized Kp must be nonzero")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("environment action LPF alpha must be finite and positive")
    if not np.isfinite(action_scale) or action_scale == 0.0:
        raise ValueError("environment action_scale must be finite and nonzero")

    previous_filtered_target = np.asarray(
        env._filtered_q_target, dtype=np.float64
    )
    desired_filtered_target = q_des + tau_ff / kp
    raw_q_target = previous_filtered_target + (
        desired_filtered_target - previous_filtered_target
    ) / alpha
    raw_action = (raw_q_target - env.default_joint_pos) / action_scale
    return (
        raw_action.astype(np.float64),
        desired_filtered_target.astype(np.float64),
        raw_q_target.astype(np.float64),
    )


def _rollout_control_step_with_env_action(
    env: QuadrupedVelocityTrackingEnv,
    *,
    raw_action: np.ndarray,
    tau_mpx: np.ndarray,
    q_des: np.ndarray,
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    time_traj: np.ndarray,
    tau_applied_traj: np.ndarray,
    tau_mpx_traj: np.ndarray,
    q_des_traj: np.ndarray,
    commands_traj: np.ndarray,
    commands: np.ndarray,
    sim_idx_start: int,
    viewer=None,
):
    """Advance one exact environment transition and copy substep telemetry."""
    next_obs, reward, terminated, truncated, info = env.step(raw_action)
    diagnostics = info.get("substep_diagnostics")
    if diagnostics is None:
        raise RuntimeError(
            "env-step trajectory generation requires substep diagnostics"
        )

    completed_substeps = env.decimation
    sim_slice = slice(
        sim_idx_start + 1, sim_idx_start + completed_substeps + 1
    )
    qpos_traj[:, sim_slice] = np.asarray(diagnostics["qpos"]).T
    qvel_traj[:, sim_slice] = np.asarray(diagnostics["qvel"]).T
    time_traj[sim_slice] = np.asarray(diagnostics["time"])
    tau_applied_traj[:, sim_idx_start : sim_idx_start + completed_substeps] = (
        np.asarray(diagnostics["applied_torques"]).T
    )
    tau_mpx_traj[:, sim_idx_start : sim_idx_start + completed_substeps] = (
        np.asarray(tau_mpx, dtype=np.float64)[:, None]
    )
    q_des_traj[:, sim_idx_start : sim_idx_start + completed_substeps] = (
        np.asarray(q_des, dtype=np.float64)[:, None]
    )
    commands_traj[:, sim_idx_start : sim_idx_start + completed_substeps] = (
        np.asarray(commands, dtype=np.float64)[:, None]
    )

    non_foot_ground_contact = None
    contact_steps = np.flatnonzero(diagnostics["non_foot_ground_contact"])
    if contact_steps.size:
        first_substep = int(contact_steps[0])
        non_foot_ground_contact = (
            str(diagnostics["non_foot_ground_contact_body"][first_substep]),
            str(diagnostics["non_foot_ground_contact_detail"][first_substep]),
        )

    viewer_closed = False
    if viewer is not None:
        viewer.sync()
        time.sleep(env.control_dt)
        viewer_closed = not viewer.is_running()

    return (
        next_obs,
        float(reward),
        bool(terminated),
        bool(truncated),
        info,
        viewer_closed,
        non_foot_ground_contact,
        diagnostics,
    )


def _rollout_control_step_with_torques(
    env: QuadrupedVelocityTrackingEnv,
    *,
    action: np.ndarray,
    tau_mpx: np.ndarray,
    q_des: np.ndarray,
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    time_traj: np.ndarray,
    tau_applied_traj: np.ndarray,
    tau_mpx_traj: np.ndarray,
    q_des_traj: np.ndarray,
    commands_traj: np.ndarray,
    commands: np.ndarray,
    sim_idx_start: int,
    viewer=None,
):
    """Mirror env.step() bookkeeping while applying precomputed torques directly."""
    action = np.clip(action, -1.0, 1.0).astype(np.float64)
    env._prev_last_action = env._last_action.copy()
    env._last_action = action.copy()

    viewer_closed = False
    non_foot_ground_contact = None
    diagnostics = {
        "requested_torques": np.zeros(
            (env.decimation, env.num_joints), dtype=np.float64
        ),
        "applied_torques": np.zeros(
            (env.decimation, env.num_joints), dtype=np.float64
        ),
        "torque_saturation_mask": np.zeros(
            (env.decimation, env.num_joints), dtype=bool
        ),
        "torque_saturation_magnitude": np.zeros(
            (env.decimation, env.num_joints), dtype=np.float64
        ),
        "foot_contacts": np.zeros((env.decimation, env._num_feet), dtype=bool),
        "non_foot_ground_contact": np.zeros(env.decimation, dtype=bool),
        "non_foot_ground_contact_body": np.full(
            env.decimation, "", dtype="<U64"
        ),
        "non_foot_ground_contact_detail": np.full(
            env.decimation, "", dtype="<U256"
        ),
        "push_delta_qvel": np.zeros(6, dtype=np.float64),
        "completed_substeps": 0,
    }
    for substep in range(env.decimation):
        sim_idx = sim_idx_start + substep
        q_current = env.mjData.qpos[7 : 7 + env.num_joints].copy()
        dq_current = env.mjData.qvel[6 : 6 + env.num_joints].copy()
        tau_feedback = 10.0 * (q_des - q_current) - 2.0 * dq_current
        total_tau = tau_mpx + tau_feedback
        clipped_torques = np.clip(
            np.asarray(total_tau, dtype=np.float64),
            env.torque_limits[:, 0],
            env.torque_limits[:, 1],
        )
        torque_clip_delta = np.asarray(total_tau) - clipped_torques
        diagnostics["requested_torques"][substep] = total_tau
        diagnostics["applied_torques"][substep] = clipped_torques
        diagnostics["torque_saturation_mask"][substep] = (
            torque_clip_delta != 0.0
        )
        diagnostics["torque_saturation_magnitude"][substep] = np.abs(
            torque_clip_delta
        )
        env._applied_torques = clipped_torques
        tau_applied_traj[:, sim_idx] = clipped_torques
        tau_mpx_traj[:, sim_idx] = tau_mpx
        q_des_traj[:, sim_idx] = q_des
        commands_traj[:, sim_idx] = commands
        env.mjData.ctrl[:] = clipped_torques
        mujoco.mj_step(env.mjModel, env.mjData)

        qpos_traj[:, sim_idx + 1] = env.mjData.qpos.copy()
        qvel_traj[:, sim_idx + 1] = env.mjData.qvel.copy()
        time_traj[sim_idx + 1] = env.mjData.time
        diagnostics["foot_contacts"][substep] = env._get_foot_contacts()
        diagnostics["completed_substeps"] = substep + 1

        non_foot_ground_contact = env._find_non_foot_ground_contact()
        if non_foot_ground_contact is not None:
            body_name, detail = non_foot_ground_contact
            diagnostics["non_foot_ground_contact"][substep] = True
            diagnostics["non_foot_ground_contact_body"][substep] = body_name
            diagnostics["non_foot_ground_contact_detail"][substep] = detail
            break

        if viewer is not None:
            viewer.sync()
            time.sleep(env.sim_dt)
            if not viewer.is_running():
                viewer_closed = True
                break

    env._step_count += 1
    env._steps_since_command_resample += 1
    base_qvel_before_push = env.mjData.qvel[:6].copy()
    env._maybe_push_robot()
    diagnostics["push_delta_qvel"] = env.mjData.qvel[:6] - base_qvel_before_push
    env._update_feet_air_time()

    joint_vel_current = env.mjData.qvel[6:].copy()
    env._joint_acc = (joint_vel_current - env._last_joint_vel) / env.control_dt
    env._last_joint_vel = joint_vel_current

    next_obs = env._get_obs()
    terminated = env._check_termination()
    reward = env._compute_reward(action, terminated)
    info = env._get_info()
    env._swing_peak *= ~env._current_contacts

    return (
        next_obs,
        float(reward),
        bool(terminated),
        info,
        viewer_closed,
        non_foot_ground_contact,
        diagnostics,
    )


def _select_summary_geom_friction(
    mj_model: mujoco.MjModel,
    geom_friction: np.ndarray,
    target_geom_names: tuple[str, ...] | None,
) -> float:
    """Pick the friction value that best represents startup friction DR."""
    if target_geom_names is None:
        return float(geom_friction[0, 0])

    normalized_targets = {name.lower() for name in target_geom_names}
    for geom_id in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and geom_name.lower() in normalized_targets:
            return float(geom_friction[geom_id, 0])

    return float(geom_friction[0, 0])


def _summarize_dr_bundle(
    base_body_id: int,
    dr_bundle: dict,
    *,
    summary_geom_friction: float,
) -> dict:
    """Create a compact JSON-serializable DR summary for the manifest."""
    encoder_bias = np.asarray(dr_bundle["dr_encoder_bias"], dtype=np.float64)
    return {
        "dr_enabled": bool(dr_bundle["dr_enabled"]),
        "dr_config_type": str(dr_bundle["dr_config_type"]),
        "dr_seed": int(dr_bundle["dr_seed"]),
        "dr_applied_fields": [str(field) for field in dr_bundle["dr_applied_fields"].tolist()],
        "dr_added_mass_kg": float(dr_bundle["dr_added_mass_kg"]),
        "dr_motor_strength_scale": float(dr_bundle["dr_motor_strength_scale"]),
        "geom_friction": float(summary_geom_friction),
        "base_ipos": dr_bundle["dr_patch_body_ipos"][base_body_id].tolist(),
        "encoder_bias_min": float(np.min(encoder_bias)),
        "encoder_bias_max": float(np.max(encoder_bias)),
    }


def generate_trajectory(
    seed,
    *,
    domain_rand_config_type=DEFAULT_MPX_TRAJECTORY_DOMAIN_RAND_PRESET,
    dr_seed_offset=1_000_000,
    mpc=None,
    episode_length=1000,
    verbose=1,
    render=False,
    use_go2_sysid=True,
    deterministic_push_schedule=None,
    fixed_command=None,
    command_ramp_control_steps=0,
    gait=None,
    duty_factor=None,
    step_frequency_hz=None,
    step_height_m=None,
    mpx_qrot_pitch_cost=None,
    mpx_qomega_pitch_cost=None,
    mpx_robot_height_m=None,
    joint_kp=None,
    joint_kd=None,
    max_pitch=0.5,
    max_roll=0.5,
    min_base_height=0.1,
    action_interface_id=DEFAULT_ACTION_INTERFACE_ID,
    action_conversion_mode=None,
    provenance=None,
):
    """Generate one MPX-controlled trajectory inside the RL quadruped env."""
    action_interface, action_conversion_mode = _resolve_action_configuration(
        action_interface_id, action_conversion_mode
    )
    rng = np.random.RandomState(seed)
    dr_seed = seed + dr_seed_offset
    dr_rng = np.random.RandomState(dr_seed)
    resolved_dr_type, domain_rand_cfg = resolve_startup_domain_rand_config(
        domain_rand_config_type
    )

    configured_command_schedule = None
    if fixed_command is not None:
        configured_command_schedule = build_command_schedule(
            fixed_command,
            episode_length=episode_length,
            ramp_control_steps=command_ramp_control_steps,
        )
    elif command_ramp_control_steps != 0:
        raise ValueError("command ramp requires a fixed target command")

    # Keep startup dyanmics DR, but make the observations clean so we have
    # clean obs -> MPC(clean state) instead of noisy obs -> MPC(noisy state) which would be a different distribution shift
    domain_rand_cfg.obs_noise_level = 0.0
    domain_rand_cfg.encoder_bias_range = (0.0, 0.0)

    env = QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        render_mode="rgb_array" if render else None,
        domain_rand_cfg=domain_rand_cfg,
        apply_startup_domain_rand_on_init=False,
        simple_reward=True,
        use_go2_sysid=use_go2_sysid,
        enable_substep_diagnostics=True,
        deterministic_push_schedule=deterministic_push_schedule,
        kp=joint_kp,
        kd=joint_kd,
        max_pitch=max_pitch,
        max_roll=max_roll,
        min_base_height=min_base_height,
        **action_interface.env_kwargs(),
    )

    own_mpc = mpc is None
    try:
        dr_bundle = env.sample_startup_domain_rand_bundle(
            rng=dr_rng,
            dr_config_type=resolved_dr_type,
            dr_seed=dr_seed,
        )
        env.apply_startup_domain_rand_bundle(dr_bundle)
        if configured_command_schedule is None:
            commands = sample_commands(rng)
        else:
            commands = configured_command_schedule[0].copy()
        env.set_commands(
            vx=float(commands[0]),
            vy=float(commands[1]),
            wz=float(commands[2]),
        )
        obs, _ = env.reset(seed=seed)
        dr_bundle = env.export_startup_domain_rand_patch(
            dr_config_type=resolved_dr_type,
            dr_seed=dr_seed,
        )

        n_joints = env.num_joints
        sim_steps_per_ctrl = env.decimation
        total_sim_steps = episode_length * sim_steps_per_ctrl

        if own_mpc:
            controller_config = _controller_config_with_overrides(
                mpx_qrot_pitch_cost,
                mpx_qomega_pitch_cost,
                mpx_robot_height_m,
            )
            mpc = mpc_wrapper.MPCControllerWrapper(
                controller_config,
                use_go2_sysid=use_go2_sysid,
                gait=gait,
                duty_factor=duty_factor,
                step_frequency_hz=step_frequency_hz,
                step_height_m=step_height_m,
                enable_planned_contact_diagnostics=(
                    configured_command_schedule is not None
                ),
            )
            mpc.robot_height = controller_config.robot_height
            if verbose > 0:
                print(f"[Seed {seed}] Pre-compiling JAX MPC kernels...")
            dummy_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
            dummy_contact = np.zeros(config.n_contact)
            compile_start = timer()
            mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
            mpc.run(env.mjData.qpos.copy(), env.mjData.qvel.copy(), dummy_input, dummy_contact)
            if verbose > 0:
                print(f"[Seed {seed}] JIT compilation done in {timer() - compile_start:.1f}s")

        _validate_requested_gait(
            mpc,
            gait=gait,
            duty_factor=duty_factor,
            step_frequency_hz=step_frequency_hz,
            step_height_m=step_height_m,
        )

        mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
        gait_metadata = _controller_gait_metadata(mpc)
        moving_duty_factor = float(gait_metadata["gait_duty_factor"])
        total_command = _configure_mpc_duty_factor(
            commands,
            mpc,
            moving_duty_factor=moving_duty_factor,
            move_on_any_nonzero_command=(configured_command_schedule is not None),
        )
        controller_configuration = _controller_configuration_arrays(env, mpc)

        nq = env.mjModel.nq
        nv = env.mjModel.nv
        qpos_traj = np.zeros((nq, total_sim_steps + 1), dtype=np.float64)
        qvel_traj = np.zeros((nv, total_sim_steps + 1), dtype=np.float64)
        tau_applied_traj = np.zeros((n_joints, total_sim_steps), dtype=np.float64)
        tau_mpx_traj = np.zeros((n_joints, total_sim_steps), dtype=np.float64)
        q_des_traj = np.zeros((n_joints, total_sim_steps), dtype=np.float64)
        time_traj = np.zeros(total_sim_steps + 1, dtype=np.float64)
        commands_traj = np.zeros((3, total_sim_steps), dtype=np.float64)

        qpos_traj[:, 0] = env.mjData.qpos.copy()
        qvel_traj[:, 0] = env.mjData.qvel.copy()
        time_traj[0] = env.mjData.time

        policy_obs_traj = []
        next_policy_obs_traj = []
        privileged_obs_traj = []
        next_privileged_obs_traj = []
        raw_actions_traj = []
        actions_traj = []
        action_clipping_mask_traj = []
        action_clipping_magnitude_traj = []
        rewards_traj = []
        terminated_ctrl_traj = []
        truncated_ctrl_traj = []
        commands_ctrl_traj = []
        measured_contacts_ctrl_traj = []
        planned_contact_schedule_ctrl_traj = []
        next_qpos_ctrl_traj = []
        next_qvel_ctrl_traj = []
        requested_torques_ctrl_traj = []
        applied_torques_ctrl_traj = []
        torque_saturation_mask_traj = []
        torque_saturation_magnitude_traj = []
        foot_contacts_substep_traj = []
        non_foot_ground_contact_substep_traj = []
        push_delta_qvel_traj = []
        solver_finite_ctrl_traj = []

        steps_since_resample = 0

        if verbose > 0:
            print(f"[Seed {seed}] Starting DR trajectory generation")
            print(f"  DR preset: {resolved_dr_type}")
            print(f"  DR applied fields: {dr_bundle['dr_applied_fields'].tolist()}")
            print(
                "  Episode length: "
                f"{episode_length} ctrl steps ({total_sim_steps} sim steps)"
            )
            print(f"  Init qpos (joints): {env.mjData.qpos[7:7+n_joints]}")
            print(
                f"  Init commands: vx={commands[0]:.2f}, "
                f"vy={commands[1]:.2f}, wz={commands[2]:.2f}"
            )
            print(f"  Initial total command: {total_command:.3f}")
            print(f"  Action interface: {action_interface.interface_id}")
            print(f"  Action conversion: {action_conversion_mode}")

        viewer = None
        if render:
            import mujoco.viewer as mjviewer

            viewer = mjviewer.launch_passive(env.mjModel, env.mjData)

        mpc_solve_times = []
        fell = False
        failure_reason = ""
        completed_control_steps = 0

        for ctrl_step in range(episode_length):
            if configured_command_schedule is not None:
                scheduled_commands = configured_command_schedule[ctrl_step]
                if not np.array_equal(scheduled_commands, commands):
                    commands = scheduled_commands.copy()
                    total_command = _configure_mpc_duty_factor(
                        commands,
                        mpc,
                        moving_duty_factor=moving_duty_factor,
                        move_on_any_nonzero_command=True,
                    )
                    env.set_commands(
                        vx=float(commands[0]),
                        vy=float(commands[1]),
                        wz=float(commands[2]),
                    )
                    obs = env._get_obs()
            elif (
                steps_since_resample >= env.command_resample_interval
                and ctrl_step > 0
            ):
                commands = sample_commands(rng)
                steps_since_resample = 0
                total_command = _configure_mpc_duty_factor(
                    commands,
                    mpc,
                    moving_duty_factor=moving_duty_factor,
                )
                env.set_commands(
                    vx=float(commands[0]),
                    vy=float(commands[1]),
                    wz=float(commands[2]),
                )
                obs = env._get_obs()
                if verbose > 1:
                    print(
                        f"  [ctrl={ctrl_step}] Resampled commands: "
                        f"vx={commands[0]:.2f}, vy={commands[1]:.2f}, wz={commands[2]:.2f} "
                        f"(total={total_command:.3f})"
                    )
            steps_since_resample += 1

            policy_obs_traj.append(obs["policy"].copy())
            privileged_obs_traj.append(obs["privileged"].copy())
            commands_ctrl_traj.append(commands.copy())

            mpx_input = np.array(
                [commands[0], commands[1], 0.0, 0.0, 0.0, commands[2], config.robot_height]
            )
            contact = env._get_foot_contacts()
            measured_contacts_ctrl_traj.append(contact.copy())

            solve_start = timer()
            tau_mpx, q_des, _ = mpc.run(
                env.mjData.qpos.copy(),
                env.mjData.qvel.copy(),
                mpx_input,
                contact,
            )
            mpc_solve_times.append(timer() - solve_start)

            tau_mpx = np.asarray(tau_mpx, dtype=np.float64)
            q_des = np.asarray(q_des, dtype=np.float64)
            solver_finite = bool(
                tau_mpx.shape == (n_joints,)
                and q_des.shape == (n_joints,)
                and np.all(np.isfinite(tau_mpx))
                and np.all(np.isfinite(q_des))
            )
            if not solver_finite:
                policy_obs_traj.pop()
                privileged_obs_traj.pop()
                commands_ctrl_traj.pop()
                measured_contacts_ctrl_traj.pop()
                failure_reason = "nonfinite_or_invalid_mpx_output"
                break
            solver_finite_ctrl_traj.append(True)
            if getattr(mpc, "enable_planned_contact_diagnostics", False):
                planned_schedule = mpc.planned_contact_schedule
                if planned_schedule is None:
                    raise RuntimeError(
                        "planned-contact diagnostics did not produce a schedule"
                    )
                planned_contact_schedule_ctrl_traj.append(
                    np.asarray(planned_schedule, dtype=np.int8)
                )

            sim_idx_start = ctrl_step * sim_steps_per_ctrl
            if action_conversion_mode == INFERRED_ACTION_DIRECT_TORQUE_MODE:
                q_current = env.mjData.qpos[7 : 7 + n_joints].copy()
                dq_current = env.mjData.qvel[6 : 6 + n_joints].copy()
                tau_feedback_first = 10.0 * (q_des - q_current) - 2.0 * dq_current
                tau_applied_first = np.clip(
                    tau_mpx + tau_feedback_first,
                    env.torque_limits[:, 0],
                    env.torque_limits[:, 1],
                )
                raw_action = _inverse_pd_raw_residual_action(
                    env, tau_applied_first
                )
            else:
                raw_action, _, _ = _mpx_to_env_step_raw_action(
                    env, tau_mpx, q_des
                )
            action = np.clip(raw_action, -1.0, 1.0).astype(np.float64)

            if action_conversion_mode == INFERRED_ACTION_DIRECT_TORQUE_MODE:
                (
                    next_obs,
                    reward,
                    terminated,
                    _,
                    viewer_closed,
                    non_foot_ground_contact,
                    step_diagnostics,
                ) = _rollout_control_step_with_torques(
                    env,
                    action=action,
                    tau_mpx=tau_mpx,
                    q_des=q_des,
                    qpos_traj=qpos_traj,
                    qvel_traj=qvel_traj,
                    time_traj=time_traj,
                    tau_applied_traj=tau_applied_traj,
                    tau_mpx_traj=tau_mpx_traj,
                    q_des_traj=q_des_traj,
                    commands_traj=commands_traj,
                    commands=commands,
                    sim_idx_start=sim_idx_start,
                    viewer=viewer,
                )
                truncated = False
            else:
                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                    viewer_closed,
                    non_foot_ground_contact,
                    step_diagnostics,
                ) = _rollout_control_step_with_env_action(
                    env,
                    raw_action=raw_action,
                    tau_mpx=tau_mpx,
                    q_des=q_des,
                    qpos_traj=qpos_traj,
                    qvel_traj=qvel_traj,
                    time_traj=time_traj,
                    tau_applied_traj=tau_applied_traj,
                    tau_mpx_traj=tau_mpx_traj,
                    q_des_traj=q_des_traj,
                    commands_traj=commands_traj,
                    commands=commands,
                    sim_idx_start=sim_idx_start,
                    viewer=viewer,
                )

            if action_conversion_mode == INFERRED_ACTION_DIRECT_TORQUE_MODE:
                recorded_raw_action = raw_action
                recorded_action = action
                recorded_clipping_mask = raw_action != action
                recorded_clipping_magnitude = np.abs(raw_action - action)
            else:
                recorded_raw_action = np.asarray(
                    step_diagnostics["raw_action"], dtype=np.float64
                )
                recorded_action = np.asarray(
                    step_diagnostics["clipped_action"], dtype=np.float64
                )
                recorded_clipping_mask = np.asarray(
                    step_diagnostics["action_clipping_mask"], dtype=bool
                )
                recorded_clipping_magnitude = np.asarray(
                    step_diagnostics["action_clipping_magnitude"],
                    dtype=np.float64,
                )
            raw_actions_traj.append(recorded_raw_action.copy())
            actions_traj.append(recorded_action.copy())
            action_clipping_mask_traj.append(recorded_clipping_mask.copy())
            action_clipping_magnitude_traj.append(
                recorded_clipping_magnitude.copy()
            )

            next_policy_obs_traj.append(next_obs["policy"].copy())
            next_privileged_obs_traj.append(next_obs["privileged"].copy())
            rewards_traj.append(reward)
            terminated_ctrl_traj.append(terminated)
            truncated_ctrl_traj.append(truncated)
            next_qpos_ctrl_traj.append(env.mjData.qpos.copy())
            next_qvel_ctrl_traj.append(env.mjData.qvel.copy())
            requested_torques_ctrl_traj.append(
                step_diagnostics["requested_torques"].copy()
            )
            applied_torques_ctrl_traj.append(
                step_diagnostics["applied_torques"].copy()
            )
            torque_saturation_mask_traj.append(
                step_diagnostics["torque_saturation_mask"].copy()
            )
            torque_saturation_magnitude_traj.append(
                step_diagnostics["torque_saturation_magnitude"].copy()
            )
            foot_contacts_substep_traj.append(
                step_diagnostics["foot_contacts"].copy()
            )
            non_foot_ground_contact_substep_traj.append(
                step_diagnostics["non_foot_ground_contact"].copy()
            )
            push_delta_qvel_traj.append(
                step_diagnostics["push_delta_qvel"].copy()
            )
            obs = next_obs
            completed_control_steps = ctrl_step + 1

            if viewer_closed:
                failure_reason = "viewer_closed"
                break

            if non_foot_ground_contact is not None:
                fell = True
                body_name, contact_detail = non_foot_ground_contact
                if body_name.lower() in {"base", "torso", "trunk"}:
                    failure_reason = f"torso_ground_contact: {contact_detail}"
                else:
                    failure_reason = f"non_foot_ground_contact: {contact_detail}"
                if verbose > 1:
                    print(
                        f"  [Seed {seed}] {failure_reason} at ctrl step "
                        f"{completed_control_steps} - aborting"
                    )
                break

            if terminated:
                fell = True
                failure_reason = "terminated"
                if verbose > 1:
                    print(
                        f"  [Seed {seed}] Robot fell at ctrl step "
                        f"{completed_control_steps} - aborting"
                    )
                break

        if viewer is not None:
            viewer.close()

        if verbose > 0:
            avg_solve = np.mean(mpc_solve_times) if mpc_solve_times else 0.0
            status = "FELL" if fell else ("INCOMPLETE" if completed_control_steps < episode_length else "OK")
            print(f"  [Seed {seed}] Done [{status}]. Avg MPC solve: {avg_solve*1000:.1f}ms")

        dr_summary_geom_friction = _select_summary_geom_friction(
            env.mjModel,
            np.asarray(dr_bundle["dr_patch_geom_friction"], dtype=np.float64),
            env.domain_rand_cfg.friction_target_geom_names,
        )
        deterministic_push_steps = sorted(
            (deterministic_push_schedule or {}).keys()
        )
        target_command = (
            np.asarray(fixed_command, dtype=np.float64)
            if fixed_command is not None
            else np.zeros(3, dtype=np.float64)
        )
        schedule_identifier = (
            command_schedule_id(
                target_command,
                episode_length=episode_length,
                ramp_control_steps=command_ramp_control_steps,
            )
            if configured_command_schedule is not None
            else "legacy_resampled_commands_v1"
        )
        generation_settings = {
            "action_conversion_mode": action_conversion_mode,
            "action_interface_id": action_interface.interface_id,
            "command_ramp_control_steps": int(command_ramp_control_steps),
            "command_schedule_id": schedule_identifier,
            "domain_rand_config_type": resolved_dr_type,
            "episode_length": int(episode_length),
            "fixed_command_enabled": configured_command_schedule is not None,
            "gait": {
                "duty_factor": float(gait_metadata["gait_duty_factor"]),
                "name": str(gait_metadata["gait_name"]),
                "step_frequency_hz": float(
                    gait_metadata["gait_step_frequency_hz"]
                ),
                "step_height_m": float(gait_metadata["gait_step_height_m"]),
            },
            "controller": {
                "environment_kd": controller_configuration[
                    "environment_kd"
                ].tolist(),
                "environment_kp": controller_configuration[
                    "environment_kp"
                ].tolist(),
                "environment_max_pitch_rad": controller_configuration[
                    "environment_max_pitch_rad"
                ],
                "environment_max_roll_rad": controller_configuration[
                    "environment_max_roll_rad"
                ],
                "environment_min_base_height_m": controller_configuration[
                    "environment_min_base_height_m"
                ],
                "environment_torque_limits": controller_configuration[
                    "environment_torque_limits"
                ].tolist(),
                "mpx_robot_height_m": controller_configuration[
                    "controller_robot_height_m"
                ],
                "mpx_qrot_pitch_cost": controller_configuration[
                    "controller_qrot_pitch_cost"
                ],
                "mpx_qomega_pitch_cost": controller_configuration[
                    "controller_qomega_pitch_cost"
                ],
                "mpx_weight_matrix_sha256": controller_configuration[
                    "controller_weight_matrix_sha256"
                ],
            },
            "target_command": target_command.tolist(),
            "use_go2_sysid": bool(use_go2_sysid),
        }
        recorded_provenance = provenance or {
            "generator_root_commit": "unrecorded_direct_call",
            "generator_mpx_commit": "unrecorded_direct_call",
            "generator_primal_dual_ilqr_commit": "unrecorded_direct_call",
        }

        return {
            "qpos": qpos_traj,
            "qvel": qvel_traj,
            "tau_applied": tau_applied_traj,
            "tau_mpx": tau_mpx_traj,
            "q_des": q_des_traj,
            "time": time_traj,
            "commands": commands_traj,
            "policy_obs": np.asarray(policy_obs_traj, dtype=np.float64),
            "next_policy_obs": np.asarray(next_policy_obs_traj, dtype=np.float64),
            "privileged_obs": np.asarray(privileged_obs_traj, dtype=np.float64),
            "next_privileged_obs": np.asarray(next_privileged_obs_traj, dtype=np.float64),
            "raw_actions": np.asarray(raw_actions_traj, dtype=np.float64),
            "actions": np.asarray(actions_traj, dtype=np.float64),
            "action_clipping_mask": np.asarray(
                action_clipping_mask_traj, dtype=bool
            ),
            "action_clipping_magnitude": np.asarray(
                action_clipping_magnitude_traj, dtype=np.float64
            ),
            "rewards": np.asarray(rewards_traj, dtype=np.float32),
            "terminated_ctrl": np.asarray(terminated_ctrl_traj, dtype=bool),
            "truncated_ctrl": np.asarray(truncated_ctrl_traj, dtype=bool),
            "commands_ctrl": np.asarray(commands_ctrl_traj, dtype=np.float64),
            "measured_contacts_ctrl": np.asarray(
                measured_contacts_ctrl_traj, dtype=bool
            ),
            "planned_contact_schedule_ctrl": (
                np.asarray(planned_contact_schedule_ctrl_traj, dtype=np.int8)
                if planned_contact_schedule_ctrl_traj
                else np.zeros(
                    (completed_control_steps, 0, config.n_contact),
                    dtype=np.int8,
                )
            ),
            "next_qpos_ctrl": np.asarray(next_qpos_ctrl_traj, dtype=np.float64),
            "next_qvel_ctrl": np.asarray(next_qvel_ctrl_traj, dtype=np.float64),
            "requested_torques_ctrl": np.asarray(
                requested_torques_ctrl_traj, dtype=np.float64
            ),
            "applied_torques_ctrl": np.asarray(
                applied_torques_ctrl_traj, dtype=np.float64
            ),
            "torque_saturation_mask": np.asarray(
                torque_saturation_mask_traj, dtype=bool
            ),
            "torque_saturation_magnitude": np.asarray(
                torque_saturation_magnitude_traj, dtype=np.float64
            ),
            "foot_contacts_substeps": np.asarray(
                foot_contacts_substep_traj, dtype=bool
            ),
            "non_foot_ground_contact_substeps": np.asarray(
                non_foot_ground_contact_substep_traj, dtype=bool
            ),
            "push_delta_qvel": np.asarray(
                push_delta_qvel_traj, dtype=np.float64
            ),
            "solver_finite_ctrl": np.asarray(
                solver_finite_ctrl_traj, dtype=bool
            ),
            "deterministic_push_steps": np.asarray(
                deterministic_push_steps, dtype=np.int64
            ),
            "deterministic_push_deltas": np.asarray(
                [deterministic_push_schedule[step] for step in deterministic_push_steps]
                if deterministic_push_schedule
                else [],
                dtype=np.float64,
            ).reshape(-1, 6),
            "seed": seed,
            "dr_seed": dr_seed,
            "base_body_id": int(env._base_body_id),
            "dr_summary_geom_friction": float(dr_summary_geom_friction),
            "default_joint_pos": env.default_joint_pos.copy(),
            "joint_names": _actuated_joint_names(env),
            **action_interface_metadata(
                action_interface,
                action_lpf_alpha=float(env.action_lpf_alpha),
            ),
            "sim_dt": float(env.sim_dt),
            "control_dt": float(env.control_dt),
            "episode_length": int(episode_length),
            "completed_control_steps": int(completed_control_steps),
            "completed_sim_steps": int(completed_control_steps * sim_steps_per_ctrl),
            "fell": bool(fell),
            "failure_reason": failure_reason,
            "fixed_command_enabled": bool(fixed_command is not None),
            "fixed_command": target_command.copy(),
            "target_command": target_command.copy(),
            "command_ramp_control_steps": int(command_ramp_control_steps),
            "command_schedule_id": schedule_identifier,
            "configured_command_schedule_ctrl": (
                configured_command_schedule.copy()
                if configured_command_schedule is not None
                else np.zeros((0, 3), dtype=np.float64)
            ),
            "generation_settings_json": json.dumps(
                generation_settings,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "domain_rand_config_type": resolved_dr_type,
            **gait_metadata,
            "action_conversion_mode": action_conversion_mode,
            "controller_delay_steps": 0,
            "controller_delay_s": 0.0,
            "go2_sysid_expected_vector": _go2_sysid_signature_vector(),
            "go2_sysid_enabled": bool(use_go2_sysid),
            **controller_configuration,
            **recorded_provenance,
            **dr_bundle,
        }
    finally:
        env.close()


def _append_manifest_record(manifest_path: Path, record: dict):
    """Append one generation attempt record to the JSONL manifest."""
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL record at {path}:{line_number}"
            ) from exc
    return records


def _command_filename_token(command: np.ndarray) -> str:
    def encode(value: float) -> str:
        text = f"{abs(float(value)):.6f}".rstrip("0").rstrip(".") or "0"
        return ("m" if value < 0 else "") + text.replace(".", "p")

    return "_".join(
        (f"vx{encode(command[0])}", f"vy{encode(command[1])}", f"wz{encode(command[2])}")
    )


def _bounding_filename(traj_data: dict, seed: int, episode_length: int) -> str:
    gait = str(traj_data["gait_name"])
    command = np.asarray(traj_data["target_command"], dtype=np.float64)
    return (
        f"quadruped_bound_{gait}_{_command_filename_token(command)}_"
        f"seed_{seed:06d}_ep_{episode_length}.npz"
    )


def _attempt_evidence(traj_data: dict, *, ramp_control_steps: int) -> dict:
    """Return compact measured evidence even when an attempt ends early."""
    completed = int(traj_data["completed_control_steps"])
    completed_substeps = int(traj_data["completed_sim_steps"])
    clipping_mask = np.asarray(traj_data["action_clipping_mask"], dtype=bool)
    clipping_magnitude = np.asarray(
        traj_data["action_clipping_magnitude"], dtype=np.float64
    )
    saturation_mask = np.asarray(traj_data["torque_saturation_mask"], dtype=bool)
    saturation_magnitude = np.asarray(
        traj_data["torque_saturation_magnitude"], dtype=np.float64
    )
    contacts = np.asarray(traj_data["foot_contacts_substeps"], dtype=bool).reshape(
        -1, 4
    )
    hold_start = int(ramp_control_steps) * 4
    hold_contacts = contacts[hold_start:completed_substeps]
    qpos = np.asarray(traj_data["qpos"], dtype=np.float64).T[
        : completed_substeps + 1
    ]
    evidence = {
        "completed_control_steps": completed,
        "completed_sim_steps": completed_substeps,
        "fell": bool(traj_data["fell"]),
        "failure_reason": str(traj_data["failure_reason"]),
        "terminated_count": int(
            np.count_nonzero(traj_data["terminated_ctrl"])
        ),
        "truncated_count": int(np.count_nonzero(traj_data["truncated_ctrl"])),
        "non_foot_ground_contact_substep_count": int(
            np.count_nonzero(traj_data["non_foot_ground_contact_substeps"])
        ),
        "action_clipping": {
            "element_fraction": (
                float(np.mean(clipping_mask)) if clipping_mask.size else 0.0
            ),
            "max_magnitude": (
                float(np.max(clipping_magnitude))
                if clipping_magnitude.size
                else 0.0
            ),
        },
        "torque_saturation": {
            "element_fraction": (
                float(np.mean(saturation_mask)) if saturation_mask.size else 0.0
            ),
            "max_requested_torque_overshoot_nm": (
                float(np.max(saturation_magnitude))
                if saturation_magnitude.size
                else 0.0
            ),
        },
        "solver_all_finite": bool(
            np.asarray(traj_data["solver_finite_ctrl"]).size == completed
            and np.all(traj_data["solver_finite_ctrl"])
        ),
    }
    if hold_contacts.size:
        evidence["measured_contact"] = {
            "hold_substeps_observed": int(hold_contacts.shape[0]),
            "front_pair_agreement": float(
                np.mean(hold_contacts[:, 0] == hold_contacts[:, 1])
            ),
            "rear_pair_agreement": float(
                np.mean(hold_contacts[:, 2] == hold_contacts[:, 3])
            ),
            "front_only_fraction": float(
                np.mean(np.all(hold_contacts == [1, 1, 0, 0], axis=1))
            ),
            "rear_only_fraction": float(
                np.mean(np.all(hold_contacts == [0, 0, 1, 1], axis=1))
            ),
        }
    if qpos.size:
        from scipy.spatial.transform import Rotation

        euler = Rotation.from_quat(np.roll(qpos[:, 3:7], -1, axis=1)).as_euler(
            "xyz"
        )
        evidence["posture"] = {
            "min_base_height_m": float(np.min(qpos[:, 2])),
            "max_abs_roll_rad": float(np.max(np.abs(euler[:, 0]))),
            "max_abs_pitch_rad": float(np.max(np.abs(euler[:, 1]))),
        }
    return evidence


def _compare_regeneration(expected: dict, actual: dict) -> dict:
    ignored = set()
    missing = sorted(set(expected) ^ set(actual))
    mismatched = []
    for key in sorted(set(expected) & set(actual) - ignored):
        expected_value = np.asarray(expected[key])
        actual_value = np.asarray(actual[key])
        if expected_value.shape != actual_value.shape or not np.array_equal(
            expected_value, actual_value
        ):
            mismatched.append(key)
    return {
        "passed": not missing and not mismatched,
        "missing_or_extra_fields": missing,
        "mismatched_fields": mismatched,
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_run_artifacts(
    *,
    output_dir: Path,
    attempt_records: list[dict],
    validation_records: list[dict],
    provenance: dict,
    generation_settings: dict,
) -> dict:
    def aggregate_values(values):
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(np.mean(array)) if array.size else None,
            "min": float(np.min(array)) if array.size else None,
            "max": float(np.max(array)) if array.size else None,
            "p05": float(np.percentile(array, 5)) if array.size else None,
            "p95": float(np.percentile(array, 95)) if array.size else None,
        }

    rejection_reasons: dict[str, int] = {}
    for record in attempt_records:
        for reason in record.get("failure_reasons", []):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    accepted = [record for record in attempt_records if record["result"] == "passed"]
    accepted_metrics = [record["metrics"] for record in validation_records]
    metric_aggregates = {}
    if accepted_metrics:
        contact_keys = (
            "front_pair_agreement",
            "rear_pair_agreement",
            "front_only_fraction",
            "rear_only_fraction",
            "all_four_fraction",
            "flight_fraction",
            "diagonal_lateral_only_fraction",
            "paired_interval_alternation_fraction",
            "complete_front_rear_cycles",
        )
        metric_aggregates["contact_per_file"] = {
            key: aggregate_values(
                [metrics["contact"][key] for metrics in accepted_metrics]
            )
            for key in contact_keys
        }
        tracking_fields = (
            "forward_velocity_m_per_s",
            "lateral_velocity_m_per_s",
            "yaw_rate_error_rad_per_s",
        )
        metric_aggregates["tracking_per_file"] = {
            field: {
                statistic: aggregate_values(
                    [
                        metrics["tracking"][field][statistic]
                        for metrics in accepted_metrics
                    ]
                )
                for statistic in ("mean", "signed_bias", "mae", "rmse", "p05", "p95")
            }
            for field in tracking_fields
        }
        metric_aggregates["tracking_per_trajectory"] = {
            key: aggregate_values(
                [metrics["tracking"][key] for metrics in accepted_metrics]
            )
            for key in ("lateral_displacement_m", "accumulated_yaw_drift_rad")
        }
        metric_aggregates["per_gait_cycle_mean_forward_velocity_m_per_s"] = (
            aggregate_values(
                [
                    value
                    for metrics in accepted_metrics
                    for value in metrics["tracking"][
                        "per_gait_cycle_mean_forward_velocity_m_per_s"
                    ]
                ]
            )
        )
        metric_aggregates["posture_per_file"] = {
            key: aggregate_values(
                [metrics["posture"][key] for metrics in accepted_metrics]
            )
            for key in (
                "min_base_height_m",
                "max_abs_roll_rad",
                "max_abs_pitch_rad",
            )
        }
        metric_aggregates["action_clipping_per_file"] = {
            key: aggregate_values(
                [metrics["action_clipping"][key] for metrics in accepted_metrics]
            )
            for key in ("element_fraction", "max_magnitude", "max_abs_raw_action")
        }
        metric_aggregates["torque_saturation_per_file"] = {
            key: aggregate_values(
                [metrics["torque_saturation"][key] for metrics in accepted_metrics]
            )
            for key in ("element_fraction", "max_requested_torque_overshoot_nm")
        }
    checksums = {
        record["output_filename"]: sha256_file(output_dir / record["output_filename"])
        for record in accepted
    }
    checksum_path = output_dir / "checksums.sha256.json"
    _write_json_atomic(
        checksum_path,
        {
            "algorithm": "sha256",
            "files": checksums,
        },
    )
    summary = {
        "schema_version": 1,
        "attempted": len(attempt_records),
        "accepted": len(accepted),
        "rejected": len(attempt_records) - len(accepted),
        "rejection_reasons": rejection_reasons,
        "all_accepted_files_fully_validated": bool(
            accepted
            and len(validation_records) == len(accepted)
            and all(record["passed"] for record in validation_records)
        ),
        "provenance": provenance,
        "normalized_generator_settings": generation_settings,
        "aggregate_metrics": metric_aggregates,
        "checksum_index": checksum_path.name,
    }
    _write_json_atomic(output_dir / "validation_summary.json", summary)
    return summary


def gen_traj_quadruped_dr(
    *,
    num_trajectories=100,
    episode_length=1000,
    start_seed=0,
    output_dir=None,
    max_attempts=None,
    verbose=1,
    render=False,
    domain_rand_config_type=DEFAULT_MPX_TRAJECTORY_DOMAIN_RAND_PRESET,
    dr_seed_offset=1_000_000,
    manifest_filename="generation_manifest.jsonl",
    mpc=None,
    use_go2_sysid=True,
    action_interface_id=DEFAULT_ACTION_INTERFACE_ID,
    action_conversion_mode=None,
    fixed_command=None,
    command_ramp_control_steps=0,
    gait=None,
    duty_factor=None,
    step_frequency_hz=None,
    step_height_m=None,
    mpx_qrot_pitch_cost=None,
    mpx_qomega_pitch_cost=None,
    mpx_robot_height_m=None,
    joint_kp=None,
    joint_kd=None,
    max_pitch=0.5,
    max_roll=0.5,
    min_base_height=0.1,
    acceptance_declaration=None,
    stage="legacy",
    validation_filename="validation_results.jsonl",
    verify_deterministic_regeneration=True,
    pilot_validation_summary=None,
):
    """Generate quadruped MPC trajectories with startup DR matched to the RL env."""
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive or None")
    normalized_declaration = None
    pilot_summary_record = None
    pilot_summary_path = None
    if acceptance_declaration is not None:
        normalized_declaration = validate_acceptance_declaration(
            acceptance_declaration
        )
        if stage not in {"commissioning", "pilot", "production"}:
            raise ValueError(
                "validated bounding generation requires commissioning, pilot, "
                "or production stage"
            )
        if stage in {"pilot", "production"} and not normalized_declaration["frozen"]:
            raise ValueError(f"{stage} generation requires a frozen declaration")
        if stage == "pilot" and num_trajectories != 100:
            raise ValueError("the bounding pilot must request exactly 100 trajectories")
        if max_attempts is None:
            raise ValueError("validated bounding generation requires finite max_attempts")
        minimum_seed, maximum_seed = BOUNDING_STAGE_SEED_RANGES[stage]
        if not minimum_seed <= start_seed <= maximum_seed:
            raise ValueError(
                f"{stage} start_seed must be in the disjoint range "
                f"[{minimum_seed}, {maximum_seed}]"
            )
        last_possible_seed = start_seed + max_attempts - 1
        if last_possible_seed > maximum_seed:
            raise ValueError(
                f"{stage} max_attempts crosses its disjoint seed range"
            )
        if episode_length != int(normalized_declaration["episode_length"]):
            raise ValueError("episode_length does not match acceptance declaration")
        fixed_command = np.asarray(
            normalized_declaration["target_command"], dtype=np.float64
        )
        command_ramp_control_steps = int(
            normalized_declaration["ramp_control_steps"]
        )
        gait = normalized_declaration["gait"]["name"]
        duty_factor = float(normalized_declaration["gait"]["duty_factor"])
        step_frequency_hz = float(
            normalized_declaration["gait"]["step_frequency_hz"]
        )
        step_height_m = float(normalized_declaration["gait"]["step_height_m"])
        controller_declaration = normalized_declaration["controller"]
        joint_kp = controller_declaration["joint_kp_arg"]
        joint_kd = controller_declaration["joint_kd_arg"]
        max_pitch = float(controller_declaration["max_pitch_rad"])
        max_roll = float(controller_declaration["max_roll_rad"])
        min_base_height = float(controller_declaration["min_base_height_m"])
        mpx_qrot_pitch_cost = controller_declaration["mpx_qrot_pitch_cost"]
        mpx_qomega_pitch_cost = controller_declaration[
            "mpx_qomega_pitch_cost"
        ]
        mpx_robot_height_m = controller_declaration["mpx_robot_height_m"]
        domain_rand_config_type = "disabled"
        use_go2_sysid = True
        action_interface_id = MPX_BOUND_ACTION_INTERFACE_ID
        action_conversion_mode = MPX_BOUND_ENV_STEP_MODE
        if stage == "production":
            if pilot_validation_summary is None:
                raise ValueError(
                    "production requires the unchanged passing pilot validation summary"
                )
            if isinstance(pilot_validation_summary, (str, Path)):
                pilot_summary_path = Path(pilot_validation_summary)
                pilot_summary_record = json.loads(
                    pilot_summary_path.read_text(encoding="utf-8")
                )
            else:
                pilot_summary_record = json.loads(
                    json.dumps(pilot_validation_summary, allow_nan=False)
                )
            if (
                int(pilot_summary_record.get("accepted", -1)) != 100
                or not pilot_summary_record.get(
                    "all_accepted_files_fully_validated", False
                )
            ):
                raise ValueError("production pilot summary does not prove 100 passes")
            pilot_settings = pilot_summary_record.get(
                "normalized_generator_settings", {}
            )
            if pilot_settings.get("stage") != "pilot":
                raise ValueError("production evidence is not a pilot summary")
            if pilot_settings.get("acceptance_declaration") != normalized_declaration:
                raise ValueError(
                    "production acceptance declaration differs from the pilot"
                )
        elif pilot_validation_summary is not None:
            raise ValueError("pilot_validation_summary is only valid for production")

    action_interface, action_conversion_mode = _resolve_action_configuration(
        action_interface_id, action_conversion_mode
    )
    resolved_dr_type, _ = resolve_startup_domain_rand_config(domain_rand_config_type)

    if output_dir is None:
        if normalized_declaration is None:
            output_dir = (
                Path(__file__).parent.parent.parent
                / "data"
                / "quadruped_dr"
                / resolved_dr_type
            )
        else:
            output_dir = (
                Path(__file__).parent.parent.parent
                / "data"
                / "quadruped_bound"
                / "nominal_vx0p5"
                / stage
            )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_filename
    validation_path = output_dir / validation_filename
    staging_dir = output_dir / ".staging"
    existing_accepted_count = 0
    if normalized_declaration is not None:
        staging_dir.mkdir(parents=True, exist_ok=True)
        existing_records = _read_jsonl(manifest_path)
        if stage in {"pilot", "production"}:
            for existing_record in existing_records:
                if existing_record.get("acceptance_declaration_id") != (
                    normalized_declaration["declaration_id"]
                ):
                    raise ValueError(
                        f"existing {stage} manifest mixes acceptance declarations"
                    )
            existing_seeds = {int(record["seed"]) for record in existing_records}
            overlapping_seeds = sorted(
                seed
                for seed in existing_seeds
                if start_seed <= seed < start_seed + max_attempts
            )
            if overlapping_seeds:
                raise ValueError(
                    f"{stage} invocation repeats existing attempt seeds; first "
                    f"overlap is {overlapping_seeds[0]}"
                )
        existing_accepted_count = sum(
            record.get("result") == "passed" for record in existing_records
        )
        if existing_accepted_count > num_trajectories:
            raise ValueError(
                f"existing stage already has {existing_accepted_count} accepted "
                f"files, exceeding requested total {num_trajectories}"
            )
    provenance = _repository_provenance()
    if pilot_summary_record is not None:
        if pilot_summary_record.get("provenance") != provenance:
            raise ValueError("repository or submodule provenance changed after the pilot")
        if pilot_summary_path is not None:
            checksum_path = pilot_summary_path.parent / pilot_summary_record[
                "checksum_index"
            ]
            checksum_index = json.loads(checksum_path.read_text(encoding="utf-8"))
            if len(checksum_index.get("files", {})) != 100:
                raise ValueError("pilot checksum index does not cover 100 files")
            for filename, expected_checksum in checksum_index["files"].items():
                pilot_file = pilot_summary_path.parent / filename
                if not pilot_file.is_file() or sha256_file(pilot_file) != expected_checksum:
                    raise ValueError(f"pilot file checksum mismatch: {filename}")

    if verbose > 0:
        print(f"Generating {num_trajectories} valid DR quadruped trajectories")
        print(f"  DR preset: {resolved_dr_type}")
        print(f"  Episode length: {episode_length} ctrl steps")
        print(f"  Starting seed: {start_seed}")
        print(f"  DR seed offset: {dr_seed_offset}")
        print(f"  Max attempts: {'unlimited' if max_attempts is None else max_attempts}")
        print(f"  Output: {output_dir}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Action interface: {action_interface.interface_id}")
        print(f"  Action conversion: {action_conversion_mode}")
        if fixed_command is not None:
            print(f"  Fixed target command: {np.asarray(fixed_command).tolist()}")
            print(f"  Command ramp steps: {command_ramp_control_steps}")
            print(
                "  Gait: "
                f"{gait}, duty={duty_factor}, frequency={step_frequency_hz}, "
                f"height={step_height_m}"
            )
        if normalized_declaration is not None:
            print(f"  Acceptance declaration: {normalized_declaration['declaration_id']}")
            print(f"  Stage: {stage}")
        print()

    shared_mpc = mpc
    if shared_mpc is None:
        if verbose > 0:
            print("Pre-compiling JAX MPC kernels (once for all trajectories)...")
        controller_config = _controller_config_with_overrides(
            mpx_qrot_pitch_cost,
            mpx_qomega_pitch_cost,
            mpx_robot_height_m,
        )
        shared_mpc = mpc_wrapper.MPCControllerWrapper(
            controller_config,
            use_go2_sysid=use_go2_sysid,
            gait=gait,
            duty_factor=duty_factor,
            step_frequency_hz=step_frequency_hz,
            step_height_m=step_height_m,
            enable_planned_contact_diagnostics=(fixed_command is not None),
        )
        shared_mpc.robot_height = controller_config.robot_height
        dummy_qpos = np.concatenate(
            [np.array(config.p0), np.array(config.quat0), np.array(config.q0)]
        )
        dummy_qvel = np.zeros(config.n_joints + 6)
        dummy_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
        dummy_contact = np.zeros(config.n_contact)
        shared_mpc.reset(dummy_qpos, dummy_qvel)
        compile_start = timer()
        shared_mpc.run(dummy_qpos, dummy_qvel, dummy_input, dummy_contact)
        if verbose > 0:
            print(f"JIT compilation done in {timer() - compile_start:.1f}s")
            print()

    _validate_requested_gait(
        shared_mpc,
        gait=gait,
        duty_factor=duty_factor,
        step_frequency_hz=step_frequency_hz,
        step_height_m=step_height_m,
    )

    saved = existing_accepted_count
    attempt = 0
    attempt_records = []
    validation_records = []
    regeneration_checked = False
    while saved < num_trajectories:
        if max_attempts is not None and attempt >= max_attempts:
            print(
                f"Warning: reached max_attempts={max_attempts} after saving "
                f"{saved}/{num_trajectories} trajectories. Stopping."
            )
            break

        seed = start_seed + attempt
        attempt += 1

        if verbose > 0:
            print(f"--- Attempt {attempt} | Saved {saved}/{num_trajectories} (seed={seed}) ---")

        traj_data = generate_trajectory(
            seed=seed,
            domain_rand_config_type=resolved_dr_type,
            dr_seed_offset=dr_seed_offset,
            mpc=shared_mpc,
            episode_length=episode_length,
            verbose=verbose,
            render=render,
            use_go2_sysid=use_go2_sysid,
            fixed_command=fixed_command,
            command_ramp_control_steps=command_ramp_control_steps,
            gait=gait,
            duty_factor=duty_factor,
            step_frequency_hz=step_frequency_hz,
            step_height_m=step_height_m,
            mpx_qrot_pitch_cost=mpx_qrot_pitch_cost,
            mpx_qomega_pitch_cost=mpx_qomega_pitch_cost,
            mpx_robot_height_m=mpx_robot_height_m,
            joint_kp=joint_kp,
            joint_kd=joint_kd,
            max_pitch=max_pitch,
            max_roll=max_roll,
            min_base_height=min_base_height,
            action_interface_id=action_interface.interface_id,
            action_conversion_mode=action_conversion_mode,
            provenance=provenance,
        )

        base_manifest_record = {
            "attempt_index": int(attempt),
            "seed": int(seed),
            "dr_seed": int(traj_data["dr_seed"]),
            "fell": bool(traj_data["fell"]),
            "failure_reason": str(traj_data["failure_reason"]),
            "completed_control_steps": int(traj_data["completed_control_steps"]),
            "completed_sim_steps": int(traj_data["completed_sim_steps"]),
            "controller_delay_steps": int(traj_data["controller_delay_steps"]),
            "controller_delay_s": float(traj_data["controller_delay_s"]),
            "action_conversion_mode": str(
                traj_data["action_conversion_mode"]
            ),
            "action_interface_id": str(traj_data["action_interface_id"]),
            "dr_summary": _summarize_dr_bundle(
                int(traj_data["base_body_id"]),
                extract_dr_bundle_from_traj(traj_data),
                summary_geom_friction=float(traj_data["dr_summary_geom_friction"]),
            ),
            "generation_settings": json.loads(
                str(traj_data["generation_settings_json"])
            ),
            "provenance": provenance,
            "measured_evidence": _attempt_evidence(
                traj_data,
                ramp_control_steps=command_ramp_control_steps,
            ),
        }
        if normalized_declaration is None:
            legacy_passed = bool(
                (not traj_data["fell"])
                and (traj_data["completed_control_steps"] == episode_length)
            )
            legacy_failure_reasons = []
            if not legacy_passed:
                legacy_failure_reasons.append(
                    str(traj_data["failure_reason"] or "incomplete_horizon")
                )
            manifest_record = {
                **base_manifest_record,
                "success": legacy_passed,
                "passed": legacy_passed,
                "result": "passed" if legacy_passed else "failed",
                "failure_reasons": legacy_failure_reasons,
            }
            _append_manifest_record(manifest_path, manifest_record)

            if traj_data["fell"] or traj_data["completed_control_steps"] != episode_length:
                if verbose > 0:
                    print(
                        f"  [Seed {seed}] Attempt rejected for training data "
                        f"(reason={traj_data['failure_reason'] or 'incomplete'})"
                    )
                    print()
                continue

            filename = (
                _bounding_filename(traj_data, seed, episode_length)
                if fixed_command is not None
                else (
                    f"quadruped_dr_{resolved_dr_type}_seed_{seed:06d}_"
                    f"ep_{episode_length}.npz"
                )
            )
            filepath = output_dir / filename
            if filepath.exists():
                raise FileExistsError(f"refusing to overwrite existing trajectory {filepath}")
            np.savez_compressed(filepath, **traj_data)
            saved += 1
            if verbose > 0:
                print(f"  Saved ({saved}/{num_trajectories}): {filepath}")
                print()
            continue

        filename = _bounding_filename(traj_data, seed, episode_length)
        filepath = output_dir / filename
        staging_path = staging_dir / filename
        if filepath.exists() or staging_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing trajectory or staging file for {filename}"
            )

        pre_replay_validation = validate_bounding_trajectory_data(
            traj_data, normalized_declaration
        )
        blocking_failures = [
            reason
            for reason in pre_replay_validation["failure_reasons"]
            if reason != "full_saved_action_replay_not_run"
        ]
        validation = pre_replay_validation
        regeneration = None
        if not blocking_failures:
            with staging_path.open("xb") as stream:
                np.savez_compressed(stream, **traj_data)
                stream.flush()
                os.fsync(stream.fileno())
            from mpc_rl.planner.check_mpx_transition_parity import (
                validate_bounding_trajectory_file,
            )

            validation = validate_bounding_trajectory_file(
                staging_path, normalized_declaration
            )
            if validation["passed"] and verify_deterministic_regeneration and not regeneration_checked:
                regenerated = generate_trajectory(
                    seed=seed,
                    domain_rand_config_type=resolved_dr_type,
                    dr_seed_offset=dr_seed_offset,
                    mpc=shared_mpc,
                    episode_length=episode_length,
                    verbose=verbose,
                    render=False,
                    use_go2_sysid=use_go2_sysid,
                    fixed_command=fixed_command,
                    command_ramp_control_steps=command_ramp_control_steps,
                    gait=gait,
                    duty_factor=duty_factor,
                    step_frequency_hz=step_frequency_hz,
                    step_height_m=step_height_m,
                    mpx_qrot_pitch_cost=mpx_qrot_pitch_cost,
                    mpx_qomega_pitch_cost=mpx_qomega_pitch_cost,
                    mpx_robot_height_m=mpx_robot_height_m,
                    joint_kp=joint_kp,
                    joint_kd=joint_kd,
                    max_pitch=max_pitch,
                    max_roll=max_roll,
                    min_base_height=min_base_height,
                    action_interface_id=action_interface.interface_id,
                    action_conversion_mode=action_conversion_mode,
                    provenance=provenance,
                )
                regeneration = _compare_regeneration(traj_data, regenerated)
                regeneration_checked = True
                if not regeneration["passed"]:
                    validation["passed"] = False
                    validation["failure_reasons"].append(
                        "deterministic_regeneration_failed"
                    )

        if validation["passed"]:
            os.replace(staging_path, filepath)
            saved += 1
        elif staging_path.exists():
            staging_path.unlink()

        validation_record = {
            "seed": int(seed),
            "filename": filename,
            "passed": bool(validation["passed"]),
            "failure_reasons": validation["failure_reasons"],
            "threshold_checks": validation.get("threshold_checks"),
            "metrics": validation.get("metrics"),
            "replay": validation.get("replay"),
            "deterministic_regeneration": regeneration,
        }
        if not blocking_failures:
            _append_manifest_record(validation_path, validation_record)
            if validation["passed"]:
                validation_records.append(validation_record)

        manifest_record = {
            **base_manifest_record,
            "stage": stage,
            "acceptance_declaration_id": normalized_declaration[
                "declaration_id"
            ],
            "result": "passed" if validation["passed"] else "failed",
            "passed": bool(validation["passed"]),
            "failure_reasons": validation["failure_reasons"],
            "threshold_checks": validation.get("threshold_checks"),
            "validation_metrics": validation.get("metrics"),
            "output_filename": filename if validation["passed"] else None,
            "validation_filename": (
                validation_path.name if not blocking_failures else None
            ),
        }
        _append_manifest_record(manifest_path, manifest_record)
        attempt_records.append(manifest_record)

        if verbose > 0:
            if validation["passed"]:
                print(f"  Saved ({saved}/{num_trajectories}): {filepath}")
            else:
                print(
                    f"  [Seed {seed}] FAILED: "
                    + ", ".join(validation["failure_reasons"])
                )
            print()

    summary = None
    if normalized_declaration is not None:
        generation_settings = {
            "acceptance_declaration": normalized_declaration,
            "action_conversion_mode": action_conversion_mode,
            "action_interface_id": action_interface.interface_id,
            "joint_kp": joint_kp,
            "joint_kd": joint_kd,
            "max_pitch": max_pitch,
            "max_roll": max_roll,
            "min_base_height": min_base_height,
            "mpx_qrot_pitch_cost": mpx_qrot_pitch_cost,
            "mpx_qomega_pitch_cost": mpx_qomega_pitch_cost,
            "mpx_robot_height_m": mpx_robot_height_m,
            "stage": stage,
            "start_seed": int(start_seed),
        }
        all_attempt_records = _read_jsonl(manifest_path)
        all_validation_records = [
            record for record in _read_jsonl(validation_path) if record["passed"]
        ]
        summary = _write_run_artifacts(
            output_dir=output_dir,
            attempt_records=all_attempt_records,
            validation_records=all_validation_records,
            provenance=provenance,
            generation_settings=generation_settings,
        )

    if verbose > 0:
        print(
            f"Done! Saved {saved} valid trajectories in {output_dir} "
            f"({attempt} attempts executed in this invocation)"
        )
    return summary


def extract_dr_bundle_from_traj(traj_data: dict) -> dict:
    """Return only the saved DR bundle fields from a trajectory dict."""
    return {
        key: traj_data[key]
        for key in traj_data
        if key.startswith("dr_")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPX quadruped trajectories with RL-matched startup domain randomization"
    )
    parser.add_argument(
        "--num-trajectories",
        "-n",
        type=int,
        default=100,
        help="Number of trajectories to generate (default: 100)",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="Episode length in control steps at 50 Hz (default: 1000 = 20s)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting random seed (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/quadruped_dr/<preset>/)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10000,
        help="Maximum total attempts before stopping (default: 10000)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (default: 1)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a MuJoCo viewer window to display each trajectory attempt in real time",
    )
    parser.add_argument(
        "--domain-rand-config-type",
        type=str,
        default=DEFAULT_MPX_TRAJECTORY_DOMAIN_RAND_PRESET,
        choices=STARTUP_DOMAIN_RAND_PRESET_NAMES,
        help="Startup domain-randomization preset to apply to each trajectory",
    )
    parser.add_argument(
        "--dr-seed-offset",
        type=int,
        default=1_000_000,
        help="Offset added to the rollout seed for the independent DR RNG stream",
    )
    parser.add_argument(
        "--manifest-filename",
        type=str,
        default="generation_manifest.jsonl",
        help="Per-process manifest filename inside the output directory",
    )
    parser.add_argument(
        "--target-command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        default=None,
        help="Fixed target [vx, vy, wz] command; legacy default samples commands",
    )
    parser.add_argument(
        "--command-ramp-control-steps",
        type=int,
        default=0,
        help="Linear ramp steps before holding the fixed target (default: 0)",
    )
    parser.add_argument(
        "--gait",
        choices=BOUNDING_GAITS,
        default=None,
        help="Instance-owned named bounding gait",
    )
    parser.add_argument("--duty-factor", type=float, default=None)
    parser.add_argument("--step-frequency-hz", type=float, default=None)
    parser.add_argument("--step-height-m", type=float, default=None)
    parser.add_argument(
        "--mpx-qrot-pitch-cost",
        type=float,
        default=None,
        help="Optional instance-local MPX body-pitch orientation cost",
    )
    parser.add_argument(
        "--mpx-qomega-pitch-cost",
        type=float,
        default=None,
        help="Optional instance-local MPX body pitch-rate cost",
    )
    parser.add_argument(
        "--mpx-robot-height-m",
        type=float,
        default=None,
        help="Optional instance-local MPX body-height reference in meters",
    )
    parser.add_argument(
        "--joint-kp",
        type=float,
        default=None,
        help="Optional scalar environment joint proportional gain",
    )
    parser.add_argument(
        "--joint-kd",
        type=float,
        default=None,
        help="Optional scalar environment joint derivative gain",
    )
    parser.add_argument("--max-pitch", type=float, default=0.5)
    parser.add_argument("--max-roll", type=float, default=0.5)
    parser.add_argument("--min-base-height", type=float, default=0.1)
    acceptance_group = parser.add_mutually_exclusive_group()
    acceptance_group.add_argument(
        "--commissioning-acceptance",
        action="store_true",
        help="Apply the plan thresholds as an explicitly unfrozen commissioning predicate",
    )
    acceptance_group.add_argument(
        "--acceptance-declaration",
        type=str,
        default=None,
        help="Frozen acceptance declaration JSON required for pilot/production",
    )
    parser.add_argument(
        "--stage",
        choices=("legacy", "commissioning", "pilot", "production"),
        default="legacy",
    )
    parser.add_argument(
        "--validation-filename",
        type=str,
        default="validation_results.jsonl",
    )
    parser.add_argument(
        "--pilot-validation-summary",
        type=str,
        default=None,
        help="Passing unchanged 100-file pilot summary required for production",
    )
    parser.add_argument(
        "--no-deterministic-regeneration-check",
        dest="verify_deterministic_regeneration",
        action="store_false",
        default=True,
        help="Disable first-accepted-seed exact regeneration (diagnostics only)",
    )
    parser.add_argument(
        "--action-interface",
        dest="action_interface_id",
        choices=tuple(ACTION_INTERFACES),
        default=DEFAULT_ACTION_INTERFACE_ID,
        help=(
            "Versioned quadruped action interface. The default preserves the "
            "existing 0.5-radian/5-Hz behavior; the MPX bound interface is "
            "opt-in."
        ),
    )
    parser.add_argument(
        "--action-conversion-mode",
        choices=ACTION_CONVERSION_MODES,
        default=None,
        help=(
            "Optional explicit MPX-to-rollout conversion mode. When omitted, "
            "the selected action interface chooses its required mode; the "
            "legacy interface retains the historical direct-torque default."
        ),
    )
    parser.add_argument(
        "--use_go2_sysid",
        dest="use_go2_sysid",
        action="store_true",
        default=True,
        help="Enable the Go2 sysID joint-dynamics patch in the RL env and MPX controller (default: enabled)",
    )
    parser.add_argument(
        "--no_use_go2_sysid",
        dest="use_go2_sysid",
        action="store_false",
        help="Disable the Go2 sysID joint-dynamics patch in the RL env and MPX controller",
    )
    
    args = parser.parse_args()

    acceptance_declaration = None
    if args.acceptance_declaration is not None:
        acceptance_declaration = json.loads(
            Path(args.acceptance_declaration).read_text(encoding="utf-8")
        )
    elif args.commissioning_acceptance:
        required = {
            "target_command": args.target_command,
            "gait": args.gait,
            "duty_factor": args.duty_factor,
            "step_frequency_hz": args.step_frequency_hz,
            "step_height_m": args.step_height_m,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            parser.error(
                "--commissioning-acceptance requires explicit " + ", ".join(missing)
            )
        acceptance_declaration = make_commissioning_acceptance_declaration(
            target_command=args.target_command,
            episode_length=args.episode_length,
            ramp_control_steps=args.command_ramp_control_steps,
            gait_name=args.gait,
            gait_duty_factor=args.duty_factor,
            gait_step_frequency_hz=args.step_frequency_hz,
            gait_step_height_m=args.step_height_m,
            joint_kp=args.joint_kp,
            joint_kd=args.joint_kd,
            max_pitch_rad=args.max_pitch,
            max_roll_rad=args.max_roll,
            min_base_height_m=args.min_base_height,
            mpx_qrot_pitch_cost=args.mpx_qrot_pitch_cost,
            mpx_qomega_pitch_cost=args.mpx_qomega_pitch_cost,
            mpx_robot_height_m=args.mpx_robot_height_m,
        )

    gen_traj_quadruped_dr(
        num_trajectories=args.num_trajectories,
        episode_length=args.episode_length,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
        max_attempts=args.max_attempts,
        verbose=args.verbose,
        render=args.render,
        domain_rand_config_type=args.domain_rand_config_type,
        dr_seed_offset=args.dr_seed_offset,
        manifest_filename=args.manifest_filename,
        use_go2_sysid=args.use_go2_sysid,
        action_interface_id=args.action_interface_id,
        action_conversion_mode=args.action_conversion_mode,
        fixed_command=args.target_command,
        command_ramp_control_steps=args.command_ramp_control_steps,
        gait=args.gait,
        duty_factor=args.duty_factor,
        step_frequency_hz=args.step_frequency_hz,
        step_height_m=args.step_height_m,
        mpx_qrot_pitch_cost=args.mpx_qrot_pitch_cost,
        mpx_qomega_pitch_cost=args.mpx_qomega_pitch_cost,
        mpx_robot_height_m=args.mpx_robot_height_m,
        joint_kp=args.joint_kp,
        joint_kd=args.joint_kd,
        max_pitch=args.max_pitch,
        max_roll=args.max_roll,
        min_base_height=args.min_base_height,
        acceptance_declaration=acceptance_declaration,
        stage=args.stage,
        validation_filename=args.validation_filename,
        verify_deterministic_regeneration=args.verify_deterministic_regeneration,
        pilot_validation_summary=args.pilot_validation_summary,
    )


if __name__ == "__main__":
    main()
