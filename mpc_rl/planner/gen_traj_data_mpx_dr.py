import argparse
import json
import time
from pathlib import Path
from timeit import default_timer as timer

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


def _configure_mpc_duty_factor(commands: np.ndarray, mpc) -> float:
    """Match the nominal MPX standing-vs-trotting duty-factor heuristic."""
    total_command = np.linalg.norm(commands[:2]) + abs(commands[2])
    if total_command < COMMAND_THRESHOLD:
        config.duty_factor = 1.0
        mpc.duty_factor = 1.0
    else:
        config.duty_factor = 0.5
        mpc.duty_factor = 0.5
    return float(total_command)


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
    action_interface_id=DEFAULT_ACTION_INTERFACE_ID,
    action_conversion_mode=None,
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
        if fixed_command is None:
            commands = sample_commands(rng)
        else:
            commands = np.asarray(fixed_command, dtype=np.float64)
            if commands.shape != (3,) or not np.all(np.isfinite(commands)):
                raise ValueError("fixed_command must be a finite vector with shape (3,)")
            commands = commands.copy()
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
            mpc = mpc_wrapper.MPCControllerWrapper(
                config,
                use_go2_sysid=use_go2_sysid,
            )
            mpc.robot_height = config.robot_height
            if verbose > 0:
                print(f"[Seed {seed}] Pre-compiling JAX MPC kernels...")
            dummy_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
            dummy_contact = np.zeros(config.n_contact)
            compile_start = timer()
            mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
            mpc.run(env.mjData.qpos.copy(), env.mjData.qvel.copy(), dummy_input, dummy_contact)
            if verbose > 0:
                print(f"[Seed {seed}] JIT compilation done in {timer() - compile_start:.1f}s")

        mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
        gait_metadata = _controller_gait_metadata(mpc)
        total_command = _configure_mpc_duty_factor(commands, mpc)

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
            if (
                fixed_command is None
                and steps_since_resample >= env.command_resample_interval
                and ctrl_step > 0
            ):
                commands = sample_commands(rng)
                steps_since_resample = 0
                total_command = _configure_mpc_duty_factor(commands, mpc)
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
            if getattr(mpc, "enable_planned_contact_diagnostics", False):
                planned_contact_schedule_ctrl_traj.append(
                    np.asarray(mpc.planned_contact_schedule, dtype=np.int8)
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
            "fixed_command": (
                np.asarray(commands, dtype=np.float64)
                if fixed_command is not None
                else np.zeros(3, dtype=np.float64)
            ),
            **gait_metadata,
            "action_conversion_mode": action_conversion_mode,
            "controller_delay_steps": 0,
            "controller_delay_s": 0.0,
            "go2_sysid_expected_vector": _go2_sysid_signature_vector(),
            "go2_sysid_enabled": bool(use_go2_sysid),
            **dr_bundle,
        }
    finally:
        env.close()


def _append_manifest_record(manifest_path: Path, record: dict):
    """Append one generation attempt record to the JSONL manifest."""
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(record, sort_keys=True) + "\n")


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
):
    """Generate quadruped MPC trajectories with startup DR matched to the RL env."""
    action_interface, action_conversion_mode = _resolve_action_configuration(
        action_interface_id, action_conversion_mode
    )
    resolved_dr_type, _ = resolve_startup_domain_rand_config(domain_rand_config_type)

    if output_dir is None:
        output_dir = (
            Path(__file__).parent.parent.parent / "data" / "quadruped_dr" / resolved_dr_type
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_filename

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
        print()

    shared_mpc = mpc
    if shared_mpc is None:
        if verbose > 0:
            print("Pre-compiling JAX MPC kernels (once for all trajectories)...")
        shared_mpc = mpc_wrapper.MPCControllerWrapper(
            config,
            use_go2_sysid=use_go2_sysid,
        )
        shared_mpc.robot_height = config.robot_height
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

    saved = 0
    attempt = 0
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
            action_interface_id=action_interface.interface_id,
            action_conversion_mode=action_conversion_mode,
        )

        manifest_record = {
            "seed": int(seed),
            "dr_seed": int(traj_data["dr_seed"]),
            "success": bool(
                (not traj_data["fell"])
                and (traj_data["completed_control_steps"] == episode_length)
            ),
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

        filename = f"quadruped_dr_{resolved_dr_type}_seed_{seed:06d}_ep_{episode_length}.npz"
        filepath = output_dir / filename
        np.savez_compressed(filepath, **traj_data)
        saved += 1

        if verbose > 0:
            print(f"  Saved ({saved}/{num_trajectories}): {filepath}")
            print()

    if verbose > 0:
        print(
            f"Done! Saved {saved} valid DR trajectories in {output_dir} "
            f"({attempt - saved} rejected attempts logged to manifest)"
        )


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
    )


if __name__ == "__main__":
    main()
