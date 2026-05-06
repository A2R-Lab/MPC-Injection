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


def _geom_body_label(mj_model: mujoco.MjModel, geom_id: int) -> str:
    """Return a compact geom/body label for contact rejection messages."""
    geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    body_id = int(mj_model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    geom_label = geom_name if geom_name else f"geom_{geom_id}"
    body_label = body_name if body_name else f"body_{body_id}"
    return f"{geom_label}({body_label})"


def _body_is_descendant_of(
    mj_model: mujoco.MjModel,
    body_id: int,
    root_body_id: int,
) -> bool:
    """Return whether body_id is root_body_id or belongs to its body subtree."""
    while body_id != 0:
        if body_id == root_body_id:
            return True
        body_id = int(mj_model.body_parentid[body_id])
    return False


def _find_non_foot_ground_contact(
    env: QuadrupedVelocityTrackingEnv,
) -> tuple[str, str] | None:
    """Return contact details if a non-foot geom touches a world-body ground geom."""
    for contact_idx in range(env.mjData.ncon):
        contact = env.mjData.contact[contact_idx]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(env.mjModel.geom_bodyid[geom1])
        body2 = int(env.mjModel.geom_bodyid[geom2])

        if body1 == 0 and body2 == 0:
            continue
        if body1 != 0 and body2 != 0:
            continue

        ground_geom = geom1 if body1 == 0 else geom2
        robot_geom = geom2 if body1 == 0 else geom1
        if robot_geom in env._foot_geom_id_set:
            continue

        robot_body_id = int(env.mjModel.geom_bodyid[robot_geom])
        if not _body_is_descendant_of(
            env.mjModel, robot_body_id, int(env._base_body_id)
        ):
            continue

        robot_body_name = mujoco.mj_id2name(
            env.mjModel, mujoco.mjtObj.mjOBJ_BODY, robot_body_id
        )
        detail = (
            f"{_geom_body_label(env.mjModel, robot_geom)} touched "
            f"{_geom_body_label(env.mjModel, ground_geom)}"
        )
        return (robot_body_name or f"body_{robot_body_id}", detail)

    return None


def _inverse_pd_residual_action(env: QuadrupedVelocityTrackingEnv, tau_applied_first: np.ndarray) -> np.ndarray:
    """Recover the RL residual action whose PD torques match the first applied torque."""
    q_current = env.mjData.qpos[7 : 7 + env.num_joints].copy()
    dq_current = env.mjData.qvel[6 : 6 + env.num_joints].copy()
    q_target = q_current + (tau_applied_first + env.kd * dq_current) / env.kp
    action = (q_target - env.default_joint_pos) / env.action_scale
    return np.clip(action, -1.0, 1.0).astype(np.float64)


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

        non_foot_ground_contact = _find_non_foot_ground_contact(env)
        if non_foot_ground_contact is not None:
            break

        if viewer is not None:
            viewer.sync()
            time.sleep(env.sim_dt)
            if not viewer.is_running():
                viewer_closed = True
                break

    env._step_count += 1
    env._steps_since_command_resample += 1
    env._maybe_push_robot()
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
):
    """Generate one MPX-controlled trajectory inside the RL quadruped env."""
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
    )

    own_mpc = mpc is None
    try:
        dr_bundle = env.sample_startup_domain_rand_bundle(
            rng=dr_rng,
            dr_config_type=resolved_dr_type,
            dr_seed=dr_seed,
        )
        env.apply_startup_domain_rand_bundle(dr_bundle)
        commands = sample_commands(rng)
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
        actions_traj = []
        rewards_traj = []
        terminated_ctrl_traj = []
        commands_ctrl_traj = []

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

        viewer = None
        if render:
            import mujoco.viewer as mjviewer

            viewer = mjviewer.launch_passive(env.mjModel, env.mjData)

        mpc_solve_times = []
        fell = False
        failure_reason = ""
        completed_control_steps = 0

        for ctrl_step in range(episode_length):
            if steps_since_resample >= env.command_resample_interval and ctrl_step > 0:
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

            sim_idx_start = ctrl_step * sim_steps_per_ctrl
            q_current = env.mjData.qpos[7 : 7 + n_joints].copy()
            dq_current = env.mjData.qvel[6 : 6 + n_joints].copy()
            tau_feedback_first = 10.0 * (q_des - q_current) - 2.0 * dq_current
            tau_applied_first = np.clip(
                tau_mpx + tau_feedback_first,
                env.torque_limits[:, 0],
                env.torque_limits[:, 1],
            )
            action = _inverse_pd_residual_action(env, tau_applied_first)
            actions_traj.append(action.copy())

            (
                next_obs,
                reward,
                terminated,
                _,
                viewer_closed,
                non_foot_ground_contact,
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

            next_policy_obs_traj.append(next_obs["policy"].copy())
            next_privileged_obs_traj.append(next_obs["privileged"].copy())
            rewards_traj.append(reward)
            terminated_ctrl_traj.append(terminated)
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
            "actions": np.asarray(actions_traj, dtype=np.float64),
            "rewards": np.asarray(rewards_traj, dtype=np.float32),
            "terminated_ctrl": np.asarray(terminated_ctrl_traj, dtype=bool),
            "commands_ctrl": np.asarray(commands_ctrl_traj, dtype=np.float64),
            "seed": seed,
            "dr_seed": dr_seed,
            "base_body_id": int(env._base_body_id),
            "dr_summary_geom_friction": float(dr_summary_geom_friction),
            "default_joint_pos": env.default_joint_pos.copy(),
            "action_scale": float(env.action_scale),
            "sim_dt": float(env.sim_dt),
            "control_dt": float(env.control_dt),
            "episode_length": int(episode_length),
            "completed_control_steps": int(completed_control_steps),
            "completed_sim_steps": int(completed_control_steps * sim_steps_per_ctrl),
            "fell": bool(fell),
            "failure_reason": failure_reason,
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
):
    """Generate quadruped MPC trajectories with startup DR matched to the RL env."""
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
    )


if __name__ == "__main__":
    main()
