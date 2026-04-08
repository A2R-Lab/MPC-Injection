import argparse
import time
from pathlib import Path
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
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

from gym_quadruped.quadruped_env import QuadrupedEnv

import mpx.config.config_go2 as config
import mpx.utils.mpc_wrapper as mpc_wrapper
from mpc_rl.envs.domain_randomization import (
    DEFAULT_STARTUP_DOMAIN_RAND_PRESET,
    STARTUP_DOMAIN_RAND_PRESET_NAMES,
    apply_startup_domain_rand_patch,
    resolve_startup_domain_rand_config,
    sample_startup_domain_rand_patch,
)
from mpc_rl.planner.gen_traj_data_mpx import (
    COMMAND_THRESHOLD,
    RL_ACTION_SCALE,
    RL_COMMAND_RESAMPLE_INTERVAL,
    RL_CONTROL_DT,
    _check_fell,
    randomize_initial_state,
    sample_commands,
)


try:
    gpu_device = jax.devices("gpu")[0]
except RuntimeError:
    gpu_device = jax.devices("cpu")[0]
jax.default_device(gpu_device)


def _quadruped_base_body_id(mj_model: mujoco.MjModel) -> int:
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    return base_id if base_id >= 0 else 1


def generate_trajectory(
    seed,
    *,
    domain_rand_config_type=DEFAULT_STARTUP_DOMAIN_RAND_PRESET,
    dr_seed_offset=1_000_000,
    mpc=None,
    episode_length=1000,
    verbose=1,
    render=False,
):
    """Generate one MPX-controlled trajectory on a startup-randomized plant."""
    rng = np.random.RandomState(seed)
    dr_seed = seed + dr_seed_offset
    dr_rng = np.random.RandomState(dr_seed)
    resolved_dr_type, domain_rand_cfg = resolve_startup_domain_rand_config(
        domain_rand_config_type
    )

    n_joints = config.n_joints
    sim_frequency = 200.0
    mpc_frequency = config.mpc_frequency
    sim_steps_per_ctrl = int(sim_frequency / mpc_frequency)
    total_sim_steps = episode_length * sim_steps_per_ctrl

    robot_feet_geom_names = dict(FR="FR", FL="FL", RR="RR", RL="RL")
    env = QuadrupedEnv(
        robot="go2",
        scene="flat",
        sim_dt=1 / sim_frequency,
        ref_base_lin_vel=0.0,
        ground_friction_coeff=0.7,
        base_vel_command_type="human",
        state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
    )

    dr_patch = sample_startup_domain_rand_patch(
        env.mjModel,
        domain_rand_cfg,
        rng=dr_rng,
        dr_config_type=resolved_dr_type,
        dr_seed=dr_seed,
        base_body_id=_quadruped_base_body_id(env.mjModel),
    )
    torque_limits = apply_startup_domain_rand_patch(env.mjModel, env.mjData, dr_patch)

    env.reset(random=False)

    default_joint_pos = env.mjModel.key_qpos[0, 7 : 7 + n_joints].copy()
    randomize_initial_state(env, rng)

    own_mpc = mpc is None
    if own_mpc:
        mpc = mpc_wrapper.MPCControllerWrapper(config)
        mpc.robot_height = config.robot_height
        if verbose > 0:
            print(f"[Seed {seed}] Pre-compiling JAX MPC kernels...")
        _jit_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
        _jit_contact_temp, _ = env.feet_contact_state()
        _jit_contact = np.array(
            [_jit_contact_temp[robot_feet_geom_names[leg]] for leg in ["FL", "FR", "RL", "RR"]]
        )
        _start_compile = timer()
        mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
        mpc.run(env.mjData.qpos.copy(), env.mjData.qvel.copy(), _jit_input, _jit_contact)
        if verbose > 0:
            print(f"[Seed {seed}] JIT compilation done in {timer() - _start_compile:.1f}s")

    mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())

    tau = jnp.zeros(n_joints)
    q_des = config.q0.copy()
    dq_des = jnp.zeros(n_joints)

    nq = env.mjModel.nq
    nv = env.mjModel.nv
    qpos_traj = np.zeros((nq, total_sim_steps + 1))
    qvel_traj = np.zeros((nv, total_sim_steps + 1))
    tau_applied_traj = np.zeros((n_joints, total_sim_steps))
    tau_mpx_traj = np.zeros((n_joints, total_sim_steps))
    q_des_traj = np.zeros((n_joints, total_sim_steps))
    time_traj = np.zeros(total_sim_steps + 1)
    commands_traj = np.zeros((3, total_sim_steps))

    qpos_traj[:, 0] = env.mjData.qpos.copy()
    qvel_traj[:, 0] = env.mjData.qvel.copy()
    time_traj[0] = env.mjData.time

    commands = sample_commands(rng)
    total_cmd = np.linalg.norm(commands[:2]) + abs(commands[2])
    print("total command: ", total_cmd)
    if total_cmd < COMMAND_THRESHOLD:
        config.duty_factor = 1.0
        mpc.duty_factor = 1.0
    else:
        config.duty_factor = 0.5
        mpc.duty_factor = 0.5

    steps_since_resample = 0

    if verbose > 0:
        print(f"[Seed {seed}] Starting DR trajectory generation")
        print(f"  DR preset: {resolved_dr_type}")
        print(f"  DR applied fields: {dr_patch['dr_applied_fields'].tolist()}")
        print(f"  Episode length: {episode_length} ctrl steps ({total_sim_steps} sim steps)")
        print(f"  Init qpos (joints): {env.mjData.qpos[7:7+n_joints]}")
        print(
            f"  Init commands: vx={commands[0]:.2f}, vy={commands[1]:.2f}, wz={commands[2]:.2f}"
        )

    mpc_solve_times = []
    fell = False

    if render:
        import mujoco.viewer as mjviewer

        viewer = mjviewer.launch_passive(env.mjModel, env.mjData)
    else:
        viewer = None

    torque_limits_active = (
        "torque_limits" in set(dr_patch["dr_applied_fields"].tolist())
        if dr_patch["dr_enabled"]
        else False
    )

    for t in range(total_sim_steps):
        qpos = env.mjData.qpos.copy()
        qvel = env.mjData.qvel.copy()

        is_mpc_step = t % sim_steps_per_ctrl == 0
        if is_mpc_step:
            ctrl_step = t // sim_steps_per_ctrl

            if steps_since_resample >= RL_COMMAND_RESAMPLE_INTERVAL and ctrl_step > 0:
                commands = sample_commands(rng)
                steps_since_resample = 0
                if verbose > 1:
                    print(
                        f"  [t={t}] Resampled commands: vx={commands[0]:.2f}, "
                        f"vy={commands[1]:.2f}, wz={commands[2]:.2f}"
                    )
            steps_since_resample += 1

            mpx_input = np.array(
                [commands[0], commands[1], 0.0, 0.0, 0.0, commands[2], config.robot_height]
            )

            contact_temp, _ = env.feet_contact_state()
            contact = np.array(
                [contact_temp[robot_feet_geom_names[leg]] for leg in ["FL", "FR", "RL", "RR"]]
            )

            start_t = timer()
            tau, q_des, dq_des = mpc.run(qpos, qvel, mpx_input, contact)
            solve_time = timer() - start_t
            mpc_solve_times.append(solve_time)

        tau_f64 = np.asarray(tau, dtype=np.float64)
        q_des_f64 = np.asarray(q_des, dtype=np.float64)

        tau_fb = 10 * (q_des_f64 - qpos[7 : 7 + n_joints]) - 2 * qvel[6 : 6 + n_joints]
        total_tau = tau_f64 + tau_fb
        if torque_limits_active:
            total_tau = np.clip(total_tau, torque_limits[:, 0], torque_limits[:, 1])

        commands_traj[:, t] = mpx_input[[0, 1, 5]]
        tau_applied_traj[:, t] = total_tau
        tau_mpx_traj[:, t] = tau_f64
        q_des_traj[:, t] = q_des_f64

        env.step(action=total_tau)

        qpos_traj[:, t + 1] = env.mjData.qpos.copy()
        qvel_traj[:, t + 1] = env.mjData.qvel.copy()
        time_traj[t + 1] = env.mjData.time

        if viewer is not None:
            viewer.sync()
            time.sleep(1.0 / sim_frequency)
            if not viewer.is_running():
                if verbose > 0:
                    print(f"  [Seed {seed}] Viewer closed by user - ending trajectory")
                break

        if (t + 1) % sim_steps_per_ctrl == 0 and _check_fell(env.mjData):
            fell = True
            if verbose > 1:
                ctrl_step_done = (t + 1) // sim_steps_per_ctrl
                print(f"  [Seed {seed}] Robot fell at ctrl step {ctrl_step_done} - aborting")
            break

        if verbose > 0 and (t + 1) % 1000 == 0:
            print(f"  [Seed {seed}] Sim step {t+1}/{total_sim_steps}")

    if viewer is not None:
        viewer.close()

    if verbose > 0:
        avg_solve = np.mean(mpc_solve_times) if mpc_solve_times else 0
        status = "FELL" if fell else "OK"
        print(f"  [Seed {seed}] Done [{status}]. Avg MPC solve: {avg_solve*1000:.1f}ms")

    return {
        "qpos": qpos_traj,
        "qvel": qvel_traj,
        "tau_applied": tau_applied_traj,
        "tau_mpx": tau_mpx_traj,
        "q_des": q_des_traj,
        "time": time_traj,
        "commands": commands_traj,
        "seed": seed,
        "default_joint_pos": default_joint_pos,
        "action_scale": RL_ACTION_SCALE,
        "sim_dt": 1 / sim_frequency,
        "control_dt": RL_CONTROL_DT,
        "episode_length": episode_length,
        "fell": fell,
        **dr_patch,
    }


def gen_traj_quadruped_dr(
    *,
    num_trajectories=100,
    episode_length=1000,
    start_seed=0,
    output_dir=None,
    max_attempts=None,
    verbose=1,
    render=False,
    domain_rand_config_type=DEFAULT_STARTUP_DOMAIN_RAND_PRESET,
    dr_seed_offset=1_000_000,
):
    """Generate quadruped MPC trajectories on startup-randomized plants."""
    resolved_dr_type, _ = resolve_startup_domain_rand_config(domain_rand_config_type)

    if output_dir is None:
        output_dir = (
            Path(__file__).parent.parent.parent / "data" / "quadruped_dr" / resolved_dr_type
        )
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose > 0:
        print(f"Generating {num_trajectories} valid DR quadruped trajectories")
        print(f"  DR preset: {resolved_dr_type}")
        print(f"  Episode length: {episode_length} ctrl steps")
        print(f"  Starting seed: {start_seed}")
        print(f"  DR seed offset: {dr_seed_offset}")
        print(f"  Max attempts: {'unlimited' if max_attempts is None else max_attempts}")
        print(f"  Output: {output_dir}")
        print()

    if verbose > 0:
        print("Pre-compiling JAX MPC kernels (once for all trajectories)...")
    shared_mpc = mpc_wrapper.MPCControllerWrapper(config)
    shared_mpc.robot_height = config.robot_height
    _dummy_qpos = np.concatenate(
        [np.array(config.p0), np.array(config.quat0), np.array(config.q0)]
    )
    _dummy_qvel = np.zeros(config.n_joints + 6)
    _dummy_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
    _dummy_contact = np.zeros(config.n_contact)
    shared_mpc.reset(_dummy_qpos, _dummy_qvel)
    _t0 = timer()
    shared_mpc.run(_dummy_qpos, _dummy_qvel, _dummy_input, _dummy_contact)
    if verbose > 0:
        print(f"JIT compilation done in {timer() - _t0:.1f}s")
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
        )

        if traj_data["fell"]:
            if verbose > 0:
                print(f"  [Seed {seed}] Robot fell - discarding trajectory, trying next seed")
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
            f"({attempt - saved} discarded due to falls)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPX quadruped trajectories with startup domain randomization"
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
        default=DEFAULT_STARTUP_DOMAIN_RAND_PRESET,
        choices=STARTUP_DOMAIN_RAND_PRESET_NAMES,
        help="Startup domain-randomization preset to apply to each trajectory",
    )
    parser.add_argument(
        "--dr-seed-offset",
        type=int,
        default=1_000_000,
        help="Offset added to the rollout seed for the independent DR RNG stream",
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
    )


if __name__ == "__main__":
    main()
