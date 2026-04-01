import jax.numpy as jnp
import jax
import mujoco
# JAX configuration (must be before other JAX imports)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import numpy as np
from pathlib import Path
import sys
import argparse
import time
from scipy.spatial.transform import Rotation
from timeit import default_timer as timer

from gym_quadruped.quadruped_env import QuadrupedEnv

import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_go2 as config

# Select device (GPU if available, else CPU)
try:
    gpu_device = jax.devices('gpu')[0]
except RuntimeError:
    gpu_device = jax.devices('cpu')[0]
jax.default_device(gpu_device)


# RL environment defaults (from QuadrupedVelocityTrackingEnv)
RL_SIM_DT = 0.005          # 200 Hz physics
RL_DECIMATION = 4           # control at 50 Hz
RL_CONTROL_DT = RL_SIM_DT * RL_DECIMATION  # 0.02s
RL_ACTION_SCALE = 0.5
RL_LIN_VEL_X_RANGE = (-0.5, 1.0/2)
RL_LIN_VEL_Y_RANGE = (-0.5/2, 0.5/2)
RL_ANG_VEL_Z_RANGE = (-1.0/2, 1.0/2)
RL_COMMAND_RESAMPLE_INTERVAL = 250  # control steps
RL_JOINT_POS_NOISE = 0.05          # radians
RL_BASE_ORIENT_NOISE = 0.03        # radians (roll, pitch)
RL_JOINT_VEL_NOISE = 0.05          # rad/s
# Termination thresholds - must match QuadrupedVelocityTrackingEnv defaults
RL_MAX_ROLL = 0.5          # radians
RL_MAX_PITCH = 0.5         # radians
RL_MIN_BASE_HEIGHT = 0.1   # meters
COMMAND_THRESHOLD = 0.05    # m/s threshold to go from standing to walking


def sample_commands(rng):
    """Sample velocity commands matching the RL env distribution."""
    vx = rng.uniform(*RL_LIN_VEL_X_RANGE)
    vy = rng.uniform(*RL_LIN_VEL_Y_RANGE)
    wz = rng.uniform(*RL_ANG_VEL_Z_RANGE)
    return np.array([vx, vy, wz])


def randomize_initial_state(env, rng):
    """Apply initial state randomization matching QuadrupedVelocityTrackingEnv.reset().

    Args:
        env: QuadrupedEnv instance whose mjData will be modified in-place.
        rng: numpy RandomState for deterministic randomization.
    """
    n_joints = config.n_joints

    # Reset to home keyframe
    keyframe_id = mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(env.mjModel, env.mjData, keyframe_id)

    # Add joint position noise (matching RL env)
    env.mjData.qpos[7:7 + n_joints] += rng.uniform(
        -RL_JOINT_POS_NOISE, RL_JOINT_POS_NOISE, size=n_joints
    )

    # Small random base orientation perturbation (roll, pitch)
    roll_noise = rng.uniform(-RL_BASE_ORIENT_NOISE, RL_BASE_ORIENT_NOISE)
    pitch_noise = rng.uniform(-RL_BASE_ORIENT_NOISE, RL_BASE_ORIENT_NOISE)
    base_quat_wxyz = env.mjData.qpos[3:7].copy()
    base_quat_xyzw = np.roll(base_quat_wxyz, -1)
    base_rot = Rotation.from_quat(base_quat_xyzw)
    noise_rot = Rotation.from_euler("xyz", [roll_noise, pitch_noise, 0.0])
    combined_rot = noise_rot * base_rot
    combined_quat_xyzw = combined_rot.as_quat()
    env.mjData.qpos[3:7] = np.roll(combined_quat_xyzw, 1)  # back to wxyz

    # Zero all velocities
    env.mjData.qvel[:] = 0.0
    env.mjData.qacc[:] = 0.0
    env.mjData.ctrl[:] = 0.0

    # Forward kinematics (without advancing sim)
    mujoco.mj_forward(env.mjModel, env.mjData)

    # Add small joint velocity noise after forward kinematics (matching RL env)
    nv = env.mjModel.nv
    env.mjData.qvel[6:] = rng.uniform(-RL_JOINT_VEL_NOISE, RL_JOINT_VEL_NOISE, size=nv - 6)


def _check_fell(mjdata):
    """Check whether the robot has fallen using the same criteria as the RL env.

    Mirrors ``QuadrupedVelocityTrackingEnv._check_termination()``: the episode
    ends if the base roll or pitch exceeds 0.5 rad, or the base height drops
    below 0.1 m.  Body-ground contact is intentionally NOT checked here because
    the RL training environment does not use that criterion.

    Args:
        mjdata: mujoco.MjData of the running simulation.

    Returns:
        True if the robot should be considered fallen, False otherwise.
    """
    quat_wxyz = mjdata.qpos[3:7]
    quat_xyzw = np.roll(quat_wxyz, -1)
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    roll, pitch = euler[0], euler[1]
    if abs(roll) > RL_MAX_ROLL or abs(pitch) > RL_MAX_PITCH:
        return True
    if mjdata.qpos[2] < RL_MIN_BASE_HEIGHT:
        return True
    return False


def generate_trajectory(seed, mpc=None, episode_length=1000, verbose=1, render=False):
    """Generate a single MPX-controlled quadruped trajectory with random init and commands.

    The simulation runs at 200 Hz. The MPX controller updates at 50 Hz (every 4 sim steps).
    Velocity commands are resampled every 250 control steps (= 1000 sim steps), matching the
    RL environment's command_resample_interval.

    A trajectory is marked as fallen (``fell=True``) when the base roll/pitch exceeds
    0.5 rad or the base height drops below 0.1 m, matching the termination criteria of
    ``QuadrupedVelocityTrackingEnv``.  The check is performed once per control step (50 Hz)
    to match the RL env's behavior.  The sim loop exits early when the robot has fallen.

    Args:
        seed: Random seed for reproducible initial state and command sampling.
        mpc: Optional pre-built MPCControllerWrapper. When provided (e.g. from
            gen_traj_quadruped), the JIT-compiled kernels are reused across
            trajectories and only mpc.reset() is called to reinitialise the
            warm-start. When None, a new wrapper is created and compiled here.
        episode_length: Number of control steps (50 Hz). Total sim steps = episode_length * 4.
        verbose: 0=quiet, 1=progress, 2=detailed.
        render: If True, open a passive MuJoCo viewer window and display the simulation
            in real time.  Closes automatically when the trajectory ends or the robot falls.

    Returns:
        dict with trajectory data arrays, including a ``fell`` boolean key.
    """
    rng = np.random.RandomState(seed)
    n_joints = config.n_joints
    sim_frequency = 200.0
    mpc_frequency = config.mpc_frequency  # 50 Hz
    sim_steps_per_ctrl = int(sim_frequency / mpc_frequency)  # 4
    total_sim_steps = episode_length * sim_steps_per_ctrl

    robot_feet_geom_names = dict(FR='FR', FL='FL', RR='RR', RL='RL')
    env = QuadrupedEnv(
        robot="go2",
        scene="flat",
        sim_dt=1 / sim_frequency,
        ref_base_lin_vel=0.0,
        ground_friction_coeff=0.7,
        base_vel_command_type="human",
        state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
    )
    env.reset(random=False)

    # Default joint positions from the MuJoCo model keyframe (float64).
    # Using the keyframe directly avoids precision loss from JAX's float32
    # default when reading config.q0.
    default_joint_pos = env.mjModel.key_qpos[0, 7:7 + n_joints].copy()

    # Apply random initial state (matching RL env distribution)
    randomize_initial_state(env, rng)

    # Create the MPC controller if one was not supplied by the caller.
    # When called from gen_traj_quadruped a single pre-compiled instance is
    # passed in so that JIT compilation only occurs once for all trajectories.
    own_mpc = mpc is None
    if own_mpc:
        mpc = mpc_wrapper.MPCControllerWrapper(config)
        mpc.robot_height = config.robot_height
        # Trigger JIT compilation before opening the viewer or recording data.
        if verbose > 0:
            print(f"[Seed {seed}] Pre-compiling JAX MPC kernels...")
        _jit_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
        _jit_contact_temp, _ = env.feet_contact_state()
        _jit_contact = np.array([
            _jit_contact_temp[robot_feet_geom_names[leg]]
            for leg in ['FL', 'FR', 'RL', 'RR']
        ])
        _start_compile = timer()
        mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())
        mpc.run(env.mjData.qpos.copy(), env.mjData.qvel.copy(), _jit_input, _jit_contact)
        if verbose > 0:
            print(f"[Seed {seed}] JIT compilation done in {timer() - _start_compile:.1f}s")

    # Reset warm-start to match the current randomised initial state.
    mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())

    # Initialize MPC state variables
    tau = jnp.zeros(n_joints)
    q_des = config.q0.copy()
    dq_des = jnp.zeros(n_joints)

    # Pre-allocate trajectory storage (at sim frequency)
    nq = env.mjModel.nq  # 19
    nv = env.mjModel.nv  # 18
    qpos_traj = np.zeros((nq, total_sim_steps + 1))
    qvel_traj = np.zeros((nv, total_sim_steps + 1))
    tau_applied_traj = np.zeros((n_joints, total_sim_steps))     # total torques applied
    tau_mpx_traj = np.zeros((n_joints, total_sim_steps))         # MPX feedforward torques
    q_des_traj = np.zeros((n_joints, total_sim_steps))           # MPX desired joint positions
    time_traj = np.zeros(total_sim_steps + 1)
    commands_traj = np.zeros((3, total_sim_steps))

    # Cache initial state
    qpos_traj[:, 0] = env.mjData.qpos.copy()
    qvel_traj[:, 0] = env.mjData.qvel.copy()
    time_traj[0] = env.mjData.time

    # Sample target velocity commands; they will be ramped up from zero to
    # avoid hitting the MPC with full-speed requests from a standing start.
    commands = sample_commands(rng)

    # Check if commands are above the threshold for zeroing (matching RL env)
    cmd_lin_norm = np.linalg.norm(commands[:2])
    cmd_ang_norm = abs(commands[2])
    total_cmd = cmd_lin_norm + cmd_ang_norm
    print("total command: ", total_cmd)
    if total_cmd < COMMAND_THRESHOLD:
        config.duty_factor = 1.0  # standing still, so set duty factor to 1.0 (no walking)
        mpc.duty_factor = 1.0
    else:
        config.duty_factor = 0.5  # moving, so set duty factor to 0.5 (trotting)
        mpc.duty_factor = 0.5

    steps_since_resample = 0  # in control steps
    # Number of control steps over which to linearly ramp commands to full value.
    # At 50 Hz, 50 steps = 1 second of ramp-up time.
    CMD_RAMP_STEPS = 50 # TODO: test out longer like 150

    if verbose > 0:
        print(f"[Seed {seed}] Starting trajectory generation")
        print(f"  Episode length: {episode_length} ctrl steps ({total_sim_steps} sim steps)")
        print(f"  Init qpos (joints): {env.mjData.qpos[7:7+n_joints]}")
        print(f"  Init commands: vx={commands[0]:.2f}, vy={commands[1]:.2f}, wz={commands[2]:.2f}")

    mpc_solve_times = []
    fell = False

    if render:
        import mujoco.viewer as mjviewer
        viewer = mjviewer.launch_passive(env.mjModel, env.mjData)
    else:
        viewer = None

    for t in range(total_sim_steps):
        qpos = env.mjData.qpos.copy()
        qvel = env.mjData.qvel.copy()

        is_mpc_step = (t % sim_steps_per_ctrl == 0)

        if is_mpc_step:
            ctrl_step = t // sim_steps_per_ctrl

            # Resample commands periodically (every 250 control steps)
            if steps_since_resample >= RL_COMMAND_RESAMPLE_INTERVAL and ctrl_step > 0:
                commands = sample_commands(rng)
                steps_since_resample = 0
                if verbose > 1:
                    print(f"  [t={t}] Resampled commands: vx={commands[0]:.2f}, "
                          f"vy={commands[1]:.2f}, wz={commands[2]:.2f}")
            steps_since_resample += 1

            # Build MPX input: [vx, vy, vz, wx, wy, wz, height]
            # Linearly ramp commands from zero to the target over CMD_RAMP_STEPS
            # so the MPC solver is not hit with full-speed requests from step 0.
            ramp_scale = min(1.0, ctrl_step / CMD_RAMP_STEPS)
            mpx_input = np.array([
                ramp_scale * commands[0], ramp_scale * commands[1], 0.0,
                0.0, 0.0, ramp_scale * commands[2],
                config.robot_height
            ])

            # Get foot contact states
            contact_temp, _ = env.feet_contact_state()
            contact = np.array([
                contact_temp[robot_feet_geom_names[leg]]
                for leg in ['FL', 'FR', 'RL', 'RR']
            ])

            # Solve MPC
            start_t = timer()
            tau, q_des, dq_des = mpc.run(qpos, qvel, mpx_input, contact)
            solve_time = timer() - start_t
            mpc_solve_times.append(solve_time)

        # Convert MPC outputs from JAX float32 to numpy float64 before the PD
        # computation.  JAX demotes mixed float32/float64 ops to float32, which
        # would silently truncate the float64 MuJoCo state and produce torques
        # that differ from a pure-float64 replay.
        tau_f64 = np.asarray(tau, dtype=np.float64)
        q_des_f64 = np.asarray(q_des, dtype=np.float64)

        # Compute PD feedback and total torque (same gains as MPXPlanner: kp=10, kd=2)
        tau_fb = 10 * (q_des_f64 - qpos[7:7 + n_joints]) - 2 * qvel[6:6 + n_joints]
        total_tau = tau_f64 + tau_fb

        # Record (at sim frequency)
        commands_traj[:, t] = mpx_input[[0, 1, 5]] # commands
        tau_applied_traj[:, t] = total_tau
        tau_mpx_traj[:, t] = tau_f64
        q_des_traj[:, t] = q_des_f64

        # Step simulation
        env.step(action=total_tau)

        # Record state after stepping
        qpos_traj[:, t + 1] = env.mjData.qpos.copy()
        qvel_traj[:, t + 1] = env.mjData.qvel.copy()
        time_traj[t + 1] = env.mjData.time

        if viewer is not None:
            viewer.sync()
            time.sleep(1.0 / sim_frequency)  # pace to real time
            if not viewer.is_running():
                if verbose > 0:
                    print(f"  [Seed {seed}] Viewer closed by user - ending trajectory")
                break

        # Check termination once per complete control step (matching RL env at 50 Hz).
        # Uses roll/pitch/height thresholds - NOT body-ground contact - so that valid
        # trajectories are not discarded for transient knee/thigh grazes.
        if (t + 1) % sim_steps_per_ctrl == 0:
            if _check_fell(env.mjData):
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
        "qpos": qpos_traj,                      # (nq, T+1) at sim freq
        "qvel": qvel_traj,                       # (nv, T+1) at sim freq
        "tau_applied": tau_applied_traj,         # (n_joints, T) total torques
        "tau_mpx": tau_mpx_traj,                 # (n_joints, T) MPX feedforward torques
        "q_des": q_des_traj,                     # (n_joints, T) MPX desired joint positions
        "time": time_traj,                       # (T+1,) timestamps
        "commands": commands_traj,               # (3, T) velocity commands [vx, vy, wz]
        "seed": seed,
        "default_joint_pos": default_joint_pos,  # (n_joints,) for RL action conversion
        "action_scale": RL_ACTION_SCALE,
        "sim_dt": 1 / sim_frequency,
        "control_dt": RL_CONTROL_DT,
        "episode_length": episode_length,
        "fell": fell,
    }


def gen_traj_quadruped(
    num_trajectories=100,
    episode_length=1000,
    start_seed=0,
    output_dir=None,
    max_attempts=None,
    verbose=1,
    render=False,
):
    """Generate N quadruped trajectories where the robot does not fall.

    Trajectories where the robot falls (roll/pitch > 0.5 rad or height < 0.1 m,
    matching the RL env) are discarded and a new attempt is made with the next
    seed.  The function keeps retrying until exactly ``num_trajectories`` valid
    trajectories are saved, or until ``max_attempts`` total attempts are made.

    Args:
        num_trajectories: Number of valid (non-fallen) trajectories to collect.
        episode_length: Control steps per trajectory (50 Hz). Sim steps = episode_length * 4.
        start_seed: First seed value.  Seeds increment by 1 for every attempt
            (both successful and failed).
        output_dir: Output directory. Defaults to MPC-RL/data/quadruped/.
        max_attempts: Maximum total attempts (successful + failed) before stopping.
            If None, retries indefinitely until num_trajectories are saved.
        verbose: Verbosity level.
        render: If True, open a viewer window for each trajectory attempt.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "quadruped"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose > 0:
        print(f"Generating {num_trajectories} valid (non-fallen) quadruped trajectories")
        print(f"  Episode length: {episode_length} ctrl steps")
        print(f"  Starting seed: {start_seed}")
        print(f"  Max attempts: {'unlimited' if max_attempts is None else max_attempts}")
        print(f"  Output: {output_dir}")
        print()

    # Build and JIT-compile the MPC controller once. All trajectory calls reuse
    # the same compiled kernels; only mpc.reset() is called between trajectories
    # to reinitialise the warm-start.
    if verbose > 0:
        print("Pre-compiling JAX MPC kernels (once for all trajectories)...")
    shared_mpc = mpc_wrapper.MPCControllerWrapper(config)
    shared_mpc.robot_height = config.robot_height
    _dummy_qpos = np.concatenate([np.array(config.p0), np.array(config.quat0), np.array(config.q0)])
    _dummy_qvel = np.zeros(config.n_joints + 6)
    _dummy_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
    _dummy_contact = np.zeros(config.n_contact)
    shared_mpc.reset(_dummy_qpos, _dummy_qvel)
    _t0 = timer()
    shared_mpc.run(_dummy_qpos, _dummy_qvel, _dummy_input, _dummy_contact)
    if verbose > 0:
        print(f"JIT compilation done in {timer() - _t0:.1f}s")
        print()

    saved = 0       # number of valid trajectories saved
    attempt = 0     # total attempts (valid + fallen)

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

        filename = f"quadruped_seed_{seed:06d}_ep_{episode_length}.npz"
        filepath = output_dir / filename

        np.savez_compressed(filepath, **traj_data)
        saved += 1

        if verbose > 0:
            print(f"  Saved ({saved}/{num_trajectories}): {filepath}")
            print()

    if verbose > 0:
        print(
            f"Done! Saved {num_trajectories} valid trajectories in {output_dir} "
            f"({attempt - num_trajectories} discarded due to falls)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPX quadruped trajectories for MPC-Injection into RL"
    )
    parser.add_argument(
        "--num-trajectories", "-n", type=int, default=100,
        help="Number of trajectories to generate (default: 100)"
    )
    parser.add_argument(
        "--episode-length", type=int, default=1000,
        help="Episode length in control steps at 50 Hz (default: 1000 = 20s)"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0,
        help="Starting random seed (default: 0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/quadruped/)"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=10000,
        help="Maximum total attempts before stopping (default: 10000)"
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level (default: 1)"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Open a MuJoCo viewer window to display each trajectory attempt in real time"
    )

    args = parser.parse_args()

    gen_traj_quadruped(
        num_trajectories=args.num_trajectories,
        episode_length=args.episode_length,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
        max_attempts=args.max_attempts,
        verbose=args.verbose,
        render=args.render,
    )


if __name__ == "__main__":
    main()
