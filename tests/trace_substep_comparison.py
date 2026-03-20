"""Trace exactly what env.step() does vs manual PD at the substep level.

Uses a SINGLE env instance to eliminate any model-compilation differences.
"""

import numpy as np
import copy
import sys
from pathlib import Path
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig


def load_trajectory(traj_file):
    data = np.load(traj_file, allow_pickle=True)
    return {
        "qpos": data["qpos"], "qvel": data["qvel"],
        "tau_applied": data["tau_applied"], "tau_mpx": data["tau_mpx"],
        "q_des": data["q_des"], "time": data["time"],
        "commands": data["commands"], "seed": int(data["seed"]),
        "default_joint_pos": data["default_joint_pos"],
        "action_scale": float(data["action_scale"]),
        "sim_dt": float(data["sim_dt"]),
        "control_dt": float(data["control_dt"]),
        "episode_length": int(data["episode_length"]),
    }


def main():
    data_dir = Path("data/quadruped")
    files = sorted(data_dir.glob("*.npz"))
    traj_file = files[0]
    traj = load_trajectory(traj_file)
    print(f"Trajectory: {traj_file.name}")

    env = QuadrupedVelocityTrackingEnv(
        robot="go2", scene="flat", render_mode=None,
        kp=10.0, kd=2.0, action_scale=4.0,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
    )

    # Match friction
    friction = np.array([0.7, 0.005, 0.0])
    foot_geom_ids = set(env._foot_geom_ids.values())
    for gid in range(env.mjModel.ngeom):
        name = mujoco.mj_id2name(env.mjModel, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if (name and name.lower() in {"floor", "ground"}) or gid in foot_geom_ids:
            env.mjModel.geom_friction[gid, :] = friction

    env.max_pitch = np.pi
    env.max_roll = np.pi
    env.min_base_height = -1.0

    # Check if env.default_joint_pos matches traj
    env_default = env.default_joint_pos
    traj_default = traj["default_joint_pos"]
    print(f"\ndefault_joint_pos match: {np.array_equal(env_default, traj_default)}")
    print(f"  env:  {env_default}")
    print(f"  traj: {traj_default}")
    if not np.array_equal(env_default, traj_default):
        print(f"  diff: {env_default - traj_default}")
        print(f"  max diff: {np.max(np.abs(env_default - traj_default))}")

    # Check kp / kd
    print(f"\nkp: {env.kp}")
    print(f"kd: {env.kd}")
    print(f"action_scale: {env.action_scale}")
    print(f"torque_limits: {env.torque_limits}")

    # -------- Approach 1: Use env.step(), save ctrl at each substep ---------
    print("\n" + "=" * 60)
    print("Step-by-step trace: env.step() internals")
    print("=" * 60)

    env.reset(seed=42)
    env.mjData.qpos[:] = traj["qpos"][:, 0]
    env.mjData.qvel[:] = traj["qvel"][:, 0]
    env.mjData.ctrl[:] = 0.0
    env.mjData.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.mjModel, env.mjData)
    env._last_action[:] = 0.0
    env._prev_last_action[:] = 0.0
    env._last_joint_vel = env.mjData.qvel[6:].copy()

    # Save initial state for part 2
    init_qpos = env.mjData.qpos.copy()
    init_qvel = env.mjData.qvel.copy()

    decimation = 4
    default_joint_pos = traj["default_joint_pos"]
    q_des_recorded = traj["q_des"]
    tau_mpx_recorded = traj["tau_mpx"]
    commands_recorded = traj["commands"]
    qpos_recorded = traj["qpos"]
    kp_val = 10.0
    kd_val = 2.0
    action_scale = 4.0

    # Instrument step by monkey-patching to capture substep data
    substep_data_env = []  # list of (q_target, q_current, dq_current, torques, ctrl)

    original_step = env.step

    def instrumented_step(action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        env._prev_last_action = env._last_action.copy()
        env._last_action = action.copy()
        q_target = env.default_joint_pos + env.action_scale * action

        for _ in range(env.decimation):
            q_current = env.mjData.qpos[7:].copy()
            dq_current = env.mjData.qvel[6:].copy()
            torques = env.kp * (q_target - q_current) + env.kd * (0.0 - dq_current)
            torques = np.clip(torques, env.torque_limits[:, 0], env.torque_limits[:, 1])
            substep_data_env.append({
                'q_target': q_target.copy(),
                'q_current': q_current.copy(),
                'dq_current': dq_current.copy(),
                'torques': torques.copy(),
            })
            env._applied_torques = torques
            env.mjData.ctrl[:] = torques
            mujoco.mj_step(env.mjModel, env.mjData)

        env._step_count += 1
        env._steps_since_command_resample += 1
        env._maybe_push_robot()
        env._update_feet_air_time()
        joint_vel_current = env.mjData.qvel[6:].copy()
        env._joint_acc = (joint_vel_current - env._last_joint_vel) / env.control_dt
        env._last_joint_vel = joint_vel_current
        if not env._fixed_commands and env._steps_since_command_resample >= env.command_resample_interval:
            env._sample_commands()
        obs = env._get_obs()
        terminated = env._check_termination()
        reward = env._compute_reward(action, terminated)
        truncated = False
        info = env._get_info()
        env._swing_peak *= ~env._current_contacts
        return obs, reward, terminated, truncated, info

    env_qpos_hist = []
    max_steps = 500

    for ctrl_step in range(max_steps):
        sim_idx = ctrl_step * decimation
        q_des = q_des_recorded[:, sim_idx]
        tau_mpx = tau_mpx_recorded[:, sim_idx]
        q_target = q_des + tau_mpx / kp_val
        action = (q_target - default_joint_pos) / action_scale
        action = np.clip(action, -1.0, 1.0)

        cmd = commands_recorded[:, sim_idx]
        env.set_commands(vx=cmd[0], vy=cmd[1], wz=cmd[2])

        obs, reward, terminated, truncated, info = instrumented_step(action)
        env_qpos_hist.append(env.mjData.qpos.copy())

        if terminated or truncated:
            print(f"env terminated at step {ctrl_step+1}")
            break

    # -------- Approach 2: Raw PD on the SAME env model/data ---------
    print(f"\nEnv.step done. Resetting for raw PD on same model...")

    # Reset data to initial state (reuse same model)
    env.mjData.qpos[:] = init_qpos
    env.mjData.qvel[:] = init_qvel
    env.mjData.ctrl[:] = 0.0
    env.mjData.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.mjModel, env.mjData)

    substep_data_raw = []
    raw_qpos_hist = []

    for ctrl_step in range(max_steps):
        sim_idx = ctrl_step * decimation
        q_des = q_des_recorded[:, sim_idx]
        tau_mpx = tau_mpx_recorded[:, sim_idx]
        q_target = q_des + tau_mpx / kp_val
        action = (q_target - default_joint_pos) / action_scale
        action = np.clip(action, -1.0, 1.0)
        q_target_clipped = default_joint_pos + action_scale * action

        for _ in range(decimation):
            q_current = env.mjData.qpos[7:19].copy()
            dq_current = env.mjData.qvel[6:18].copy()
            torques = kp_val * (q_target_clipped - q_current) + kd_val * (0.0 - dq_current)
            torques = np.clip(torques, env.mjModel.actuator_ctrlrange[:, 0],
                              env.mjModel.actuator_ctrlrange[:, 1])
            substep_data_raw.append({
                'q_target': q_target_clipped.copy(),
                'q_current': q_current.copy(),
                'dq_current': dq_current.copy(),
                'torques': torques.copy(),
            })
            env.mjData.ctrl[:12] = torques
            mujoco.mj_step(env.mjModel, env.mjData)

        raw_qpos_hist.append(env.mjData.qpos.copy())

    env.close()

    # -------- Compare --------
    num_steps = min(len(env_qpos_hist), len(raw_qpos_hist))
    print(f"\nComparing {num_steps} control steps ({num_steps * 4} substeps)...")

    # Compare substep-level data for the first control step
    print(f"\n--- First 4 substeps (ctrl step 0) ---")
    for sub in range(4):
        e = substep_data_env[sub]
        r = substep_data_raw[sub]
        q_target_diff = np.max(np.abs(e['q_target'] - r['q_target']))
        q_curr_diff = np.max(np.abs(e['q_current'] - r['q_current']))
        dq_diff = np.max(np.abs(e['dq_current'] - r['dq_current']))
        tau_diff = np.max(np.abs(e['torques'] - r['torques']))
        print(f"  sub {sub}: q_target_diff={q_target_diff:.2e}, q_curr_diff={q_curr_diff:.2e}, "
              f"dq_diff={dq_diff:.2e}, torque_diff={tau_diff:.2e}")

    # Find first substep where torques differ
    print(f"\n--- Searching for first substep with torque difference > 1e-14 ---")
    total_substeps = min(len(substep_data_env), len(substep_data_raw))
    first_diff_step = None
    for s in range(total_substeps):
        e = substep_data_env[s]
        r = substep_data_raw[s]
        tau_diff = np.max(np.abs(e['torques'] - r['torques']))
        if tau_diff > 1e-14:
            first_diff_step = s
            ctrl_step = s // 4
            sub = s % 4
            print(f"\nFirst torque diff at substep {s} (ctrl_step={ctrl_step}, sub={sub}):")
            print(f"  torque diff: {tau_diff:.2e}")
            print(f"  q_target diff: {np.max(np.abs(e['q_target'] - r['q_target'])):.2e}")
            print(f"  q_current diff: {np.max(np.abs(e['q_current'] - r['q_current'])):.2e}")
            print(f"  dq_current diff: {np.max(np.abs(e['dq_current'] - r['dq_current'])):.2e}")

            # Show the actual values
            for j in range(12):
                if abs(e['torques'][j] - r['torques'][j]) > 1e-15:
                    print(f"  joint {j}: env_tau={e['torques'][j]:.15f}, raw_tau={r['torques'][j]:.15f}, "
                          f"diff={e['torques'][j]-r['torques'][j]:.2e}")
            break

    if first_diff_step is None:
        print("  No torque differences found at substep level!")

        # If torques are identical, check if qpos diverges
        print(f"\n--- Checking for qpos divergence despite identical torques ---")
        for s in range(total_substeps):
            e = substep_data_env[s]
            r = substep_data_raw[s]
            q_diff = np.max(np.abs(e['q_current'] - r['q_current']))
            if q_diff > 1e-14:
                print(f"  First qpos diff at substep {s}: {q_diff:.2e}")
                for j in range(12):
                    d = abs(e['q_current'][j] - r['q_current'][j])
                    if d > 1e-15:
                        print(f"    joint {j}: env={e['q_current'][j]:.15f}, raw={r['q_current'][j]:.15f}")
                break
        else:
            print("  No qpos differences found! Results should be identical.")

    # Control-level comparison
    print(f"\n--- Control-level qpos comparison ---")
    for step in range(num_steps):
        env_joints = env_qpos_hist[step][7:19]
        raw_joints = raw_qpos_hist[step][7:19]
        diff = np.linalg.norm(env_joints - raw_joints)
        if step < 5 or diff > 0.001 or step % 100 == 99:
            t = (step + 1) * traj["control_dt"]
            rec_idx = (step + 1) * decimation
            if rec_idx < qpos_recorded.shape[1]:
                rec_joints = qpos_recorded[7:19, rec_idx]
                rec_env = np.linalg.norm(env_joints - rec_joints)
                rec_raw = np.linalg.norm(raw_joints - rec_joints)
                print(f"  step {step+1:>4} t={t:>5.2f}: "
                      f"env-raw={diff:.6e}, env-rec={rec_env:.6e}, raw-rec={rec_raw:.6e}")


if __name__ == "__main__":
    main()
