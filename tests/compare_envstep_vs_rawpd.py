"""Side-by-side comparison: env.step() vs raw PD loop on the SAME model/data.

Answers: does env.step() produce different physics than raw PD + mj_step?
"""

import numpy as np
import sys
from pathlib import Path
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig


def load_trajectory(traj_file):
    data = np.load(traj_file, allow_pickle=True)
    return {
        "qpos": data["qpos"],
        "qvel": data["qvel"],
        "tau_applied": data["tau_applied"],
        "tau_mpx": data["tau_mpx"],
        "q_des": data["q_des"],
        "time": data["time"],
        "commands": data["commands"],
        "seed": int(data["seed"]),
        "default_joint_pos": data["default_joint_pos"],
        "action_scale": float(data["action_scale"]),
        "sim_dt": float(data["sim_dt"]),
        "control_dt": float(data["control_dt"]),
        "episode_length": int(data["episode_length"]),
    }


def run_env_step(traj, max_steps):
    """Replay trajectory using env.step()."""
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

    # Disable termination
    env.max_pitch = np.pi
    env.max_roll = np.pi
    env.min_base_height = -1.0

    env.reset(seed=42)
    env.mjData.qpos[:] = traj["qpos"][:, 0]
    env.mjData.qvel[:] = traj["qvel"][:, 0]
    env.mjData.ctrl[:] = 0.0
    env.mjData.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.mjModel, env.mjData)
    env._last_action[:] = 0.0
    env._prev_last_action[:] = 0.0
    env._last_joint_vel = env.mjData.qvel[6:].copy()

    decimation = 4
    default_joint_pos = traj["default_joint_pos"]
    q_des_recorded = traj["q_des"]
    tau_mpx_recorded = traj["tau_mpx"]
    commands_recorded = traj["commands"]
    kp = env.kp
    qpos_history = []

    for ctrl_step in range(max_steps):
        sim_idx = ctrl_step * decimation
        q_des = q_des_recorded[:, sim_idx]
        tau_mpx = tau_mpx_recorded[:, sim_idx]
        q_target = q_des + tau_mpx / kp
        action = (q_target - default_joint_pos) / env.action_scale
        action = np.clip(action, -1.0, 1.0)

        cmd = commands_recorded[:, sim_idx]
        env.set_commands(vx=cmd[0], vy=cmd[1], wz=cmd[2])

        obs, reward, terminated, truncated, info = env.step(action)
        qpos_history.append(env.mjData.qpos.copy())

        if terminated or truncated:
            print(f"  env.step: terminated at step {ctrl_step+1}")
            break

    env.close()
    return np.array(qpos_history)


def run_raw_pd(traj, max_steps):
    """Replay trajectory using raw PD + mj_step, same model as RL env."""
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

    model = env.mjModel
    data = env.mjData

    # Set initial state exactly like env.step path
    env.reset(seed=42)
    data.qpos[:] = traj["qpos"][:, 0]
    data.qvel[:] = traj["qvel"][:, 0]
    data.ctrl[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(model, data)

    decimation = 4
    kp = 10.0
    kd = 2.0
    action_scale = 4.0
    default_joint_pos = traj["default_joint_pos"]
    q_des_recorded = traj["q_des"]
    tau_mpx_recorded = traj["tau_mpx"]
    qpos_history = []

    for ctrl_step in range(max_steps):
        sim_idx = ctrl_step * decimation
        q_des = q_des_recorded[:, sim_idx]
        tau_mpx = tau_mpx_recorded[:, sim_idx]
        q_target = q_des + tau_mpx / kp
        action = (q_target - default_joint_pos) / action_scale
        action = np.clip(action, -1.0, 1.0)
        q_target_clipped = default_joint_pos + action_scale * action

        for _ in range(decimation):
            q_current = data.qpos[7:19]
            dq_current = data.qvel[6:18]
            torques = kp * (q_target_clipped - q_current) + kd * (0.0 - dq_current)
            torques = np.clip(torques, model.actuator_ctrlrange[:, 0],
                              model.actuator_ctrlrange[:, 1])
            data.ctrl[:12] = torques
            mujoco.mj_step(model, data)

        qpos_history.append(data.qpos.copy())

    env.close()
    return np.array(qpos_history)


def main():
    data_dir = Path("data/quadruped")
    files = sorted(data_dir.glob("*.npz"))
    traj_file = files[0]
    print(f"Trajectory: {traj_file.name}")
    traj = load_trajectory(traj_file)

    max_steps = 500
    ctrl_dt = traj["control_dt"]
    decimation = 4
    qpos_recorded = traj["qpos"]

    print("\nRunning env.step() replay...")
    env_qpos = run_env_step(traj, max_steps)

    print("Running raw PD replay...")
    raw_qpos = run_raw_pd(traj, max_steps)

    num_steps = min(len(env_qpos), len(raw_qpos))
    print(f"\nComparing {num_steps} steps...")

    # Compare env.step() vs raw PD
    print(f"\n{'Step':>5} {'Time':>6} {'env-raw L2':>12} {'env-rec L2':>12} {'raw-rec L2':>12}")
    print("-" * 50)
    for step in range(num_steps):
        t = (step + 1) * ctrl_dt
        rec_idx = (step + 1) * decimation
        if rec_idx >= qpos_recorded.shape[1]:
            break

        env_joints = env_qpos[step, 7:19]
        raw_joints = raw_qpos[step, 7:19]
        rec_joints = qpos_recorded[7:19, rec_idx]

        env_raw_diff = np.linalg.norm(env_joints - raw_joints)
        env_rec_diff = np.linalg.norm(env_joints - rec_joints)
        raw_rec_diff = np.linalg.norm(raw_joints - rec_joints)

        if step < 10 or step % 50 == 49 or env_raw_diff > 0.001:
            print(f"{step+1:>5} {t:>6.2f} {env_raw_diff:>12.6f} {env_rec_diff:>12.6f} {raw_rec_diff:>12.6f}")

    # Per-joint comparison at first divergence step
    print("\n\nDetailed per-joint comparison at first significant divergence:")
    for step in range(num_steps):
        rec_idx = (step + 1) * decimation
        if rec_idx >= qpos_recorded.shape[1]:
            break
        env_joints = env_qpos[step, 7:19]
        raw_joints = raw_qpos[step, 7:19]
        diff = np.abs(env_joints - raw_joints)
        if np.max(diff) > 1e-6:
            t = (step + 1) * ctrl_dt
            print(f"\nFirst divergence at step {step+1} (t={t:.2f}s):")
            joint_names = ["FL_hip", "FL_thigh", "FL_calf",
                           "FR_hip", "FR_thigh", "FR_calf",
                           "RL_hip", "RL_thigh", "RL_calf",
                           "RR_hip", "RR_thigh", "RR_calf"]
            for j in range(12):
                if diff[j] > 1e-8:
                    print(f"  {joint_names[j]:>12}: env={env_joints[j]:.8f}, "
                          f"raw={raw_joints[j]:.8f}, diff={diff[j]:.2e}")
            # Also compare base state
            env_base = env_qpos[step, :7]
            raw_base = raw_qpos[step, :7]
            base_diff = np.abs(env_base - raw_base)
            if np.max(base_diff) > 1e-8:
                print(f"  Base pos/quat diff: {base_diff}")
            break

    # Show final comparison
    if num_steps > 0:
        step = num_steps - 1
        rec_idx = (step + 1) * decimation
        if rec_idx < qpos_recorded.shape[1]:
            env_joints = env_qpos[step, 7:19]
            raw_joints = raw_qpos[step, 7:19]
            rec_joints = qpos_recorded[7:19, rec_idx]
            print(f"\nFinal step {step+1} (t={(step+1)*ctrl_dt:.2f}s):")
            print(f"  env.step vs raw PD  L2: {np.linalg.norm(env_joints - raw_joints):.6f}")
            print(f"  env.step vs record  L2: {np.linalg.norm(env_joints - rec_joints):.6f}")
            print(f"  raw PD   vs record  L2: {np.linalg.norm(raw_joints - rec_joints):.6f}")


if __name__ == "__main__":
    main()
