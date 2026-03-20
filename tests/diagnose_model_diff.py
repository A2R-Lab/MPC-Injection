"""Definitive test: replay MPC trajectory using the SAME MuJoCo model (go2_mjx.xml).

Compares 3 replay approaches:
  A) Torque replay on the MPX model (go2_mjx.xml) -- should be near-zero error
  B) Torque replay on the RL model (go2.xml) -- shows model-difference error
  C) Matched-gain PD replay on the RL model (go2.xml) -- current test approach
  D) Matched-gain PD replay on the RL model with all model params patched to MPX

If (A) succeeds and (C)/(D) fail, the root cause is the different MuJoCo models.
"""

import numpy as np
import sys
from pathlib import Path
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))

WORKSPACE = Path(__file__).parent.parent

# Model paths
MPX_SCENE = WORKSPACE / "deps" / "mpx" / "mpx" / "data" / "go2" / "scene_mjx.xml"
RL_MODEL = WORKSPACE / "deps" / "gym-quadruped" / "gym_quadruped" / "robot_model" / "go2" / "go2.xml"


def load_trajectory(traj_file):
    """Load and parse an MPX quadruped trajectory file."""
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


def dump_model_params(model, label):
    """Print key physics parameters of a MuJoCo model."""
    print(f"\n  [{label}] Model parameters:")
    print(f"    nq={model.nq}, nv={model.nv}, njnt={model.njnt}, ngeom={model.ngeom}")
    print(f"    dof_damping[6:18] = {model.dof_damping[6:18]}")
    print(f"    dof_frictionloss[6:18] = {model.dof_frictionloss[6:18]}")
    print(f"    opt.cone = {model.opt.cone} (0=pyramidal, 1=elliptic)")
    print(f"    opt.iterations = {model.opt.iterations}")
    print(f"    opt.ls_iterations = {model.opt.ls_iterations}")
    eulerdamp_disabled = bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    print(f"    eulerdamp disabled = {eulerdamp_disabled}")
    print(f"    opt.timestep = {model.opt.timestep}")

    # Collision geom sizes
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and ("foot" in name.lower() or "fl" == name.lower() or "fr" == name.lower()
                      or "rl" == name.lower() or "rr" == name.lower()):
            print(f"    geom '{name}': type={model.geom_type[gid]}, size={model.geom_size[gid]}, "
                  f"friction={model.geom_friction[gid]}")

    # Check ground/floor geom
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and "floor" in name.lower():
            print(f"    geom '{name}': type={model.geom_type[gid]}, size={model.geom_size[gid]}, "
                  f"friction={model.geom_friction[gid]}")


def torque_replay_raw(model, data, traj, max_sim_steps):
    """Replay recorded torques on a raw MuJoCo model (no RL env)."""
    tau_applied = traj["tau_applied"]
    total_steps = min(max_sim_steps, tau_applied.shape[1])

    # Set initial state
    data.qpos[:] = traj["qpos"][:, 0]
    data.qvel[:] = traj["qvel"][:, 0]
    data.ctrl[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(model, data)

    qpos_recorded = traj["qpos"]
    decimation = 4

    errors = []

    for t in range(total_steps):
        # Apply the exact recorded total torque
        data.ctrl[:12] = tau_applied[:, t]
        mujoco.mj_step(model, data)

        # Compute error at control frequency
        if (t + 1) % decimation == 0:
            rec_idx = t + 1
            if rec_idx < qpos_recorded.shape[1]:
                rec_joints = qpos_recorded[7:19, rec_idx]
                sim_joints = data.qpos[7:19]
                errors.append(np.linalg.norm(rec_joints - sim_joints))

    return np.array(errors)


def matched_gain_pd_replay(model, data, traj, max_ctrl_steps):
    """Replay using matched-gain PD formula q_target = q_des + tau_mpx / kp."""
    decimation = 4
    kp = 10.0
    kd = 2.0
    action_scale = 4.0
    default_joint_pos = traj["default_joint_pos"]

    q_des_recorded = traj["q_des"]
    tau_mpx_recorded = traj["tau_mpx"]
    qpos_recorded = traj["qpos"]
    episode_length = traj["episode_length"]

    # Set initial state
    data.qpos[:] = traj["qpos"][:, 0]
    data.qvel[:] = traj["qvel"][:, 0]
    data.ctrl[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(model, data)

    num_steps = min(episode_length, max_ctrl_steps)
    errors = []

    for ctrl_step in range(num_steps):
        sim_idx = ctrl_step * decimation

        # Matched-gain formula
        q_des = q_des_recorded[:, sim_idx]
        tau_mpx = tau_mpx_recorded[:, sim_idx]
        q_target = q_des + tau_mpx / kp

        # Convert to action and clip
        action = (q_target - default_joint_pos) / action_scale
        action = np.clip(action, -1.0, 1.0)

        # Recompute q_target after clipping
        q_target_clipped = default_joint_pos + action_scale * action

        # Run decimation substeps with PD
        for _ in range(decimation):
            q_current = data.qpos[7:19]
            dq_current = data.qvel[6:18]
            torques = kp * (q_target_clipped - q_current) + kd * (0.0 - dq_current)
            # Clip to actuator limits
            for j in range(12):
                lo = model.actuator_ctrlrange[j, 0]
                hi = model.actuator_ctrlrange[j, 1]
                torques[j] = np.clip(torques[j], lo, hi)
            data.ctrl[:12] = torques
            mujoco.mj_step(model, data)

        # Compute error
        rec_idx = (ctrl_step + 1) * decimation
        if rec_idx < qpos_recorded.shape[1]:
            rec_joints = qpos_recorded[7:19, rec_idx]
            sim_joints = data.qpos[7:19]
            errors.append(np.linalg.norm(rec_joints - sim_joints))

    return np.array(errors)


def patch_rl_model_to_mpx(rl_model, mpx_model):
    """Patch an RL model's parameters to match the MPX model exactly."""
    # Joint-level parameters
    rl_model.dof_damping[6:18] = mpx_model.dof_damping[6:18]
    rl_model.dof_frictionloss[6:18] = mpx_model.dof_frictionloss[6:18]

    # Solver options
    rl_model.opt.cone = mpx_model.opt.cone
    rl_model.opt.iterations = mpx_model.opt.iterations
    rl_model.opt.ls_iterations = mpx_model.opt.ls_iterations
    rl_model.opt.disableflags = mpx_model.opt.disableflags

    # Match collision geom sizes and positions for all non-visual geoms
    # Map geoms by name between models
    for gid in range(rl_model.ngeom):
        rl_name = mujoco.mj_id2name(rl_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if rl_name is None:
            continue
        # Find matching geom in MPX model
        mpx_gid = mujoco.mj_name2id(mpx_model, mujoco.mjtObj.mjOBJ_GEOM, rl_name)
        if mpx_gid >= 0:
            # Copy size, type, position, friction
            rl_model.geom_size[gid] = mpx_model.geom_size[mpx_gid]
            rl_model.geom_type[gid] = mpx_model.geom_type[mpx_gid]
            rl_model.geom_pos[gid] = mpx_model.geom_pos[mpx_gid]
            rl_model.geom_friction[gid] = mpx_model.geom_friction[mpx_gid]

    # Match floor friction to MPX generation settings (0.7, 0.005, 0.0)
    for gid in range(rl_model.ngeom):
        name = mujoco.mj_id2name(rl_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.lower() in {"floor", "ground"}:
            rl_model.geom_friction[gid] = [0.7, 0.005, 0.0]

    # Match foot friction to MPX generation settings
    foot_names = {"FL", "FR", "RL", "RR"}
    for gid in range(rl_model.ngeom):
        name = mujoco.mj_id2name(rl_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name in foot_names:
            rl_model.geom_friction[gid] = [0.7, 0.005, 0.0]


def print_error_summary(label, errors, control_dt):
    """Print error statistics at key time points."""
    if errors is None or len(errors) == 0:
        print(f"  {label}: No data")
        return
    print(f"  {label}:")
    print(f"    Mean={np.mean(errors):.5f}, Max={np.max(errors):.5f}, Final={errors[-1]:.5f}")
    for t_sec in [0.5, 1, 2, 3, 5, 8, 10]:
        step_idx = int(t_sec / control_dt) - 1
        if step_idx < len(errors):
            print(f"    t={t_sec:4.1f}s (step {step_idx+1:4d}): {errors[step_idx]:.5f} rad")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/quadruped")
    parser.add_argument("--filename", default=None)
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max control steps to replay")
    args = parser.parse_args()

    # Pick a trajectory
    data_dir = Path(args.data_dir)
    if args.filename:
        traj_file = data_dir / args.filename
    else:
        files = sorted(data_dir.glob("*.npz"))
        traj_file = files[0]

    print(f"Trajectory: {traj_file.name}")
    traj = load_trajectory(traj_file)
    print(f"  Episode length: {traj['episode_length']}, sim_dt: {traj['sim_dt']}, "
          f"control_dt: {traj['control_dt']}")
    max_sim_steps = args.max_steps * 4

    # -------------------------------------------------------------------
    # Load both models
    # -------------------------------------------------------------------
    mpx_model = mujoco.MjModel.from_xml_path(str(MPX_SCENE))
    mpx_model.opt.timestep = traj["sim_dt"]  # ensure dt matches
    mpx_data = mujoco.MjData(mpx_model)

    # For the RL model, we need to also create a scene with a floor.
    # Quick approach: load scene_mjx.xml but swap in go2.xml
    # Simpler: load go2.xml alone (no floor). But we need a floor for contacts.
    # Best: load the RL env the normal way.
    from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
    from mpc_rl.envs.domain_randomization import DomainRandomizationConfig

    env = QuadrupedVelocityTrackingEnv(
        robot="go2", scene="flat", render_mode=None,
        kp=10.0, kd=2.0, action_scale=4.0,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
    )
    rl_model = env.mjModel
    rl_data = env.mjData
    env.close()

    dump_model_params(mpx_model, "MPX model")
    dump_model_params(rl_model, "RL model")

    # -------------------------------------------------------------------
    # Experiment A: Torque replay on MPX model (ground truth)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Torque replay on MPX model (go2_mjx.xml)")
    print("=" * 70)
    # Set floor friction to match generation (0.7, 0.005, 0.0)
    for gid in range(mpx_model.ngeom):
        name = mujoco.mj_id2name(mpx_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.lower() in {"floor", "ground"}:
            mpx_model.geom_friction[gid] = [0.7, 0.005, 0.0]
        if name and name in {"FL", "FR", "RL", "RR"}:
            mpx_model.geom_friction[gid] = [0.7, 0.005, 0.0]

    mpx_data = mujoco.MjData(mpx_model)
    errors_a = torque_replay_raw(mpx_model, mpx_data, traj, max_sim_steps)
    print_error_summary("A: Torque on MPX model", errors_a, traj["control_dt"])

    # -------------------------------------------------------------------
    # Experiment B: Torque replay on RL model (go2.xml, unpatched)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Torque replay on RL model (go2.xml, unpatched)")
    print("=" * 70)

    # Reload RL model fresh
    env2 = QuadrupedVelocityTrackingEnv(
        robot="go2", scene="flat", render_mode=None,
        kp=10.0, kd=2.0, action_scale=4.0,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
    )
    rl_model_b = env2.mjModel
    rl_data_b = env2.mjData
    # Match generation friction on floor/feet
    for gid in range(rl_model_b.ngeom):
        name = mujoco.mj_id2name(rl_model_b, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.lower() in {"floor", "ground"}:
            rl_model_b.geom_friction[gid] = [0.7, 0.005, 0.0]
        if name and name in {"FL", "FR", "RL", "RR"}:
            rl_model_b.geom_friction[gid] = [0.7, 0.005, 0.0]
    env2.close()
    errors_b = torque_replay_raw(rl_model_b, rl_data_b, traj, max_sim_steps)
    print_error_summary("B: Torque on RL model (unpatched)", errors_b, traj["control_dt"])

    # -------------------------------------------------------------------
    # Experiment C: Matched-gain PD replay on RL model (unpatched)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Matched-gain PD replay on RL model (unpatched)")
    print("=" * 70)

    env3 = QuadrupedVelocityTrackingEnv(
        robot="go2", scene="flat", render_mode=None,
        kp=10.0, kd=2.0, action_scale=4.0,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
    )
    rl_model_c = env3.mjModel
    rl_data_c = env3.mjData
    for gid in range(rl_model_c.ngeom):
        name = mujoco.mj_id2name(rl_model_c, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.lower() in {"floor", "ground"}:
            rl_model_c.geom_friction[gid] = [0.7, 0.005, 0.0]
        if name and name in {"FL", "FR", "RL", "RR"}:
            rl_model_c.geom_friction[gid] = [0.7, 0.005, 0.0]
    env3.close()
    errors_c = matched_gain_pd_replay(rl_model_c, rl_data_c, traj, args.max_steps)
    print_error_summary("C: PD replay on RL model (unpatched)", errors_c, traj["control_dt"])

    # -------------------------------------------------------------------
    # Experiment D: PD replay on RL model with ALL params patched to MPX
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT D: PD replay on RL model with all params patched to MPX")
    print("=" * 70)

    env4 = QuadrupedVelocityTrackingEnv(
        robot="go2", scene="flat", render_mode=None,
        kp=10.0, kd=2.0, action_scale=4.0,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
    )
    rl_model_d = env4.mjModel
    rl_data_d = env4.mjData
    mpx_model_fresh = mujoco.MjModel.from_xml_path(str(MPX_SCENE))
    patch_rl_model_to_mpx(rl_model_d, mpx_model_fresh)
    env4.close()
    errors_d = matched_gain_pd_replay(rl_model_d, rl_data_d, traj, args.max_steps)
    print_error_summary("D: PD replay on RL model (fully patched)", errors_d, traj["control_dt"])

    # -------------------------------------------------------------------
    # Experiment E: PD replay directly on MPX model
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT E: Matched-gain PD replay on MPX model (go2_mjx.xml)")
    print("=" * 70)

    mpx_model_e = mujoco.MjModel.from_xml_path(str(MPX_SCENE))
    mpx_model_e.opt.timestep = traj["sim_dt"]
    for gid in range(mpx_model_e.ngeom):
        name = mujoco.mj_id2name(mpx_model_e, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.lower() in {"floor", "ground"}:
            mpx_model_e.geom_friction[gid] = [0.7, 0.005, 0.0]
        if name and name in {"FL", "FR", "RL", "RR"}:
            mpx_model_e.geom_friction[gid] = [0.7, 0.005, 0.0]
    mpx_data_e = mujoco.MjData(mpx_model_e)
    errors_e = matched_gain_pd_replay(mpx_model_e, mpx_data_e, traj, args.max_steps)
    print_error_summary("E: PD replay on MPX model", errors_e, traj["control_dt"])

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    ctrl_dt = traj["control_dt"]
    print(f"{'Experiment':<50} {'Mean':>8} {'@2s':>8} {'@5s':>8} {'@10s':>8}")
    print("-" * 82)
    for label, errs in [("A: Torque on MPX model", errors_a),
                        ("B: Torque on RL model (unpatched)", errors_b),
                        ("C: PD on RL model (unpatched)", errors_c),
                        ("D: PD on RL model (fully patched)", errors_d),
                        ("E: PD on MPX model", errors_e)]:
        if errs is None or len(errs) == 0:
            continue
        mean_e = np.mean(errs)
        t2 = errs[min(int(2/ctrl_dt)-1, len(errs)-1)]
        t5 = errs[min(int(5/ctrl_dt)-1, len(errs)-1)]
        t10 = errs[min(int(10/ctrl_dt)-1, len(errs)-1)]
        print(f"  {label:<48} {mean_e:>8.4f} {t2:>8.4f} {t5:>8.4f} {t10:>8.4f}")


if __name__ == "__main__":
    main()
