"""Diagnose why replayed MPC trajectories diverge from recorded ones.

Compares MuJoCo model parameters between the RL env (go2.xml) and the MPX
generation env (go2_mjx.xml), then runs controlled replay experiments to
isolate each source of divergence.
"""

import numpy as np
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig


def compare_model_parameters():
    """Load both MuJoCo models and compare physics parameters."""
    print("=" * 70)
    print("MODEL PARAMETER COMPARISON")
    print("=" * 70)

    # Create RL env to get its model
    env = QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        render_mode=None,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
    )
    rl_model = env.mjModel

    # Load MPX model directly
    mpx_xml = Path(__file__).parent.parent / "deps" / "mpx" / "mpx" / "data" / "go2" / "go2_mjx.xml"
    # Need the scene file or a standalone version. Try loading scene_mjx.xml
    mpx_scene = mpx_xml.parent / "scene_mjx.xml"
    if mpx_scene.exists():
        mpx_model = mujoco.MjModel.from_xml_path(str(mpx_scene))
    else:
        mpx_model = mujoco.MjModel.from_xml_path(str(mpx_xml))

    print(f"\nRL model:  nq={rl_model.nq}, nv={rl_model.nv}, njnt={rl_model.njnt}")
    print(f"MPX model: nq={mpx_model.nq}, nv={mpx_model.nv}, njnt={mpx_model.njnt}")

    # -- Joint parameters --
    print(f"\n--- Joint Parameters (first {min(rl_model.njnt, 13)} joints) ---")
    print(f"{'Joint':<20} {'RL damping':>12} {'MPX damping':>12} {'RL frictloss':>14} {'MPX frictloss':>14}")
    print("-" * 74)

    for i in range(min(rl_model.njnt, mpx_model.njnt)):
        rl_name = mujoco.mj_id2name(rl_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        mpx_name = mujoco.mj_id2name(mpx_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        rl_damp = rl_model.jnt_stiffness[i] if hasattr(rl_model, 'jnt_stiffness') else 0
        mpx_damp = mpx_model.jnt_stiffness[i] if hasattr(mpx_model, 'jnt_stiffness') else 0
        print(f"{rl_name or f'joint_{i}':<20} "
              f"{rl_model.dof_damping[i] if i < rl_model.nv else 'N/A':>12} "
              f"{mpx_model.dof_damping[i] if i < mpx_model.nv else 'N/A':>12} "
              f"{rl_model.dof_frictionloss[i] if i < rl_model.nv else 'N/A':>14} "
              f"{mpx_model.dof_frictionloss[i] if i < mpx_model.nv else 'N/A':>14}")

    # -- Solver options --
    print(f"\n--- Solver Options ---")
    print(f"  RL  cone: {rl_model.opt.cone}  (0=pyramidal, 1=elliptic)")
    print(f"  MPX cone: {mpx_model.opt.cone}")
    print(f"  RL  iterations: {rl_model.opt.iterations}, ls_iterations: {rl_model.opt.ls_iterations}")
    print(f"  MPX iterations: {mpx_model.opt.iterations}, ls_iterations: {mpx_model.opt.ls_iterations}")

    # Check eulerdamp flag
    # MjModel.opt.disableflags contains the eulerdamp disable bit
    eulerdamp_bit = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    rl_eulerdamp_disabled = bool(rl_model.opt.disableflags & eulerdamp_bit)
    mpx_eulerdamp_disabled = bool(mpx_model.opt.disableflags & eulerdamp_bit)
    print(f"  RL  eulerdamp disabled: {rl_eulerdamp_disabled}")
    print(f"  MPX eulerdamp disabled: {mpx_eulerdamp_disabled}")

    # -- Foot geom sizes --
    print(f"\n--- Foot Geom Sizes ---")
    for model, label in [(rl_model, "RL"), (mpx_model, "MPX")]:
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name and "foot" in name.lower():
                print(f"  {label} {name}: size={model.geom_size[gid]}, "
                      f"friction={model.geom_friction[gid]}")

    # -- Summary of differences --
    print(f"\n{'=' * 70}")
    print("CRITICAL DIFFERENCES FOUND:")
    print("=" * 70)

    # Check joint damping
    rl_joint_damping = rl_model.dof_damping[6:18]  # skip 6 base DOFs
    mpx_joint_damping = mpx_model.dof_damping[6:18]
    if not np.allclose(rl_joint_damping, mpx_joint_damping):
        print(f"  [MISMATCH] Joint damping: RL={rl_joint_damping[0]}, MPX={mpx_joint_damping[0]}")
        print(f"    -> MuJoCo applies tau_damp = -damping * qvel at EVERY sim step")
        print(f"    -> Difference of {abs(rl_joint_damping[0] - mpx_joint_damping[0]):.1f} * qvel per step compounds over time")

    # Check frictionloss
    rl_frictloss = rl_model.dof_frictionloss[6:18]
    mpx_frictloss = mpx_model.dof_frictionloss[6:18]
    if not np.allclose(rl_frictloss, mpx_frictloss):
        print(f"  [MISMATCH] Joint frictionloss: RL={rl_frictloss[0]}, MPX={mpx_frictloss[0]}")
        print(f"    -> Coulomb friction at joints: dissipates energy in RL but not in MPX")

    # Check cone type
    if rl_model.opt.cone != mpx_model.opt.cone:
        cone_names = {0: "pyramidal", 1: "elliptic"}
        print(f"  [MISMATCH] Contact cone: RL={cone_names.get(rl_model.opt.cone, '?')}, "
              f"MPX={cone_names.get(mpx_model.opt.cone, '?')}")
        print(f"    -> Different contact force approximation affects ground reaction forces")

    # Check eulerdamp
    if rl_eulerdamp_disabled != mpx_eulerdamp_disabled:
        print(f"  [MISMATCH] Euler damping: RL disabled={rl_eulerdamp_disabled}, "
              f"MPX disabled={mpx_eulerdamp_disabled}")
        print(f"    -> Implicit damping in Euler integrator differs")

    env.close()
    return rl_model, mpx_model


def replay_with_model_params_patched(data_file, max_steps=500):
    """Replay a trajectory with RL model patched to match MPX model parameters.

    Patches the RL env's MuJoCo model to use MPX-matching joint damping,
    frictionloss, cone type, and euler damping flags, then replays.
    """
    print("\n" + "=" * 70)
    print("REPLAY EXPERIMENT: PATCHING RL MODEL TO MATCH MPX")
    print("=" * 70)

    # Load trajectory
    data = np.load(data_file, allow_pickle=True)
    traj = {
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
    print(f"Trajectory: {data_file}")
    print(f"  Episode length: {traj['episode_length']}, sim_dt: {traj['sim_dt']}")

    configs = {
        "1_unpatched": {
            "joint_damping": None,    # use RL default (0.6)
            "frictionloss": None,     # use RL default (0.2)
            "cone": None,             # use RL default (elliptic)
            "eulerdamp": None,        # use RL default (enabled)
        },
        "2_damping_only": {
            "joint_damping": 2.0,     # match MPX
            "frictionloss": None,
            "cone": None,
            "eulerdamp": None,
        },
        "3_frictionloss_only": {
            "joint_damping": None,
            "frictionloss": 0.0,      # match MPX
            "cone": None,
            "eulerdamp": None,
        },
        "4_damping_and_frictionloss": {
            "joint_damping": 2.0,
            "frictionloss": 0.0,
            "cone": None,
            "eulerdamp": None,
        },
        "5_all_patched": {
            "joint_damping": 2.0,
            "frictionloss": 0.0,
            "cone": 0,                # pyramidal to match MPX
            "eulerdamp": True,        # disable to match MPX
        },
    }

    results = {}
    for label, patches in configs.items():
        print(f"\n--- Config: {label} ---")
        errors = _run_replay(traj, max_steps, patches)
        results[label] = errors
        if errors is not None:
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            print(f"  Mean L2 error: {mean_err:.4f} rad, Max: {max_err:.4f} rad")
            # Show error at various time points
            for t_sec in [1, 2, 5, 10]:
                step_idx = int(t_sec / traj["control_dt"]) - 1
                if step_idx < len(errors):
                    print(f"  Error at t={t_sec}s (step {step_idx+1}): {errors[step_idx]:.4f} rad")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Mean Joint L2 Error by Configuration")
    print("=" * 70)
    print(f"{'Config':<35} {'Mean Error':>12} {'Max Error':>12} {'Error@2s':>12} {'Error@5s':>12}")
    print("-" * 83)
    for label, errors in results.items():
        if errors is not None:
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            t2_idx = min(int(2 / traj["control_dt"]) - 1, len(errors) - 1)
            t5_idx = min(int(5 / traj["control_dt"]) - 1, len(errors) - 1)
            print(f"  {label:<33} {mean_err:>12.4f} {max_err:>12.4f} "
                  f"{errors[t2_idx]:>12.4f} {errors[t5_idx]:>12.4f}")


def _run_replay(traj, max_steps, patches):
    """Run a single replay experiment with the specified model patches."""
    env = QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        render_mode=None,
        kp=10.0,
        kd=2.0,
        action_scale=4.0,
        domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
    )

    # Match generation friction
    _match_generation_friction(env, tangential=0.7, torsional=0.005, rolling=0.0)

    # Apply patches to match MPX model
    if patches.get("joint_damping") is not None:
        env.mjModel.dof_damping[6:18] = patches["joint_damping"]
        print(f"  Patched joint_damping to {patches['joint_damping']}")

    if patches.get("frictionloss") is not None:
        env.mjModel.dof_frictionloss[6:18] = patches["frictionloss"]
        print(f"  Patched frictionloss to {patches['frictionloss']}")

    if patches.get("cone") is not None:
        env.mjModel.opt.cone = patches["cone"]
        print(f"  Patched cone to {patches['cone']}")

    if patches.get("eulerdamp"):
        env.mjModel.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        print(f"  Disabled eulerdamp")

    # Disable early termination
    env.max_pitch = np.pi
    env.max_roll = np.pi
    env.min_base_height = -1.0

    # Reset and set initial state
    env.reset(seed=42)
    env.mjData.qpos[:] = traj["qpos"][:, 0]
    env.mjData.qvel[:] = traj["qvel"][:, 0]
    env.mjData.ctrl[:] = 0.0
    env.mjData.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.mjModel, env.mjData)
    env._last_action[:] = 0.0
    env._prev_last_action[:] = 0.0
    env._last_joint_vel = env.mjData.qvel[6:].copy()

    # Replay using matched-gain formula
    sim_dt = traj["sim_dt"]
    control_dt = traj["control_dt"]
    decimation = int(round(control_dt / sim_dt))
    default_joint_pos = traj["default_joint_pos"]
    q_des_recorded = traj["q_des"]
    tau_mpx_recorded = traj["tau_mpx"]
    commands_recorded = traj["commands"]
    episode_length = traj["episode_length"]
    qpos_recorded = traj["qpos"]

    kp = env.kp
    num_steps = min(episode_length, max_steps)
    errors = []

    for ctrl_step in range(num_steps):
        sim_idx = ctrl_step * decimation
        q_des = q_des_recorded[:, sim_idx]
        tau_mpx = tau_mpx_recorded[:, sim_idx]
        q_target = q_des + tau_mpx / kp

        action = (q_target - default_joint_pos) / env.action_scale
        action = np.clip(action, -1.0, 1.0)

        cmd = commands_recorded[:, sim_idx]
        env.set_commands(vx=cmd[0], vy=cmd[1], wz=cmd[2])

        obs, reward, terminated, truncated, info = env.step(action)

        # Compute joint position error vs recorded
        rec_idx = (ctrl_step + 1) * decimation
        if rec_idx < qpos_recorded.shape[1]:
            rec_joints = qpos_recorded[7:19, rec_idx]
            env_joints = env.mjData.qpos[7:19]
            errors.append(np.linalg.norm(rec_joints - env_joints))

        if terminated or truncated:
            print(f"  Episode ended at step {ctrl_step+1} (terminated={terminated})")
            break

    env.close()
    return np.array(errors) if errors else None


def _match_generation_friction(env, tangential=0.7, torsional=0.005, rolling=0.0):
    """Set floor and foot geom friction to match the generation environment."""
    friction = np.array([tangential, torsional, rolling])
    floor_names = {"ground", "floor", "hfield", "terrain"}
    foot_geom_ids = set(env._foot_geom_ids.values())

    for geom_id in range(env.mjModel.ngeom):
        geom_name = mujoco.mj_id2name(
            env.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if (geom_name and geom_name.lower() in floor_names) or geom_id in foot_geom_ids:
            env.mjModel.geom_friction[geom_id, :] = friction


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/quadruped")
    parser.add_argument("--filename", default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    # Compare models first
    compare_model_parameters()

    # Pick a trajectory
    data_dir = Path(args.data_dir)
    if args.filename:
        traj_file = data_dir / args.filename
    else:
        files = sorted(data_dir.glob("*.npz"))
        traj_file = files[0]  # Use first deterministically

    # Run replay experiments
    replay_with_model_params_patched(str(traj_file), max_steps=args.max_steps)
