"""
Replay and validate cheetah3 MPC trajectory files.

Usage:
    python tests/gen_traj_data_cheetah3_test.py --data-dir=data/cheetah3_0_010dt
    python tests/gen_traj_data_cheetah3_test.py --data-dir=data/cheetah3_0_010dt --no-render --no-plots
"""

import argparse
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

# Add parent directory to path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.envs.cheetah3_common import (
    cheetah3_pose_diagnostics,
    is_valid_cheetah3_initial_state,
)


CHEETAH3_MODEL_PATH = Path(__file__).parent.parent / "mpc_rl/tasks/cheetah/task.xml"


def _trajectory_initial_state(traj):
    qpos = traj["init_qpos"] if "init_qpos" in traj else traj["qpos"][:, 0]
    qvel = traj["init_qvel"] if "init_qvel" in traj else traj["qvel"][:, 0]
    return qpos, qvel


def _format_pose_diagnostics(diagnostics):
    return (
        f"min_geom_z={diagnostics['min_geom_z']:.4f}, "
        f"torso_upright_z={diagnostics['torso_upright_z']:.4f}, "
        f"torso_height={diagnostics['torso_height']:.4f}"
    )


def _valid_trajectory_file(model, traj_file):
    try:
        with np.load(traj_file) as traj:
            init_qpos, init_qvel = _trajectory_initial_state(traj)
            valid = is_valid_cheetah3_initial_state(model, init_qpos, init_qvel)
            diagnostics = cheetah3_pose_diagnostics(model, init_qpos, init_qvel)
            return valid, diagnostics
    except Exception as exc:
        return False, {"error": str(exc)}


def select_trajectory(data_dir, random_select=True, filename=None, model=None, require_valid=True):
    """Select a trajectory file from the data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    traj_files = sorted(data_dir.glob("*.npz"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")

    if model is None:
        model = mujoco.MjModel.from_xml_path(str(CHEETAH3_MODEL_PATH))

    if random_select:
        if require_valid:
            valid_files = []
            invalid_count = 0
            for traj_file in traj_files:
                valid, _ = _valid_trajectory_file(model, traj_file)
                if valid:
                    valid_files.append(traj_file)
                else:
                    invalid_count += 1
            if not valid_files:
                raise RuntimeError(
                    f"No valid cheetah3 trajectory files found in {data_dir}. "
                    "Regenerate trajectories with the updated initialization sampler."
                )
            if invalid_count:
                print(
                    f"Skipped {invalid_count} invalid trajectory file(s) with clipped or non-upright initial poses."
                )
            selected_file = random.choice(valid_files)
        else:
            selected_file = random.choice(traj_files)
        print(f"Randomly selected: {selected_file.name}")
    else:
        selected_file = data_dir / filename
        if not selected_file.exists():
            raise FileNotFoundError(f"File not found: {selected_file}")
        if require_valid:
            valid, diagnostics = _valid_trajectory_file(model, selected_file)
            if not valid:
                details = diagnostics.get("error", _format_pose_diagnostics(diagnostics))
                raise RuntimeError(
                    f"Selected trajectory has an invalid cheetah3 initial pose: {details}. "
                    "Regenerate this trajectory with the updated initialization sampler."
                )
        print(f"Selected: {selected_file.name}")
    return selected_file


def _validate_shapes(model, qpos, qvel, ctrl, time):
    if qpos.shape[0] != model.nq:
        raise AssertionError(f"qpos rows {qpos.shape[0]} != model.nq {model.nq}")
    if qvel.shape[0] != model.nv:
        raise AssertionError(f"qvel rows {qvel.shape[0]} != model.nv {model.nv}")
    if ctrl.shape[0] != model.nu:
        raise AssertionError(f"ctrl rows {ctrl.shape[0]} != model.nu {model.nu}")
    if qpos.shape[1] != qvel.shape[1] or qpos.shape[1] != time.shape[0]:
        raise AssertionError("qpos, qvel, and time lengths do not match")
    if ctrl.shape[1] != qpos.shape[1] - 1:
        raise AssertionError("ctrl must have one fewer step than qpos/qvel/time")


def _validate_timestep(model, traj, time):
    expected_dt = float(model.opt.timestep)
    dt = np.diff(time)
    if not np.allclose(dt, expected_dt, rtol=0.0, atol=1.0e-12):
        raise AssertionError(
            f"Saved time spacing does not match model timestep {expected_dt}: "
            f"min={dt.min()}, max={dt.max()}"
        )
    if "physics_timestep" in traj:
        saved_dt = float(traj["physics_timestep"])
        if not np.isclose(saved_dt, expected_dt, rtol=0.0, atol=1.0e-12):
            raise AssertionError(
                f"Saved physics_timestep {saved_dt} != model timestep {expected_dt}"
            )
    print(f"Physics timestep verified: {expected_dt:.6f}s")

    if "agent_timestep" in traj:
        agent_dt = float(traj["agent_timestep"])
        steps_per_agent_update = int(round(agent_dt / expected_dt))
        print(
            f"Agent timestep metadata: {agent_dt:.6f}s "
            f"({steps_per_agent_update} physics steps/action update)"
        )


def replay_cheetah3_trajectory(
    data_dir,
    random_select=True,
    filename=None,
    max_steps=500,
    tolerance=1.0e-9,
    render=True,
    plots=True,
    require_valid_init=True,
    video_path=None,
):
    """
    Replay a saved cheetah3 trajectory in the same MuJoCo model.

    Controls are applied at the MuJoCo physics timestep, not downsampled. This
    matches how `MPCPlanner` records transitions for replay-buffer injection.
    """
    model = mujoco.MjModel.from_xml_path(str(CHEETAH3_MODEL_PATH))
    traj_file = select_trajectory(
        data_dir,
        random_select=random_select,
        filename=filename,
        model=model,
        require_valid=require_valid_init,
    )

    print(f"\nLoading trajectory from: {traj_file}")
    traj = np.load(traj_file)
    qpos_ref = traj["qpos"]
    qvel_ref = traj["qvel"]
    ctrl = traj["ctrl"]
    time_ref = traj["time"]
    init_qpos = traj["init_qpos"]
    init_qvel = traj["init_qvel"]

    _validate_shapes(model, qpos_ref, qvel_ref, ctrl, time_ref)
    _validate_timestep(model, traj, time_ref)
    init_diagnostics = cheetah3_pose_diagnostics(model, init_qpos, init_qvel)
    print(f"Initial pose diagnostics: {_format_pose_diagnostics(init_diagnostics)}")
    if require_valid_init and not is_valid_cheetah3_initial_state(model, init_qpos, init_qvel):
        raise RuntimeError(
            "Trajectory initial pose is invalid. "
            f"{_format_pose_diagnostics(init_diagnostics)}"
        )

    print(f"Initial state: qpos shape={init_qpos.shape}, qvel shape={init_qvel.shape}")
    print(f"Trajectory qpos shape: {qpos_ref.shape}")
    print(f"Trajectory qvel shape: {qvel_ref.shape}")
    print(f"Control shape: {ctrl.shape}")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = init_qpos
    data.qvel[:] = init_qvel
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera="side")
        frames.append(renderer.render().copy())

    sim_qpos = [data.qpos.copy()]
    sim_qvel = [data.qvel.copy()]

    num_steps = min(ctrl.shape[1], max_steps)
    print(f"\nReplaying {num_steps} physics steps...")
    for t in range(num_steps):
        data.ctrl[:] = ctrl[:, t]
        mujoco.mj_step(model, data)
        sim_qpos.append(data.qpos.copy())
        sim_qvel.append(data.qvel.copy())

        if renderer is not None:
            renderer.update_scene(data, camera="side")
            frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()

    sim_qpos = np.stack(sim_qpos, axis=1)
    sim_qvel = np.stack(sim_qvel, axis=1)

    qpos_ref_eval = qpos_ref[:, : num_steps + 1]
    qvel_ref_eval = qvel_ref[:, : num_steps + 1]
    qpos_abs_err = np.abs(sim_qpos - qpos_ref_eval)
    qvel_abs_err = np.abs(sim_qvel - qvel_ref_eval)
    qpos_max = float(qpos_abs_err.max())
    qvel_max = float(qvel_abs_err.max())

    print("\nReplay vs saved trajectory error:")
    print(f"  qpos max abs error: {qpos_max:.3e}")
    print(f"  qpos mean abs error: {qpos_abs_err.mean():.3e}")
    print(f"  qvel max abs error: {qvel_max:.3e}")
    print(f"  qvel mean abs error: {qvel_abs_err.mean():.3e}")

    if qpos_max > tolerance or qvel_max > tolerance:
        raise AssertionError(
            f"Replay diverged beyond tolerance {tolerance}: "
            f"qpos_max={qpos_max}, qvel_max={qvel_max}"
        )

    if video_path is not None:
        if not frames:
            raise RuntimeError("--video-path requires rendering; do not pass --no-render")
        import mediapy as media

        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(round(1.0 / model.opt.timestep))
        media.write_video(str(video_path), frames, fps=fps)
        print(f"Replay video saved to: {video_path}")

    if plots:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        num_actuators_to_plot = min(4, ctrl.shape[0])
        _, axes = plt.subplots(
            num_actuators_to_plot, 1, figsize=(10, 2.5 * num_actuators_to_plot)
        )
        if num_actuators_to_plot == 1:
            axes = [axes]
        for idx in range(num_actuators_to_plot):
            axes[idx].plot(time_ref[:-1][:num_steps], ctrl[idx, :num_steps])
            axes[idx].set_ylabel(f"u[{idx}]")
            axes[idx].grid(True)
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()

        if frames:
            fig_anim = plt.figure(figsize=(10, 6))
            img = plt.imshow(frames[0])
            plt.axis("off")
            plt.title(f"cheetah3 replay: {traj_file.name}")

            def animate_frame(i):
                img.set_data(frames[i])
                return [img]

            anim = animation.FuncAnimation(
                fig_anim,
                animate_frame,
                frames=len(frames),
                interval=1000 * model.opt.timestep,
                blit=True,
                repeat=True,
            )
            fig_anim._cheetah3_anim = anim
        plt.show()

    print("\nReplay validation passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay and validate cheetah3 MPC trajectories"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Optional path to write an MP4 replay video.",
    )
    parser.add_argument(
        "--allow-invalid-init",
        action="store_true",
        help="Allow replaying old trajectories whose initial pose is clipped or non-upright.",
    )
    args = parser.parse_args()

    replay_cheetah3_trajectory(
        data_dir=args.data_dir,
        random_select=(args.filename is None),
        filename=args.filename,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
        render=not args.no_render,
        plots=not args.no_plots,
        require_valid_init=not args.allow_invalid_init,
        video_path=args.video_path,
    )


if __name__ == "__main__":
    main()
