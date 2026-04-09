"""Interactive quadruped teleoperation with a trained policy.

Load a trained quadruped locomotion policy and control velocity commands
in real time using the keyboard **inside the MuJoCo viewer window**:

    Arrow Up / Down   :  increase / decrease linear velocity x  (forward / backward)
    Arrow Left / Right:  increase / decrease linear velocity y  (left / right)
    [ / ]             :  increase / decrease angular velocity z (turn left / turn right)
    R                 :  reset velocities to zero
    Escape            :  stop and exit

Usage:
    python mpc_rl/play_quad.py --model=logs/quadruped-velocity_tracking-SAC-*
    python mpc_rl/play_quad.py --model=logs/quadruped-velocity_tracking-TD3-*

The RL algorithm (SAC or TD3) is automatically detected from the directory
name so the correct SB3 class is used for loading.

Key capture uses MuJoCo viewer's native GLFW key_callback, so the viewer
window must have focus for keypresses to register (no pynput needed).
"""

import argparse
import glob
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np

# Add parent directory to path so mpc_rl is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import mujoco
import mujoco.viewer as mj_viewer

# Register custom quadruped velocity tracking environment
import mpc_rl.envs

from stable_baselines3 import SAC as SB3_SAC, TD3 as SB3_TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mpc_rl.asym_policies import AsymmetricSACPolicy, AsymmetricTD3Policy
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.sac_mpc.sb3_sac_mpc import SB3_SAC_MPC
from mpc_rl.td3_mpc.sb3_td3_mpc import SB3_TD3_MPC

# ========================================================================
# Algorithm detection
# ========================================================================

# Longer names first so "SAC-MPC" matches before "SAC".
ALGO_MAP = {
    "SAC-MPC": SB3_SAC_MPC,
    "TD3-MPC": SB3_TD3_MPC,
    "SAC": SB3_SAC,
    "TD3": SB3_TD3,
}


def detect_algorithm(run_dir: str) -> str:
    """Detect the RL algorithm from the run directory name.

    The directory naming convention from train.py is:
        quadruped-velocity_tracking-<ALGO>-<timestamp>[-suffix]
    e.g.  quadruped-velocity_tracking-SAC-20250101-120000

    Also checks config.json as a fallback.
    """
    dirname = Path(run_dir).name

    # Try to match algorithm from directory name
    for algo in ALGO_MAP:
        if f"-{algo}-" in dirname or dirname.endswith(f"-{algo}"):
            return algo

    # Fallback: check config.json inside the directory
    config_path = Path(run_dir) / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        algo = cfg.get("algorithm", "").upper()
        if algo in ALGO_MAP:
            return algo

    raise ValueError(
        f"Could not detect algorithm from directory name '{dirname}'. "
        "Expected a directory name containing '-SAC-' or '-TD3-' "
        "(e.g., quadruped-velocity_tracking-SAC-20250101-120000)."
    )


def resolve_model_path(model_arg: str) -> Path:
    """Resolve a model path that may contain shell glob patterns.

    Supports patterns like: logs/quadruped-velocity_tracking-SAC-*
    If multiple matches, picks the most recently modified directory.
    """
    # Expand glob patterns
    matches = sorted(glob.glob(model_arg))
    if not matches:
        # Try as-is (exact path)
        p = Path(model_arg)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"No matches for model path: {model_arg}\n"
            "Provide a path like: logs/quadruped-velocity_tracking-SAC-20250101-120000"
        )

    # Pick the most recently modified match
    matches.sort(key=lambda m: os.path.getmtime(m), reverse=True)
    return Path(matches[0])


# ========================================================================
# Keyboard controller (via MuJoCo viewer GLFW key_callback)
# ========================================================================

# GLFW key codes (from glfw3.h)
_KEY_UP = 265
_KEY_DOWN = 264
_KEY_LEFT = 263
_KEY_RIGHT = 262
_KEY_LEFT_BRACKET = 91    # [
_KEY_RIGHT_BRACKET = 93   # ]
_KEY_R = 82
_KEY_ESCAPE = 256


class VelocityCommander:
    """Thread-safe velocity command state controlled by MuJoCo viewer keys.

    The MuJoCo passive viewer calls key_callback from its GLFW rendering
    thread, so all state access is protected by a lock.
    """

    def __init__(self, step: float = 0.1):
        self.lin_x = 0.0
        self.lin_y = 0.0
        self.ang_z = 0.0
        self.step = step
        self.stop = False
        self._lock = threading.Lock()

    def on_key(self, keycode: int):
        """MuJoCo viewer key_callback (called from GLFW thread)."""
        with self._lock:
            if keycode == _KEY_UP:
                self.lin_x += self.step
            elif keycode == _KEY_DOWN:
                self.lin_x -= self.step
            elif keycode == _KEY_LEFT:
                self.lin_y += self.step
            elif keycode == _KEY_RIGHT:
                self.lin_y -= self.step
            elif keycode == _KEY_LEFT_BRACKET:
                self.ang_z += self.step
            elif keycode == _KEY_RIGHT_BRACKET:
                self.ang_z -= self.step
            elif keycode == _KEY_R:
                self.lin_x = 0.0
                self.lin_y = 0.0
                self.ang_z = 0.0
            elif keycode == _KEY_ESCAPE:
                self.stop = True

    def get(self) -> tuple[float, float, float]:
        """Return current (vx, vy, wz) commands (thread-safe)."""
        with self._lock:
            return self.lin_x, self.lin_y, self.ang_z

    def is_stopped(self) -> bool:
        with self._lock:
            return self.stop


# ========================================================================
# Main play loop
# ========================================================================


def make_quadruped_env(
    robot: str = "go2",
    render_mode: str | None = None,
    enable_domain_rand: bool = False,
    simple_reward: bool = False,
):
    """Create a quadruped velocity tracking gymnasium environment.

    Replay defaults to domain randomization OFF so it matches
    train.py --play_only evaluation and the nominal deployment plant.
    """
    domain_rand_cfg = None if enable_domain_rand else DomainRandomizationConfig.disabled()
    return gym.make(
        "QuadrupedVelocityTracking-v0",
        robot=robot,
        render_mode=render_mode,
        max_episode_steps=5000,
        domain_rand_cfg=domain_rand_cfg,
        simple_reward=simple_reward,
    )


def refresh_current_obs(vec_env):
    """Rebuild the current observation after changing commands on the base env."""
    obs_list = vec_env.env_method("_get_obs")
    obs = {key: np.array([obs_list[0][key]]) for key in obs_list[0]}
    if hasattr(vec_env, "normalize_obs"):
        obs = vec_env.normalize_obs(obs)
    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Interactively control a trained quadruped policy with keyboard."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained model directory (supports glob patterns, e.g. logs/quadruped-*-SAC-*)",
    )
    parser.add_argument(
        "--robot", type=str, default="go2",
        help="Quadruped robot model (default: go2)",
    )
    parser.add_argument(
        "--step", type=float, default=0.1,
        help="Velocity increment per key press (default: 0.1)",
    )
    parser.add_argument(
        "--domain_rand", action="store_true",
        help="Enable domain randomization during replay (default: disabled to match eval/deployment)",
    )
    args = parser.parse_args()

    # Resolve model directory
    run_dir = resolve_model_path(args.model)
    print(f"Run directory: {run_dir}")

    # Detect algorithm
    algo_name = detect_algorithm(str(run_dir))
    algo_class = ALGO_MAP[algo_name]
    is_mpc_algo = algo_name in {"SAC-MPC", "TD3-MPC"}
    print(f"Detected algorithm: {algo_name}")

    # Locate model file
    model_path = run_dir / "final_model"
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        # Try best_model
        best = run_dir / "best_model" / "best_model"
        if best.with_suffix(".zip").exists() or best.exists():
            model_path = best
            print("Using best_model checkpoint")
        else:
            raise FileNotFoundError(
                f"No model found at {model_path} or {best}. "
                "Make sure the run directory contains final_model.zip or best_model/best_model.zip"
            )

    # Create environment WITHOUT render_mode (we create the viewer ourselves)
    print(f"Creating quadruped environment (robot={args.robot})...")
    env_wrapped = make_quadruped_env(
        robot=args.robot,
        render_mode=None,
        enable_domain_rand=args.domain_rand,
        simple_reward=is_mpc_algo,
    )
    # gym.make() wraps in TimeLimit; unwrap to access QuadrupedVelocityTrackingEnv
    env_base = env_wrapped.unwrapped
    vec_env = DummyVecEnv([lambda: env_wrapped])
    print(
        "Domain randomization during replay: "
        f"{'ENABLED' if env_base.domain_rand_cfg.enable else 'DISABLED'}"
    )

    # Load normalization stats if available
    vec_normalize_path = run_dir / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        vec_env = VecNormalize.load(str(vec_normalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("Loaded observation normalization stats")

    # Load model
    print(f"Loading {algo_name} model from: {model_path}")
    model = algo_class.load(str(model_path), env=vec_env)
    print("Model loaded successfully!")

    # Create velocity commander and MuJoCo viewer with key callback
    commander = VelocityCommander(step=args.step)
    viewer = mj_viewer.launch_passive(
        env_base.mjModel,
        env_base.mjData,
        key_callback=commander.on_key,
        show_left_ui=False,
        show_right_ui=False,
    )
    mujoco.mjv_defaultFreeCamera(env_base.mjModel, viewer.cam)

    # Real-time sync: sleep for control_dt each iteration so simulation matches wall time
    control_dt = env_base.control_dt  # 0.02s for 50 Hz control

    print("\n" + "=" * 60)
    print("QUADRUPED TELEOPERATION  (press keys in the viewer window)")
    print("=" * 60)
    print(f"  Control frequency: {1.0/control_dt:.0f} Hz  (control_dt={control_dt:.4f}s)")
    print("  Up / Down    :  forward / backward  (lin_vel_x)")
    print("  Left / Right :  strafe left / right  (lin_vel_y)")
    print("  [ / ]        :  turn left / right    (ang_vel_z)")
    print("  R            :  reset all velocities to zero")
    print("  Escape       :  stop and exit")
    print("=" * 60 + "\n")

    # Main loop
    obs = vec_env.reset()
    env_base.set_commands(vx=0.0, vy=0.0, wz=0.0)
    obs = refresh_current_obs(vec_env)

    try:
        while not commander.is_stopped() and viewer.is_running():
            step_start = time.perf_counter()

            # Update velocity commands from keyboard
            vx, vy, wz = commander.get()
            env_base.set_commands(vx=vx, vy=vy, wz=wz)
            obs = refresh_current_obs(vec_env)

            # Run policy
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            # Update viewer camera to follow robot and sync
            base_pos = env_base.mjData.qpos[0:3]
            viewer.cam.lookat[:] = base_pos
            viewer.sync()

            # Print current commands periodically
            step_count = env_base._step_count
            if step_count % 50 == 0:
                base_vel = env_base._base_lin_vel_body()
                print(
                    f"  cmd: vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}  |  "
                    f"actual: vx={base_vel[0]:+.2f}  vy={base_vel[1]:+.2f}  |  "
                    f"step={step_count}"
                )

            # Reset on termination (robot fell)
            if done[0]:
                print("  [Episode reset - robot terminated]")
                obs = vec_env.reset()
                env_base.set_commands(vx=vx, vy=vy, wz=wz)
                obs = refresh_current_obs(vec_env)

            # -- Real-time synchronization ------------------------------
            # Sleep so that each control step takes exactly control_dt of
            # wall time, matching the real robot's 50 Hz control loop.
            elapsed = time.perf_counter() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        viewer.close()
        vec_env.close()
        print("Environment closed. Goodbye!")


if __name__ == "__main__":
    main()
