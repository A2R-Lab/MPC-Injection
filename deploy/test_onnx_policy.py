"""Test an exported ONNX policy in the MuJoCo simulator.

This is the ONNX-equivalent of mpc_rl/play_quad.py.  It loads the exported
policy.onnx file and runs it through the same gymnasium environment used during
training, verifying that the ONNX model behaves identically to the original
PyTorch checkpoint.

The key difference from play_quad.py is *how the action is computed*:

    play_quad.py (SB3 model)
        obs_dict  ->  VecNormalize  ->  PolicyFeaturesExtractor  ->  actor.latent_pi
                  ->  actor.mu  ->  tanh  ->  action

    this script (ONNX)
        obs_dict["policy"]  ->  [normalisation baked into ONNX]  ->  [MLP in ONNX]
                            ->  [tanh in ONNX]  ->  action

Both produce identical actions for the same raw sensor input, which confirms
that the ONNX export is correct and ready for real-robot deployment.

Usage
-----
    python deploy/test_onnx_policy.py \\
        --onnx deploy/robots/go2/config/policy/velocity/v0/exported/policy.onnx

    # Side-by-side comparison with the original SB3 model:
    python deploy/test_onnx_policy.py \\
        --onnx deploy/robots/go2/config/policy/velocity/v0/exported/policy.onnx \\
        --compare logs/quadruped-velocity_tracking-SAC-20260305-170808/final_model.zip \\
        --vecnorm logs/quadruped-velocity_tracking-SAC-20260305-170808/vec_normalize.pkl

Key controls (in the viewer window)
------------------------------------
    Up / Down    :  forward / backward  (lin_vel_x)
    Left / Right :  strafe left / right (lin_vel_y)
    [ / ]        :  turn left / right   (ang_vel_z)
    R            :  reset velocities to zero
    Escape       :  stop and exit
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import mujoco
import mujoco.viewer as mj_viewer
import onnxruntime as ort

import mpc_rl.envs  # register QuadrupedVelocityTracking-v0
from stable_baselines3.common.vec_env import DummyVecEnv
from mpc_rl.envs.domain_randomization import (
    DomainRandomizationConfig,
    STARTUP_DOMAIN_RAND_PRESET_NAMES,
)
from mpc_rl.envs.go2_sysid import assert_go2_sysid_joint_dynamics

# =============================================================================
# GLFW key codes (identical to play_quad.py)
# =============================================================================
_KEY_UP = 265
_KEY_DOWN = 264
_KEY_LEFT = 263
_KEY_RIGHT = 262
_KEY_LEFT_BRACKET = 91
_KEY_RIGHT_BRACKET = 93
_KEY_R = 82
_KEY_ESCAPE = 256


class VelocityCommander:
    def __init__(self, step: float = 0.1):
        self.lin_x = 0.0
        self.lin_y = 0.0
        self.ang_z = 0.0
        self.step = step
        self.stop = False
        self._lock = threading.Lock()

    def on_key(self, keycode: int):
        with self._lock:
            if keycode == _KEY_UP:       self.lin_x += self.step
            elif keycode == _KEY_DOWN:   self.lin_x -= self.step
            elif keycode == _KEY_LEFT:   self.lin_y += self.step
            elif keycode == _KEY_RIGHT:  self.lin_y -= self.step
            elif keycode == _KEY_LEFT_BRACKET:  self.ang_z += self.step
            elif keycode == _KEY_RIGHT_BRACKET: self.ang_z -= self.step
            elif keycode == _KEY_R:
                self.lin_x = 0.0; self.lin_y = 0.0; self.ang_z = 0.0
            elif keycode == _KEY_ESCAPE: self.stop = True

    def get(self): 
        with self._lock: return self.lin_x, self.lin_y, self.ang_z

    def is_stopped(self):
        with self._lock: return self.stop


# =============================================================================
# Optional side-by-side SB3 comparison
# =============================================================================

def load_sb3_model(model_zip: Path, vecnorm_pkl: Path, *, use_go2_sysid: bool):
    """Load original SB3 SAC or TD3 model + VecNormalize for action comparison.

    The algorithm is auto-detected from the directory name (same logic as
    export_onnx_go2.py and play_quad.py).
    """
    # Inline the same detection logic used by export_onnx_go2.py
    from mpc_rl.asym_policies import AsymmetricSACPolicy, AsymmetricTD3Policy
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.vec_env import VecNormalize

    algo = None
    for part in reversed(model_zip.parts):
        for a in ("SAC", "TD3"):
            if f"-{a}-" in part or part.endswith(f"-{a}"):
                algo = a
                break
        if algo:
            break
    if algo is None:
        # Fallback: config.json
        for d in (model_zip.parent, model_zip.parent.parent):
            cfg_path = d / "config.json"
            if cfg_path.exists():
                import json
                with open(cfg_path) as f:
                    algo = json.load(f).get("algorithm", "").upper().replace("-MPC", "")
                if algo in ("SAC", "TD3"):
                    break
    if algo not in ("SAC", "TD3"):
        print(f"WARNING: could not detect algorithm from path, defaulting to SAC")
        algo = "SAC"

    print(f"  Detected algorithm for comparison: {algo}")
    algo_cls = SAC if algo == "SAC" else TD3

    # Need a dummy env to load VecNormalize; we only use it for obs_rms stats
    dummy_env = gym.make(
        "QuadrupedVelocityTracking-v0",
        robot="go2",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        use_go2_sysid=use_go2_sysid,
    )
    if use_go2_sysid:
        assert_go2_sysid_joint_dynamics(dummy_env.unwrapped.mjModel)
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    vec_norm = VecNormalize.load(str(vecnorm_pkl), dummy_vec)
    vec_norm.training = False
    vec_norm.norm_reward = False

    # Force CPU so the comparison path matches ONNXRuntime and never depends
    # on whether the local machine auto-selects CUDA.
    model = algo_cls.load(str(model_zip), env=vec_norm, device="cpu")
    model.policy = model.policy.cpu()
    return model, vec_norm


def predict_sb3_action_from_raw_obs(sb3_model, sb3_vec_norm, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
    """Match deployment semantics: raw obs -> VecNormalize -> deterministic SB3 action."""
    raw_obs_copy = {key: value.copy() for key, value in raw_obs.items()}
    norm_obs = sb3_vec_norm.normalize_obs(raw_obs_copy)
    sb3_actions, _ = sb3_model.predict(norm_obs, deterministic=True)
    return sb3_actions


def refresh_current_raw_obs(vec_env) -> dict[str, np.ndarray]:
    """Rebuild the latest raw Dict observation after externally changing commands."""
    obs_list = vec_env.env_method("_get_obs")
    return {key: np.array([obs_list[0][key]], dtype=np.float32) for key in obs_list[0]}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test ONNX policy in MuJoCo simulator (ONNX-equivalent of play_quad.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--onnx", type=Path, required=True,
        help="Path to the exported policy.onnx file",
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
        "--compare", type=Path, default=None,
        help="(Optional) SB3 model .zip for side-by-side action comparison",
    )
    parser.add_argument(
        "--vecnorm", type=Path, default=None,
        help="(Optional) VecNormalize .pkl paired with --compare model",
    )
    parser.add_argument(
        "--domain_rand", action="store_true",
        help=(
            "Backward-compatible shorthand for "
            "--domain_rand_config_type=default_no_push"
        ),
    )
    parser.add_argument(
        "--domain_rand_config_type",
        type=str,
        default=None,
        choices=STARTUP_DOMAIN_RAND_PRESET_NAMES,
        help=(
            "Named startup DR preset for ONNX replay. "
            "Overrides --domain_rand when provided."
        ),
    )
    parser.add_argument(
        "--use_go2_sysid",
        dest="use_go2_sysid",
        action="store_true",
        default=True,
        help="Enable the Go2 sysID joint-dynamics patch in the simulator (default: enabled).",
    )
    parser.add_argument(
        "--no_use_go2_sysid",
        dest="use_go2_sysid",
        action="store_false",
        help="Disable the Go2 sysID joint-dynamics patch in the simulator.",
    )
    args = parser.parse_args()

    if not args.onnx.exists():
        parser.error(f"ONNX file not found: {args.onnx}")

    if args.compare and not args.vecnorm:
        parser.error("--vecnorm is required when --compare is provided")

    # -- Load ONNX session -----------------------------------------------------
    print(f"Loading ONNX policy from: {args.onnx}")
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name   # should be "obs"
    output_name = sess.get_outputs()[0].name  # should be "actions"
    obs_dim     = sess.get_inputs()[0].shape[1]
    action_dim  = sess.get_outputs()[0].shape[1]
    print(f"  Input : '{input_name}'  shape (1, {obs_dim})")
    print(f"  Output: '{output_name}' shape (1, {action_dim})")

    # -- Optionally load SB3 model for comparison ------------------------------
    sb3_model = None
    sb3_vec_norm = None
    if args.compare:
        print(f"\nLoading SB3 model for comparison: {args.compare}")
        sb3_model, sb3_vec_norm = load_sb3_model(
            args.compare,
            args.vecnorm,
            use_go2_sysid=args.use_go2_sysid,
        )
        print("  SB3 model loaded successfully.")

    # -- Create gymnasium environment ------------------------------------------
    print(f"\nCreating quadruped environment (robot={args.robot})...")
    if args.domain_rand_config_type is not None:
        resolved_dr_config_type = args.domain_rand_config_type
    elif args.domain_rand:
        resolved_dr_config_type = "default_no_push"
        print(
            "Warning: --domain_rand is deprecated; prefer "
            "--domain_rand_config_type=default_no_push"
        )
    else:
        resolved_dr_config_type = "disabled"

    domain_rand_cfg = DomainRandomizationConfig.from_preset(resolved_dr_config_type)
    env_wrapped = gym.make(
        "QuadrupedVelocityTracking-v0",
        robot=args.robot,
        render_mode=None,
        max_episode_steps=5000,
        domain_rand_cfg=domain_rand_cfg,
        use_go2_sysid=args.use_go2_sysid,
    )
    env_base = env_wrapped.unwrapped
    if args.robot.lower() == "go2" and args.use_go2_sysid:
        assert_go2_sysid_joint_dynamics(env_base.mjModel)
        print("Verified Go2 sysID joint dynamics in the ONNX simulation environment.")
    vec_env = DummyVecEnv([lambda: env_wrapped])
    print(
        "Domain randomization during ONNX replay: "
        f"{'ENABLED' if env_base.domain_rand_cfg.enable else 'DISABLED'}"
    )
    print(f"Domain randomization preset: {resolved_dr_config_type}")
    print(
        "Go2 sysID joint dynamics during ONNX replay: "
        f"{'ENABLED' if args.use_go2_sysid else 'DISABLED'}"
    )
    # NOTE: No VecNormalize here -- the ONNX has normalization baked in,
    # so we pass raw observations directly to the ONNX session.

    # -- Launch MuJoCo viewer --------------------------------------------------
    commander = VelocityCommander(step=args.step)
    viewer = mj_viewer.launch_passive(
        env_base.mjModel,
        env_base.mjData,
        key_callback=commander.on_key,
        show_left_ui=False,
        show_right_ui=False,
    )
    mujoco.mjv_defaultFreeCamera(env_base.mjModel, viewer.cam)
    control_dt = env_base.control_dt  # 0.02 s -> 50 Hz

    print("\n" + "=" * 60)
    print("ONNX POLICY TEST  (press keys in the viewer window)")
    print("=" * 60)
    print(f"  Policy file   : {args.onnx.name}")
    print(f"  Obs dim       : {obs_dim}  (normalization baked in)")
    print(f"  Action dim    : {action_dim}")
    print(f"  Control freq  : {1/control_dt:.0f} Hz")
    if sb3_model:
        print("  Comparison    : SB3 actions printed every 50 steps")
    print()
    print("  Up / Down    :  forward / backward  (lin_vel_x)")
    print("  Left / Right :  strafe left / right (lin_vel_y)")
    print("  [ / ]        :  turn left / right   (ang_vel_z)")
    print("  R            :  reset all velocities")
    print("  Escape       :  quit")
    print("=" * 60 + "\n")

    obs = vec_env.reset()
    env_base.set_commands(vx=0.0, vy=0.0, wz=0.0)
    obs = refresh_current_raw_obs(vec_env)

    try:
        while not commander.is_stopped() and viewer.is_running():
            step_start = time.perf_counter()

            # Update velocity commands
            vx, vy, wz = commander.get()
            env_base.set_commands(vx=vx, vy=vy, wz=wz)
            obs = refresh_current_raw_obs(vec_env)

            # -- ONNX inference ------------------------------------------------
            # obs["policy"] comes from DummyVecEnv with shape (1, 45).
            # We pass it directly -- the ONNX model handles normalisation.
            raw_obs = {key: value.astype(np.float32, copy=True) for key, value in obs.items()}
            ort_actions = sess.run([output_name], {input_name: raw_obs["policy"]})[0]  # (1, 12)

            # -- Optional comparison with SB3 ----------------------------------
            # Compare the two policies on the exact same raw observation before
            # stepping the environment. SB3 expects VecNormalize to have already
            # been applied, while the ONNX graph has that normalisation baked in.
            sb3_actions = None
            if sb3_model:
                sb3_actions = predict_sb3_action_from_raw_obs(sb3_model, sb3_vec_norm, raw_obs)

            # Step environment with ONNX actions
            obs, reward, done, info = vec_env.step(ort_actions)

            step_count = env_base._step_count
            if sb3_actions is not None and step_count % 50 == 0:
                max_diff = float(np.abs(ort_actions - sb3_actions).max())
                print(f"  [step {step_count:5d}] ONNX vs SB3 max action diff: {max_diff:.6f}")

            # -- Camera and viewer sync ----------------------------------------
            viewer.cam.lookat[:] = env_base.mjData.qpos[0:3]
            viewer.sync()

            # -- Periodic status print -----------------------------------------
            if step_count % 50 == 0:
                base_vel = env_base._base_lin_vel_body()
                print(
                    f"  cmd: vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}  |  "
                    f"actual: vx={base_vel[0]:+.2f}  vy={base_vel[1]:+.2f}  |  "
                    f"step={step_count}"
                )

            # -- Episode reset -------------------------------------------------
            if done[0]:
                print("  [Episode reset - robot terminated]")
                obs = vec_env.reset()
                env_base.set_commands(vx=vx, vy=vy, wz=wz)
                obs = refresh_current_raw_obs(vec_env)

            # -- Real-time sync ------------------------------------------------
            elapsed = time.perf_counter() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        viewer.close()
        vec_env.close()
        print("Done.")


if __name__ == "__main__":
    main()
