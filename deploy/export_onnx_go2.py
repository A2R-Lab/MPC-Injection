"""Export a trained MPC-RL SAC or TD3 policy (Go2) to ONNX for real-robot deployment.

This script converts a Stable-Baselines3 checkpoint into a self-contained
ONNX file that the C++ deployment binary can load via ONNXRuntime.
It supports both SAC and TD3 -- the algorithm is auto-detected from the
directory name (looks for "-SAC-" or "-TD3-" in the path) or can be
specified explicitly with --algo.

Background
----------
During training the policy network never sees raw sensor values directly.
Instead, SB3's VecNormalize wrapper maintains a running mean and variance of
every observation and normalises each sensor reading to approximately zero-mean
and unit-variance before handing it to the neural network.  This normalisation
lives *outside* the PyTorch model - it is a Python wrapper, not a layer.

On the real robot we have no Python, only C++.  We therefore "bake" the
VecNormalize statistics (mean and std) into the ONNX graph as frozen constant
buffers, so the C++ runtime can pass raw sensor values and the normalisation
happens transparently inside the model.

Network architecture differences between SAC and TD3
----------------------------------------------------
SAC and TD3 use different actor structures in SB3:

  SAC actor:
      actor.latent_pi  - MLP trunk: Linear(45->256)->ReLU->Linear(256->256)->ReLU
      actor.mu         - output head: Linear(256->12)  [no tanh]
      Inference: normalise(obs) -> latent_pi -> mu -> tanh()  -> actions

  TD3 actor:
      actor.mu         - full network: Linear(45->400)->ReLU->Linear(400->300)
                                       ->ReLU->Linear(300->12)->Tanh()
                         (tanh is the last layer, already inside mu)
      Inference: normalise(obs) -> mu  -> actions   [tanh already applied]

Both produce actions in [-1, 1].  The C++ action manager then applies:
    joint_target[i] = actions[i] * 0.5 + default_joint_pos[i]
using the same FL, FR, RL, RR joint order as the MuJoCo training env.

Usage
-----
    # SAC (algorithm auto-detected from directory name):
    python deploy/export_onnx_go2.py \\
        --model_zip  logs/quadruped-velocity_tracking-SAC-20260305-170808/final_model.zip \\
        --vecnorm_pkl logs/quadruped-velocity_tracking-SAC-20260305-170808/vec_normalize.pkl \\
        --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/

    # TD3 (auto-detected):
    python deploy/export_onnx_go2.py \\
        --model_zip  logs/quadruped-velocity_tracking-TD3-20260301-120000/final_model.zip \\
        --vecnorm_pkl logs/quadruped-velocity_tracking-TD3-20260301-120000/vec_normalize.pkl \\
        --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/

    # Explicit algo override:
    python deploy/export_onnx_go2.py \\
        --model_zip  .../final_model.zip \\
        --vecnorm_pkl .../vec_normalize.pkl \\
        --output_dir .../exported/ \\
        --algo TD3

    # Specific checkpoint step:
    python deploy/export_onnx_go2.py \\
        --model_zip  logs/.../checkpoints/model_4988928_steps.zip \\
        --vecnorm_pkl logs/.../checkpoints/model_vecnormalize_4988928_steps.pkl \\
        --output_dir deploy/robots/go2/config/policy/velocity/v0/exported/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make the project importable when running from the MPC-RL root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# ONNX exporter wrappers
# =============================================================================

class _BaseActorOnnxExporter(nn.Module):
    """Base class that bakes VecNormalize stats into the ONNX graph."""

    def __init__(self, obs_mean: np.ndarray, obs_std: np.ndarray):
        super().__init__()
        # Frozen constant buffers -- embedded directly in the ONNX file.
        self.register_buffer("obs_mean", torch.tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std",  torch.tensor(obs_std,  dtype=torch.float32))

    def _normalise(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply VecNormalize scaling: clip((obs - mean) / std, -10, +10)."""
        x = (obs - self.obs_mean) / self.obs_std
        return torch.clamp(x, -10.0, 10.0)


class SACActorOnnxExporter(_BaseActorOnnxExporter):
    """Exports a SB3 SAC actor to ONNX.

    SAC actor structure:
        actor.latent_pi  -  MLP trunk: Linear(45->256)->ReLU->Linear(256->256)->ReLU
        actor.mu         -  output head: Linear(256->12)  [no tanh]

    Inference pipeline baked into ONNX:
        raw_obs  ->  normalise  ->  latent_pi  ->  mu  ->  tanh()  ->  actions

    Args:
        latent_pi:  model.policy.actor.latent_pi  -- the MLP trunk.
        mu:         model.policy.actor.mu          -- the linear output head.
        obs_mean:   VecNormalize running mean for "policy" obs, shape (45,).
        obs_std:    VecNormalize running std  for "policy" obs, shape (45,).
    """

    def __init__(
        self,
        latent_pi: nn.Module,
        mu: nn.Module,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
    ):
        super().__init__(obs_mean, obs_std)
        self.latent_pi = latent_pi
        self.mu = mu

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._normalise(obs)
        latent = self.latent_pi(x)
        mean_actions = self.mu(latent)
        return torch.tanh(mean_actions)  # SAC: tanh applied externally


class TD3ActorOnnxExporter(_BaseActorOnnxExporter):
    """Exports a SB3 TD3 actor to ONNX.

    TD3 actor structure (different from SAC!):
        actor.mu  -  full network: Linear(45->400)->ReLU->Linear(400->300)
                                   ->ReLU->Linear(300->12)->Tanh()
        There is NO latent_pi attribute; tanh is the last layer inside mu itself.

    Inference pipeline baked into ONNX:
        raw_obs  ->  normalise  ->  mu (includes tanh)  ->  actions

    Args:
        mu:         model.policy.actor.mu  -- the full network including tanh.
        obs_mean:   VecNormalize running mean for "policy" obs, shape (45,).
        obs_std:    VecNormalize running std  for "policy" obs, shape (45,).
    """

    def __init__(
        self,
        mu: nn.Module,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
    ):
        super().__init__(obs_mean, obs_std)
        self.mu = mu

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._normalise(obs)
        return self.mu(x)  # TD3: tanh already the last layer inside mu


# =============================================================================
# Algorithm detection
# =============================================================================

SUPPORTED_ALGOS = ("SAC", "TD3")


def detect_algorithm(model_zip: Path) -> str:
    """Detect SAC or TD3 from the model zip path.

    Checks each parent directory name for '-SAC-' or '-TD3-',
    then falls back to config.json in the same or parent directory.
    Raises ValueError if it cannot determine the algorithm.
    """
    # Walk up the path components looking for the run directory name
    for part in reversed(model_zip.parts):
        for algo in SUPPORTED_ALGOS:
            if f"-{algo}-" in part or part.endswith(f"-{algo}"):
                return algo

    # Fallback: look for config.json alongside the zip or one level up
    for candidate_dir in (model_zip.parent, model_zip.parent.parent):
        config_path = candidate_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            algo = cfg.get("algorithm", "").upper().replace("-MPC", "")
            if algo in SUPPORTED_ALGOS:
                return algo

    raise ValueError(
        f"Could not detect algorithm from path '{model_zip}'.\n"
        f"Expected '-SAC-' or '-TD3-' in a parent directory name, "
        f"or a config.json with an 'algorithm' key.\n"
        f"Use --algo SAC or --algo TD3 to specify explicitly."
    )


# =============================================================================
# Helpers
# =============================================================================

def load_vecnormalize_stats(
    vec_normalize_path: Path, policy_obs_dim: int = 45
) -> tuple[np.ndarray, np.ndarray]:
    """Load VecNormalize and extract the 'policy' obs running statistics.

    SB3's VecNormalize stores a separate RunningMeanStd for each Dict key.
    We only need the "policy" key (the actor input); the "privileged" key
    is for the critic only and is not used at deployment.

    Returns:
        obs_mean: shape (policy_obs_dim,), float64
        obs_std:  shape (policy_obs_dim,), float64
    """
    import pickle
    with open(vec_normalize_path, "rb") as f:
        vec_norm = pickle.load(f)

    # SB3 VecNormalize with a Dict obs space stores obs_rms as a dict
    if hasattr(vec_norm, "obs_rms") and isinstance(vec_norm.obs_rms, dict):
        rms = vec_norm.obs_rms["policy"]
    else:
        # If for some reason it's a flat VecNormalize (shouldn't happen here)
        raise ValueError(
            "Expected vec_norm.obs_rms to be a dict with 'policy' key. "
            f"Got: {type(getattr(vec_norm, 'obs_rms', None))}"
        )

    obs_mean = rms.mean.astype(np.float32)
    obs_std = np.sqrt(rms.var.astype(np.float32) + 1e-8)  # same epsilon as SB3

    assert obs_mean.shape == (policy_obs_dim,), (
        f"Expected obs_mean shape ({policy_obs_dim},), got {obs_mean.shape}"
    )
    return obs_mean, obs_std


def normalise_policy_obs(
    raw_policy_obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> np.ndarray:
    """Apply the same VecNormalize transform baked into the exported ONNX graph."""
    x = (raw_policy_obs - obs_mean) / obs_std
    return np.clip(x, -10.0, 10.0).astype(np.float32)


def export(
    model_zip: Path,
    vecnorm_pkl: Path,
    output_dir: Path,
    algo: str | None = None,
    opset_version: int = 18,
    policy_obs_dim: int = 45,
    action_dim: int = 12,
):
    """Full export pipeline.

    Loads the SB3 model and VecNormalize, wraps them in the appropriate
    exporter class for SAC or TD3, exports to ONNX, and validates with
    ONNXRuntime.
    """
    import onnx

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "policy.onnx"

    # -- 0. Detect / validate algorithm ---------------------------------------
    if algo is None:
        algo = detect_algorithm(model_zip)
    algo = algo.upper()
    assert algo in SUPPORTED_ALGOS, f"Unsupported algorithm: {algo}. Choose from {SUPPORTED_ALGOS}"
    print(f"Algorithm: {algo}")

    # -- 1. Load VecNormalize statistics --------------------------------------
    print(f"Loading VecNormalize stats from:  {vecnorm_pkl}")
    obs_mean, obs_std = load_vecnormalize_stats(vecnorm_pkl, policy_obs_dim)
    print(f"  obs_mean[:6] = {obs_mean[:6]}")
    print(f"  obs_std[:6]  = {obs_std[:6]}")

    # -- 2. Load SB3 model ----------------------------------------------------
    print(f"\nLoading SB3 {algo} model from:    {model_zip}")
    from mpc_rl.asym_policies import AsymmetricSACPolicy, AsymmetricTD3Policy
    from stable_baselines3 import SAC, TD3

    algo_cls = SAC if algo == "SAC" else TD3
    # Force CPU to match the exported ONNX deployment path and avoid
    # mixed-device checks when a workstation has CUDA available.
    model = algo_cls.load(str(model_zip), device="cpu")
    model.policy = model.policy.cpu()
    actor = model.policy.actor

    # -- 3. Build algorithm-specific exporter ----------------------------------
    if algo == "SAC":
        # SAC: latent_pi (trunk) and mu (output head) are separate; tanh is external
        print(f"  Actor architecture (SAC):")
        print(f"    latent_pi : {actor.latent_pi}")
        print(f"    mu        : {actor.mu}")
        exporter = SACActorOnnxExporter(
            latent_pi=actor.latent_pi,
            mu=actor.mu,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
    else:  # TD3
        # TD3: no latent_pi; mu IS the full network and already ends with Tanh()
        print(f"  Actor architecture (TD3):")
        print(f"    mu (full net): {actor.mu}")
        exporter = TD3ActorOnnxExporter(
            mu=actor.mu,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )

    # Move everything to CPU -- the C++ ONNXRuntime deployment runs on CPU,
    # and torch.onnx.export requires all tensors on the same device.
    exporter = exporter.cpu()
    exporter.eval()

    # -- 4. Export to ONNX -----------------------------------------------------
    dummy_obs = torch.zeros(1, policy_obs_dim, dtype=torch.float32)  # CPU

    print(f"\nExporting ONNX model to:          {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            exporter,
            dummy_obs,
            str(output_path),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=opset_version,
            # No dynamic axes: the C++ runtime always runs with batch_size=1
        )

    # -- 5. Validate with onnx library -----------------------------------------
    print("\nValidating ONNX graph...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  [ok] ONNX graph is valid")

    # -- 6. Sanity-check with ONNXRuntime --------------------------------------
    print("\nRunning ONNXRuntime sanity checks...")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    # Test 1: zeros input (robot at default pose, no commands)
    zero_obs = np.zeros((1, policy_obs_dim), dtype=np.float32)
    out_zero = sess.run(["actions"], {"obs": zero_obs})[0]
    assert out_zero.shape == (1, action_dim), f"Bad output shape: {out_zero.shape}"
    assert np.all(np.abs(out_zero) <= 1.0), "tanh output out of [-1, 1]!"
    print(f"  [ok] Zero obs  -> actions shape {out_zero.shape}, "
          f"range [{out_zero.min():.4f}, {out_zero.max():.4f}]")

    # Test 2: compare PyTorch and ONNX outputs on random raw input
    rng = np.random.default_rng(42)
    rand_obs = rng.standard_normal((1, policy_obs_dim)).astype(np.float32)
    with torch.no_grad():
        pt_out = exporter(torch.from_numpy(rand_obs)).numpy()
    ort_out = sess.run(["actions"], {"obs": rand_obs})[0]
    max_diff = np.abs(pt_out - ort_out).max()
    assert max_diff < 1e-5, f"PyTorch/ONNX output mismatch: max diff = {max_diff}"
    print(f"  [ok] Random obs -> PyTorch vs ONNX max diff = {max_diff:.2e}")

    # Test 3: compare ONNX against the original SB3 policy on the same raw obs.
    raw_policy_obs = rng.standard_normal((1, policy_obs_dim)).astype(np.float32)
    norm_policy_obs = normalise_policy_obs(raw_policy_obs, obs_mean, obs_std)
    raw_obs_dict = {}
    if hasattr(model, "observation_space") and hasattr(model.observation_space, "spaces"):
        for key, space in model.observation_space.spaces.items():
            if key == "policy":
                raw_obs_dict[key] = norm_policy_obs
            else:
                raw_obs_dict[key] = np.zeros((1, *space.shape), dtype=np.float32)
    else:
        raw_obs_dict = norm_policy_obs

    sb3_out, _ = model.predict(raw_obs_dict, deterministic=True)
    onnx_out = sess.run(["actions"], {"obs": raw_policy_obs})[0]
    sb3_diff = np.abs(sb3_out - onnx_out).max()
    assert sb3_diff < 1e-5, f"SB3/ONNX output mismatch on raw obs: max diff = {sb3_diff}"
    print(f"  [ok] Random raw obs -> SB3 vs ONNX max diff = {sb3_diff:.2e}")

    # Report what joint targets look like from default pose
    print(f"\nAt default standing pose (zero obs):")
    action_scale = 0.5
    default_joint_pos = np.array([
         0.0, 0.9, -1.8,  # FL: hip, thigh, calf
         0.0, 0.9, -1.8,  # FR
         0.0, 0.9, -1.8,  # RL
         0.0, 0.9, -1.8,  # RR
    ])
    joint_targets = out_zero[0] * action_scale + default_joint_pos
    joint_names = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]
    for name, target, default in zip(joint_names, joint_targets, default_joint_pos):
        print(f"  {name:12s}: target={target:+.4f}  (default={default:+.4f})")

    print(f"\n[ok] Export complete: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export MPC-RL SAC or TD3 Go2 policy to ONNX for real-robot deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_zip", required=True, type=Path,
        help="Path to the SB3 model .zip file (e.g. final_model.zip or a checkpoint)",
    )
    parser.add_argument(
        "--vecnorm_pkl", required=True, type=Path,
        help="Path to the matching VecNormalize .pkl file",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Directory to write policy.onnx into (will be created if needed)",
    )
    parser.add_argument(
        "--algo", type=str, default=None, choices=["SAC", "TD3"],
        help="RL algorithm (default: auto-detected from directory name)",
    )
    parser.add_argument(
        "--opset", type=int, default=18,
        help="ONNX opset version (default: 18)",
    )
    args = parser.parse_args()

    if not args.model_zip.exists():
        parser.error(f"model_zip not found: {args.model_zip}")
    if not args.vecnorm_pkl.exists():
        parser.error(f"vecnorm_pkl not found: {args.vecnorm_pkl}")

    export(
        model_zip=args.model_zip,
        vecnorm_pkl=args.vecnorm_pkl,
        output_dir=args.output_dir,
        algo=args.algo,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
