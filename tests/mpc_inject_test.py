"""
TODO: Test script to try out MPC injection callback with custom SAC training.

Mainly here to experiment with before integration into the main training script.
"""

import datetime
import json
import os
from pathlib import Path
import warnings
import subprocess

# Configure JAX for GPU with compatible architecture settings
# Try to detect GPU first
gpu_available = False
try:
    # Get GPU compute capability
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    )
    compute_cap = result.stdout.strip().split('\n')[0].replace('.', '')
    print(f"Detected GPU compute capability: {compute_cap}")
    
    # Configure for GPU
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir=/usr/lib/cuda"
    gpu_available = True
except Exception as e:
    print(f"Could not detect GPU compute capability: {e}")
    print("Falling back to CPU")
    # Configure for CPU
    os.environ["JAX_PLATFORMS"] = "cpu"

# Suppress JAX warnings and info logs
warnings.filterwarnings("ignore", category=UserWarning, module="jax")
warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

from absl import app
from absl import flags
from absl import logging

import gymnasium as gym
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from sbx import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import mediapy as media
import sys

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.sac_mpc.mpc_inject_callbacks import EpisodeMPCInjectCallback, AdaptiveMPCInjectCallback
from mpc_rl.sac_mpc.sac_mpc import SAC_MPC

# Import JAX and verify backend
import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Inform user about the backend being used
if jax.default_backend() == 'gpu':
    print("JAX is using GPU acceleration")
elif jax.default_backend() == 'cpu':
    print("JAX is using CPU (GPU not available or not detected)")
else:
    print(f"JAX is using backend: {jax.default_backend()}")

# Set logging level to suppress JAX backend initialization messages
logging.set_verbosity(logging.WARNING)


def make_dmc_env(domain: str, task: str, render_mode=None):
        dm_env = suite.load(domain_name=domain, task_name=task)
        gym_env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
        gym_env = FlattenObservation(gym_env)
        return gym_env


if __name__ == "__main__":
    print("Testing if SAC-MPC working...")

    # Setup Environment
    domain = "cartpole"
    task = "swingup"

    vec_env = make_vec_env(
        lambda: make_dmc_env(domain, task), # lambda fxn so make_vec_env() can make multiple envs
        n_envs=4,
        seed=1,
    )

    # VecNormalize standardizes observations and rewards to N(0,1), which is critical for stable
    # training (prevents different scale features from dominating)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Setup Model
    model = SAC_MPC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        verbose=1,  # Enable training logs (0=no output, 1=info, 2=debug)
        seed=1,
    )

    # Create eval environment for evaluation callback
    # Must be wrapped the same way as training env (with VecNormalize)
    eval_env = make_vec_env(
        lambda: make_dmc_env(domain, task),
        n_envs=1,
        seed= 1 + 1000,
    )
    eval_env = VecNormalize(
        eval_env,
        training=False,  # Don't update stats during evaluation
        norm_obs=True,
        norm_reward=True,
    )

    # Create callback for evaluating the trained model
    # This will evaluate the model periodically during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,  # Don't save best model for this test
        log_path=None,  # Don't save eval logs for this test
        eval_freq=10_000,  # Evaluate every 10k steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=0,
    )

    print("\nStarting training...")
    print(f"Training environment: {vec_env.num_envs} parallel environments")
    print(f"Total timesteps: 1,000,000")
    print(f"Evaluation frequency: every 10,000 steps")
    
    # Train the model
    model.learn(
        total_timesteps=500_000,
        callback=eval_callback,
        log_interval=4,  # Log training metrics every 4 episodes
        progress_bar=True,
    )
    
    print("\nTraining complete!")
    
    # Final evaluation with the trained model
    print("\nRunning final evaluation...")
    
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    print(f"\nFinal Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Cleanup
    vec_env.close()
    eval_env.close()
    
    print("\nTest complete!")