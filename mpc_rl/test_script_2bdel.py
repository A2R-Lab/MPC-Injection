"""
Just a scrap paper script for testing

TODO: Delete this file when done!
"""


import datetime
import json
import os
from pathlib import Path
import warnings
import subprocess

# Configure JAX for GPU with compatible architecture settings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORMS"] = "cuda"

# Try to detect GPU compute capability and set appropriate flags
try:
    # Get GPU compute capability
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    )
    compute_cap = result.stdout.strip().split('\n')[0].replace('.', '')
    print(f"Detected GPU compute capability: {compute_cap}")
    
    # Set XLA flags to use detected compute capability
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir=/usr/lib/cuda"
except Exception as e:
    print(f"Could not detect GPU compute capability: {e}")
    # Use default settings
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

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
import numpy as np
import mediapy as media

# Import JAX and verify GPU backend
import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Verify we're using GPU
if jax.default_backend() != 'gpu':
    raise RuntimeError(
        f"JAX is not using GPU! Backend: {jax.default_backend()}. "
        "Please check your CUDA installation and JAX GPU setup."
    )

# Set logging level to suppress JAX backend initialization messages
logging.set_verbosity(logging.WARNING)

# Relative import - import the SAC_MPC class from the sac_mpc module
from sac_mpc.sac_mpc import SAC_MPC


def make_dmc_env(domain: str, task: str, render_mode=None):
        dm_env = suite.load(domain_name=domain, task_name=task)
        gym_env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
        gym_env = FlattenObservation(gym_env)
        return gym_env


if __name__ == "__main__":
    print("Testing if SAC-MPC working...")

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

    model = SAC_MPC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        seed=1,
    )
