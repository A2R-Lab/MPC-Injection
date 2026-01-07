#!/usr/bin/env python3
"""
Script to load a trained SAC-MPC model and record the trajectories of each body/limb of the walker to later plot.

The recorded trajectories will be saved in a subdirectory to here and a separate plotting script will visualize them.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.sac_mpc.sac_mpc import SAC_MPC
from mpc_rl.td3_mpc.td3_mpc import TD3_MPC

def load_config(run_dir: Path):
    """Load the configuration from config.json"""
    config_path = run_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def make_dm_env(domain: str, task: str, render_mode='rgb_array'):
    """Create a dm_control environment wrapped for gymnasium."""
    dm_env = suite.load(domain_name=domain, task_name=task)
    gym_env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    gym_env = FlattenObservation(gym_env)
    return gym_env

def load_model_and_vecnormalize(run_dir: Path, config: dict, checkpoint_step: int = None):
    """
    Load the trained model and VecNormalize wrapper.
    
    Args:
        run_dir: Path to the run directory
        config: Configuration dictionary
        checkpoint_step: Which checkpoint to load (e.g., 500000). If None, loads final_model.zip
    
    Returns:
        Tuple of (model, vec_env)
    """
    # Create the environment
    domain = config['domain']
    task = config['task']

    def env_fn():
        return make_dm_env(domain, task, render_mode=None)
    
    vec_env = DummyVecEnv([env_fn])

    # Load VecNormalize stats
    if checkpoint_step is not None:
        vecnormalize_path = run_dir / "checkpoints" / f"vecnormalize_{checkpoint_step}.pkl"
        model_path = run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip"
    else:
        vecnormalize_path = run_dir / "vec_normalize.pkl"
        model_path = run_dir / "final_model.zip"

    print(f"Loading VecNormalize from: {vecnormalize_path}")
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)

    # Set VecNormalize to not update stats during evaluation
    vec_env.training = False
    vec_env.norm_reward = False

    # Load the model
    print(f"Loading model from: {model_path}")
    model = SAC_MPC.load(model_path, env=vec_env)

    return model, vec_env

def run_episode_and_record_trajectories(model, vec_env, domain: str, max_steps: int = 1000):
    """
    Run one episode and collect trajectories of each body part for data analysis.

    Args:
        model: Trained model
        vec_env: Vectorized environment with normalization
        domain: Environment domain (e.g. 'walker' in our case)
        max_steps: Maximum number of steps per episode
    
    Returns:
        Dictionary mapping body part names to their trajectories (list of states over time)
    """
    obs = vec_env.reset()


def main():
    """Main function to load model and record body part trajectories."""
    