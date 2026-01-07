#!/usr/bin/env python3
"""
Script to load a trained SAC-MPC model and visualize its performance on the walker environment
with an animated matplotlib plot.
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
        return make_dm_env(domain, task, render_mode='rgb_array')
    
    vec_env = DummyVecEnv([env_fn])
    
    # Load VecNormalize stats
    if checkpoint_step is not None:
        vecnormalize_path = run_dir / "checkpoints" / f"model_vecnormalize_{checkpoint_step}_steps.pkl"
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


def run_episode_and_collect_frames(model, vec_env, domain: str, max_steps: int = 1000):
    """
    Run one episode and collect frames for visualization.
    
    Args:
        model: Trained model
        vec_env: Vectorized environment with normalization
        domain: Environment domain (e.g., 'walker')
        max_steps: Maximum number of steps per episode
    
    Returns:
        Tuple of (frames, total_reward, episode_length)
    """
    obs = vec_env.reset()
    frames = []
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Get action from the model (deterministic for evaluation)
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        step += 1
        
        # Get frame from the underlying environment using tracking camera for walker
        if domain == "walker":
            frame = vec_env.envs[0].unwrapped._env.physics.render(camera_id='side', height=480, width=640)
        else:
            env = vec_env.envs[0]
            frame = env.render()
        frames.append(frame)
        
        # Check if episode is done
        if done[0]:
            break
    
    print(f"Episode finished: {step} steps, total reward: {total_reward:.2f}")
    return frames, total_reward, step


def create_animated_plot(frames, total_reward, episode_length, save_path: Path = None):
    """
    Create an animated matplotlib plot from the collected frames.
    
    Args:
        frames: List of RGB arrays (frames from the environment)
        total_reward: Total reward for the episode
        episode_length: Number of steps in the episode
        save_path: Optional path to save the animation as GIF
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Display the first frame
    im = ax.imshow(frames[0])
    
    # Add text for step counter and reward
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        """Initialize animation."""
        im.set_data(frames[0])
        info_text.set_text(f'Step: 0/{episode_length}\nTotal Reward: {total_reward:.2f}')
        return [im, info_text]
    
    def animate(frame_idx):
        """Update animation for each frame."""
        im.set_data(frames[frame_idx])
        info_text.set_text(f'Step: {frame_idx}/{episode_length}\nTotal Reward: {total_reward:.2f}')
        return [im, info_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(frames), interval=20,  # 20ms between frames (50 FPS)
        blit=True, repeat=True
    )
    
    # Save as GIF if path is provided
    if save_path:
        print(f"Saving animation as GIF: {save_path}")
        anim.save(save_path, writer='pillow', fps=50)
        print(f"GIF saved successfully!")
    
    plt.title(f'Walker Performance - Total Reward: {total_reward:.2f}')
    plt.tight_layout()
    return fig, anim


def main():
    """Main function to load model and create visualization."""
    # Paths
    #run_dir = Path("/home/roy/MPC-RL/logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-113507-percentage-50pct")
    run_dir = Path("/home/roy/MPC-RL/logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-112012-percentage-0pct")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(run_dir)
    print(f"Environment: {config['domain']}-{config['task']}")
    print(f"Algorithm: {config['algorithm']}")
    
    # Load model and VecNormalize
    # You can specify a checkpoint step (e.g., 500000) or use None for final model
    checkpoint_step = 200_000  # Change this to load different checkpoints, or set to None for final
    print(f"\nLoading model (checkpoint: {checkpoint_step if checkpoint_step else 'final'})...")
    model, vec_env = load_model_and_vecnormalize(run_dir, config, checkpoint_step)
    
    # Run episode and collect frames
    print("\nRunning episode and collecting frames...")
    frames, total_reward, episode_length = run_episode_and_collect_frames(
        model, vec_env, config['domain'], max_steps=1000
    )
    
    print(f"\nCollected {len(frames)} frames")
    
    # Create animation and save as GIF
    print("\nCreating animation...")
    gif_name = f"model_{checkpoint_step}_steps.gif" if checkpoint_step else "final_model.gif"
    gif_path = run_dir / gif_name
    fig, anim = create_animated_plot(frames, total_reward, episode_length, save_path=gif_path)
    
    # Show the animation
    print("\nDisplaying animation (close window to exit)...")
    plt.show()
    
    # Clean up
    vec_env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
