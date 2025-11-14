"""
Proof of concept script for training the Shadow Dexterous Hand environment
from gymnasium-robotics using SAC with goal-conditioned learning.

This script demonstrates:
1. Multi-goal API with dictionary observations (observation, achieved_goal, desired_goal)
2. Goal-conditioned reward computation
3. Training with SBX SAC on a goal-based manipulation task
4. Proper handling of the HandManipulateBlock environment

NOTE ON DIFFICULTY:
These environments are VERY challenging and typically require:
- Hindsight Experience Replay (HER) for sparse rewards (not implemented here)
- 2-5M timesteps for meaningful learning
- Multiple parallel environments (8-16) for sample efficiency
- Dense rewards are much easier to learn than sparse
- HandManipulateBlockRotateZ is the easiest variant (only z-axis rotation)

HYPERPARAMETER CHOICES:
- Learning rate 1e-3: Higher than default for faster initial learning
- Gamma 0.98: Slightly lower for better short-term credit assignment
- Network [256,256,256]: Larger network for complex 75-dim observation space
- 8 parallel envs: Improves sample efficiency and exploration
- Train after each episode: Better for short episodes (50 steps default)
- Gradient steps = -1: Do as many updates as env steps for faster learning
"""

import os
import sys
import warnings
from pathlib import Path
import subprocess

# Configure JAX for GPU with compatible architecture settings
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
    print("JAX configured for GPU acceleration")
except Exception as e:
    print(f"Could not detect GPU, falling back to CPU: {e}")
    os.environ["JAX_PLATFORMS"] = "cpu"
    print("JAX configured for CPU (GPU not available)")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from sbx import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import mediapy as media
import jax

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)

# Verify JAX backend
print(f"\n{'='*60}")
print(f"JAX Backend Information:")
print(f"  Backend: {jax.default_backend()}")
print(f"  Devices: {jax.devices()}")
if jax.default_backend() == 'gpu':
    print(f"  Using GPU acceleration")
elif jax.default_backend() == 'cpu':
    print(f"  Using CPU (GPU not available or not detected)")
print(f"{'='*60}\n")


class ProgressCallback(BaseCallback):
    """
    Custom callback to monitor training progress and log statistics.
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics when available
        if len(self.model.ep_info_buffer) > 0 and self.n_calls % self.check_freq == 0:
            recent_ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            recent_ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
            
            if len(recent_ep_rewards) > 0:
                mean_reward = np.mean(recent_ep_rewards)
                mean_length = np.mean(recent_ep_lengths)
                
                if self.verbose > 0:
                    print(f"\n[Step {self.n_calls:,}] Recent performance:")
                    print(f"  Mean reward: {mean_reward:.2f}")
                    print(f"  Mean episode length: {mean_length:.1f}")
                    
                # Log to tensorboard if available
                self.logger.record("rollout/ep_rew_mean_recent", mean_reward)
                self.logger.record("rollout/ep_len_mean_recent", mean_length)
        
        return True


class GoalConditionedWrapper(gym.Wrapper):
    """
    Wrapper to flatten the goal-conditioned observation space for standard RL algorithms.
    
    The Shadow Hand environments use a dictionary observation space with:
    - observation: actual robot/object state
    - achieved_goal: current goal achievement (e.g., block pose)
    - desired_goal: target goal to achieve
    
    This wrapper concatenates all three into a single flat vector that standard
    algorithms like SAC can process.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get the original observation spaces
        obs_space = env.observation_space.spaces['observation']
        achieved_goal_space = env.observation_space.spaces['achieved_goal']
        desired_goal_space = env.observation_space.spaces['desired_goal']
        
        # Calculate total dimension
        total_dim = (
            obs_space.shape[0] + 
            achieved_goal_space.shape[0] + 
            desired_goal_space.shape[0]
        )
        
        # Create flattened observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
        
        print(f"Original observation dims: {obs_space.shape[0]}")
        print(f"Achieved goal dims: {achieved_goal_space.shape[0]}")
        print(f"Desired goal dims: {desired_goal_space.shape[0]}")
        print(f"Flattened observation dims: {total_dim}")
    
    def _flatten_obs(self, obs_dict):
        """Concatenate observation, achieved_goal, and desired_goal."""
        return np.concatenate([
            obs_dict['observation'],
            obs_dict['achieved_goal'],
            obs_dict['desired_goal']
        ])
    
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs_dict), info
    
    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs_dict), reward, terminated, truncated, info


def make_shadow_hand_env(env_id='HandManipulateBlockRotateZ-v1', reward_type='dense', seed=None):
    """
    Create a Shadow Dexterous Hand environment.
    
    Args:
        env_id: Environment ID. Options include:
            - HandManipulateBlockRotateZ-v1: Rotate block around z-axis
            - HandManipulateBlockRotateParallel-v1: Rotate around z and axis-aligned x/y
            - HandManipulateBlockRotateXYZ-v1: Full rotation control
            - HandManipulateBlockFull-v1: Full rotation + position control
        reward_type: 'sparse' or 'dense' reward
        seed: Random seed
    
    Returns:
        Wrapped environment
    """
    # Modify env_id for dense reward if needed
    if reward_type == 'dense' and 'Dense' not in env_id:
        env_id = env_id.replace('-v1', 'Dense-v1')
    
    print(f"\nCreating environment: {env_id}")
    env = gym.make(env_id)
    
    if seed is not None:
        env.reset(seed=seed)
    
    # Wrap to flatten the goal-conditioned observation
    env = GoalConditionedWrapper(env)
    
    return env


def evaluate_policy(model, env, n_eval_episodes=5, render=False, save_video=False, video_path=None):
    """
    Evaluate the trained policy.
    
    Args:
        model: Trained model
        env: Unwrapped environment (before GoalConditionedWrapper)
        n_eval_episodes: Number of episodes to evaluate
        render: Whether to render (for video recording)
        save_video: Whether to save videos
        video_path: Path to save videos
    
    Returns:
        Mean reward and success rate
    """
    episode_rewards = []
    episode_successes = []
    
    # Wrap environment for evaluation
    eval_env = GoalConditionedWrapper(env)
    
    for episode in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        frames = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Capture frames if rendering
            if render and save_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        
        episode_rewards.append(episode_reward)
        
        # Check if goal was achieved (info may contain 'is_success')
        success = info.get('is_success', 0.0)
        episode_successes.append(success)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {success}")
        
        # Save video if requested
        if save_video and frames and video_path:
            video_file = Path(video_path) / f"eval_episode_{episode}.mp4"
            video_file.parent.mkdir(parents=True, exist_ok=True)
            media.write_video(str(video_file), frames, fps=25)
            print(f"  Video saved to: {video_file}")
    
    mean_reward = np.mean(episode_rewards)
    mean_success = np.mean(episode_successes)
    
    print("\n" + "="*50)
    print(f"Evaluation Results ({n_eval_episodes} episodes):")
    print(f"  Mean reward: {mean_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Success rate: {mean_success:.2%}")
    print("="*50 + "\n")
    
    return mean_reward, mean_success


def train_shadow_hand(
    env_id='HandManipulateBlockRotateZ-v1',
    reward_type='dense',
    total_timesteps=1_000_000,
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=256,
    n_eval_episodes=10,
    eval_freq=10_000,
    num_envs=4,
    seed=42,
    log_dir='logs/shadow_hand_test',
    save_model=True
):
    """
    Train SAC on Shadow Dexterous Hand environment.
    
    Args:
        env_id: Environment ID
        reward_type: 'sparse' or 'dense'
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for SAC
        buffer_size: Replay buffer size
        batch_size: Minibatch size
        n_eval_episodes: Number of evaluation episodes
        eval_freq: Evaluation frequency
        num_envs: Number of parallel environments
        seed: Random seed
        log_dir: Directory for logs and checkpoints
        save_model: Whether to save the trained model
    """
    print("="*60)
    print("Shadow Dexterous Hand Training - Proof of Concept")
    print("="*60)
    print(f"Environment: {env_id}")
    print(f"Reward type: {reward_type}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {num_envs}")
    print(f"Seed: {seed}")
    print("="*60 + "\n")
    
    # Set random seeds
    np.random.seed(seed)
    
    # Create training environment with multiple parallel environments
    print(f"Creating {num_envs} parallel training environments...")
    train_env = DummyVecEnv([
        lambda: make_shadow_hand_env(env_id, reward_type, seed + i)
        for i in range(num_envs)
    ])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env_base = gym.make(env_id if reward_type == 'sparse' else env_id.replace('-v1', 'Dense-v1'))
    eval_env_base.reset(seed=seed + 1000)
    eval_env = DummyVecEnv([
        lambda: GoalConditionedWrapper(eval_env_base)
    ])
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_path / "best_model"),
        log_path=str(log_path / "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
    )
    
    # Setup progress monitoring callback
    progress_callback = ProgressCallback(check_freq=5000, verbose=1)
    
    # Combine callbacks
    callbacks = [eval_callback, progress_callback]
    
    # Create SAC model
    print("\nInitializing SAC model...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=max(1000, num_envs * 50),  # Start after enough initial samples
        tau=0.005,
        gamma=0.98,  # Slightly lower gamma for better short-term learning
        train_freq=(1, "episode"),  # Train after each episode for better sample efficiency
        gradient_steps=-1,  # Do as many gradient steps as steps done in the env
        verbose=1,
        tensorboard_log=str(log_path / "tensorboard"),
        seed=seed,
        policy_kwargs=dict(net_arch=[256, 256, 256]),  # Larger network for complex task
    )
    
    print(f"\nModel architecture:")
    print(f"  Policy: MlpPolicy with [256, 256, 256] hidden layers")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gamma: 0.98")
    print(f"  Train freq: after each episode")
    print(f"  Gradient steps: -1 (match env steps)")
    
    # Train the model
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("-"*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    print("\nTraining complete!")
    
    # Save the final model
    if save_model:
        model_path = log_path / "final_model"
        model.save(model_path)
        print(f"Model saved to: {model_path}")
    
    # Final evaluation with video recording
    print("\nRunning final evaluation...")
    eval_env_final = gym.make(
        env_id if reward_type == 'sparse' else env_id.replace('-v1', 'Dense-v1'),
        render_mode='rgb_array'
    )
    eval_env_final.reset(seed=seed + 2000)
    
    mean_reward, success_rate = evaluate_policy(
        model,
        eval_env_final,
        n_eval_episodes=n_eval_episodes,
        render=True,
        save_video=True,
        video_path=log_path / "videos"
    )
    
    # Cleanup
    train_env.close()
    eval_env.close()
    eval_env_final.close()
    
    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print(f"Final mean reward: {mean_reward:.2f}")
    print(f"Final success rate: {success_rate:.2%}")
    print(f"Results saved to: {log_path}")
    print("="*60)
    
    return model, mean_reward, success_rate


def main():
    """Main function to run the training."""
    
    # Configuration - Optimized for HandManipulateBlockRotateZ
    config = {
        'env_id': 'HandManipulateBlockRotateZ-v1',  # Easiest variant (z-axis rotation only)
        'reward_type': 'dense',  # Dense rewards are easier to learn than sparse
        'total_timesteps': 2_000_000,  # Increased for better convergence
        'learning_rate': 1e-3,  # Slightly higher LR for faster initial learning
        'buffer_size': 1_000_000,  # Large buffer for diverse experiences
        'batch_size': 256,  # Standard batch size
        'n_eval_episodes': 10,  # More episodes for reliable evaluation
        'eval_freq': 10_000,  # Evaluate every 10k steps
        'num_envs': 8,  # More parallel envs for better sample efficiency
        'seed': 42,
        'log_dir': 'logs/shadow_hand_poc',
        'save_model': True,
    }
    
    print("\n" + "="*60)
    print("CONFIGURATION:")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Run training
    model, mean_reward, success_rate = train_shadow_hand(**config)
    
    print("\nProof of concept completed successfully!")
    print(f"Model trained for {config['total_timesteps']:,} timesteps")
    #print(f"Final performance: {mean_reward:.2f} reward, {success_rate:.2%} success rate")
    """print("\nNOTE: The Shadow Hand manipulation tasks are extremely challenging.")
    print("Expected results for HandManipulateBlockRotateZ-v1 with dense rewards:")
    print("  - Initial reward: ~ -10 to -15")
    print("  - After 1M steps: ~ -3 to -8 (showing learning)")
    print("  - After 2M steps: ~ -2 to -5 (reasonable performance)")
    print("  - Success typically requires 3-5M steps or HER algorithm")"""
    print(f"\nLogs and videos saved to: {config['log_dir']}")
    
    return model


if __name__ == "__main__":
    main()
