import datetime
import json
import os
import sys
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
from dataclasses import dataclass
from typing import Optional
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from sbx import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np
import mediapy as media

# Import JAX and verify backend
import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Inform user about the backend being used
if jax.default_backend() == 'gpu':
    print("AX is using GPU acceleration")
elif jax.default_backend() == 'cpu':
    print("JAX is using CPU (GPU not available or not detected)")
else:
    print(f"JAX is using backend: {jax.default_backend()}")

# Set logging level to suppress JAX backend initialization messages
logging.set_verbosity(logging.WARNING)

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.planner.mpc_planner import MPCPlanner
from mpc_rl.sac_mpc.mpc_inject_callbacks import FixedMPCInjectCallback, PercentMPCInjectCallback
from mpc_rl.sac_mpc.sac_mpc import SAC_MPC
from mpc_rl.sac_mpc.tagged_replay_buffer import TaggedReplayBuffer


# Environment flags
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "cartpole-swingup",
    "Name of the dm_control environment (format: domain-task, e.g., cartpole-swingup)",
)
_DOMAIN = flags.DEFINE_string(
    "domain",
    None,
    "Domain name (e.g., cartpole). If None, will be parsed from env_name",
)
_TASK = flags.DEFINE_string(
    "task",
    None,
    "Task name (e.g., swingup). If None, will be parsed from env_name",
)

# Training flags
_ALGORITHM = flags.DEFINE_enum(
    "algorithm", "SAC", ["SAC", "PPO", "TD3", "SAC-MPC"], "RL algorithm to use"
)
_TOTAL_TIMESTEPS = flags.DEFINE_integer(
    "total_timesteps", 500_000, "Total number of timesteps to train"
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", 4, "Number of parallel environments for training"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")

# Evaluation flags
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only evaluate the model without training"
)
_LOAD_RUN_NAME = flags.DEFINE_string(
    "load_run_name", None, "Name of the run to load checkpoint from"
)
_NUM_EVAL_EPISODES = flags.DEFINE_integer(
    "num_eval_episodes", 5, "Number of episodes to evaluate"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 3, "Number of videos to record during evaluation"
)

# Experiment flags
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_LOGDIR = flags.DEFINE_string("logdir", "logs", "Base directory for logs")
_ENABLE_LOGGING = flags.DEFINE_boolean(
    "enable_logging", True, "Enable checkpoints, videos, and TensorBoard logging. Set to False for hyperparameter optimization with optuna."
)

# Hyperparameter flags (this is for SAC for now, not optimized yet)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
_BUFFER_SIZE = flags.DEFINE_integer("buffer_size", 1_000_000, "Replay buffer size")
_LEARNING_STARTS = flags.DEFINE_integer(
    "learning_starts", 10_000, "Steps of model to collect transitions before learning starts"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Minibatch size")
_TAU = flags.DEFINE_float("tau", 0.005, "Soft update coefficient")
_GAMMA = flags.DEFINE_float("gamma", 0.99, "Discount factor")

# MPC injection flags
_INJECT_N_TIMESTEPS = flags.DEFINE_integer(
    "inject_n_timesteps", 5000, "Inject MPC trajectories every N timesteps"
)
_INJECT_TYPE = flags.DEFINE_enum(
    "inject_type", "percentage", ["percentage", "fixed"], "Type of injection of MPC trajectories"
)
_PERCENTAGE = flags.DEFINE_integer(
    "percentage", 25, "Percentage of the replay buffer that should be MPC trajectories"
)
_NUM_TRAJ = flags.DEFINE_integer(
    "num_traj", 10, "Number of MPC trajectories to inject each time"
)
_RANDOM_SELECT = flags.DEFINE_boolean(
    "random_select", True, "Randomly select trajectories to inject"
)
_DATA_DIR = flags.DEFINE_string(
    "data_dir", "data/cartpole_0_001dt/", "Directory containing pre-generated MPC trajectories"
)

# Checkpoint flags
_CHECKPOINT_FREQ = flags.DEFINE_integer(
    "checkpoint_freq", 25_000, "Save checkpoint every N steps"
)
_EVAL_FREQ = flags.DEFINE_integer(
    "eval_freq", 10_000, "Evaluate policy every N steps"
)


@dataclass
class AllConfig:
    algorithm: str
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    seed: int
    tensorboard_log: str
    inject_n_timesteps: int
    inject_type: str
    percentage: int
    num_traj: int
    random_select: bool
    data_dir: str


def parse_env_name(env_name: str) -> tuple[str, str]:
    """
    Parse environment name into domain and task.
    
    Args:
        env_name: Environment name in format 'domain-task' or 'domain_task'
    
    Returns:
        Tuple of (domain, task)
    """
    # Replace underscores with hyphens and split
    env_name = env_name.replace("_", "-")
    parts = env_name.split("-")
    
    if len(parts) < 2:
        raise ValueError(
            f"Invalid env_name format: {env_name}. "
            "Expected format: 'domain-task' (e.g., 'cartpole-swingup')"
        )
    
    domain = parts[0]
    task = "-".join(parts[1:])  # Handle tasks with hyphens like 'stand-and-reach'
    return domain, task


def make_dm_env(domain: str, task: str, render_mode=None):
    """
    Create a dm_control environment wrapped for gymnasium.
    
    Args:
        domain: Domain name (e.g., 'cartpole')
        task: Task name (e.g., 'swingup')
        render_mode: Render mode for the environment
    
    Returns:
        Wrapped gymnasium environment
    """
    dm_env = suite.load(domain_name=domain, task_name=task)
    gym_env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    gym_env = FlattenObservation(gym_env)
    return gym_env


def create_experiment_name(env_name: str, algorithm: str, suffix: str = None,
                          inject_type: str = None, percentage: int = None) -> str:
    """Create unique experiment name with timestamp and algorithm."""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{algorithm}-{timestamp}"
    
    # Add injection type for SAC-MPC
    if algorithm == "SAC-MPC" and inject_type:
        exp_name += f"-{inject_type}"
        # Add percentage if using percentage-based injection
        if inject_type == "percentage" and percentage is not None:
            exp_name += f"-{percentage}pct"
    
    if suffix:
        exp_name += f"-{suffix}"
    return exp_name


def save_config(logdir: Path, config: dict):
    """Save configuration to JSON file."""
    config_path = logdir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")


def load_model(algorithm: str, model_path: Path, env):
    """Load a trained model."""
    algo_class = {"SAC": SAC, "PPO": PPO, "TD3": TD3}[algorithm]
    print(f"Loading model from: {model_path}")
    return algo_class.load(model_path, env=env)


def create_model(env, cfg):
    """
    Create a new model instance.
    
    Uses a factory pattern: algo_class is a class object (not an instance), selected by
    algorithm string. Calling algo_class(...) invokes the class constructor (__init__) to
    create a new agent instance with the specified hyperparameters.
    """
    algo_class = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "SAC-MPC": SAC_MPC}[cfg.algorithm]
    
    if cfg.algorithm == "SAC":
        model = algo_class(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=cfg.tensorboard_log,
        )
    elif cfg.algorithm == "SAC-MPC":
        model = algo_class(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            replay_buffer_class=TaggedReplayBuffer,  # Use custom tagged replay buffer
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=cfg.tensorboard_log,
        )
    elif cfg.algorithm == "PPO":
        model = algo_class(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=cfg.tensorboard_log,
        )
    elif cfg.algorithm == "TD3":
        model = algo_class(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=cfg.tensorboard_log,
        )
    
    return model


def create_callbacks(cfg: AllConfig, enable_logging: bool, logdir: Path, 
                     domain: str, task: str, seed: int,
                     checkpoint_freq: int, eval_freq: int):
    """
    Factory function to create all callbacks based on configuration.
    
    Args:
        cfg: AllConfig containing algorithm and MPC injection parameters
        enable_logging: Whether to enable checkpoints and eval callbacks
        logdir: Path to log directory
        domain: Environment domain name
        task: Environment task name
        seed: Random seed for eval environment
        checkpoint_freq: Frequency to save checkpoints
        eval_freq: Frequency to run evaluation
    
    Returns:
        Tuple of (callbacks list, eval_env or None, inject_callback or None)
        eval_env is returned so it can be closed after training
    """
    callbacks = []
    eval_env = None
    inject_callback = None  # Initialize to None for non-SAC-MPC algorithms
    
    # Add checkpoint callback if logging is enabled
    if enable_logging:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(logdir / "checkpoints"),
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
    
    # Add eval callback if logging is enabled
    if enable_logging:
        # Create eval environment for evaluation callback
        # Must be wrapped the same way as training env (with VecNormalize)
        eval_env = make_vec_env(
            lambda: make_dm_env(domain, task),
            n_envs=1,
            seed=seed + 1000,
        )
        eval_env = VecNormalize(
            eval_env,
            training=False,  # Don't update stats during evaluation
            norm_obs=True,
            norm_reward=True,
        )
        
        # Create callback for evaluating the trained model
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(logdir / "best_model"),
            log_path=str(logdir / "eval_logs"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )
        callbacks.append(eval_callback)
    
    # Add MPC injection callback if using SAC-MPC
    if cfg.algorithm == "SAC-MPC":
        if _INJECT_TYPE.value == "fixed":
            print("\nSetting up FIXED MPC Injection from pre-generated trajectories...")
            inject_callback = FixedMPCInjectCallback(
                inject_every_n_timesteps=cfg.inject_n_timesteps,
                num_mpc_trajectories=cfg.num_traj,
                data_dir=cfg.data_dir,
                random_select=cfg.random_select,
                seed=seed,  # Pass seed for reproducible trajectory selection
                verbose=1,
            )
        elif _INJECT_TYPE.value == "percentage":
            print("\nSetting up PERCENTAGE MPC Injection from pre-generated trajectories...")
            inject_callback = PercentMPCInjectCallback(
                target_percentage=cfg.percentage,
                data_dir=cfg.data_dir,
                random_select=cfg.random_select,
                seed=seed,  # Pass seed for reproducible trajectory selection
                verbose=1,
            )
        callbacks.append(inject_callback)
        
        # Store reference to callback in list so model can access it later
        return (callbacks if callbacks else None), eval_env, inject_callback if cfg.algorithm == "SAC-MPC" else None
    
    return (callbacks if callbacks else None), eval_env, None


def evaluate_and_record(model, domain: str, task: str, num_episodes: int, 
                        num_videos: int, video_dir: Path, normalize_env=None):
    """
    Evaluate model and record videos.
    
    Args:
        model: Trained model
        domain: Environment domain
        task: Environment task
        num_episodes: Number of episodes to evaluate
        num_videos: Number of videos to record
        video_dir: Directory to save videos
        normalize_env: VecNormalize wrapper for observation normalization
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        # Create evaluation environment
        eval_env_base = make_dm_env(domain, task, render_mode="rgb_array")
        
        # Wrap in VecEnv for compatibility with model
        eval_env = DummyVecEnv([lambda: eval_env_base])
        
        # Apply normalization if available
        if normalize_env is not None:
            eval_env = VecNormalize.load(
                normalize_env, 
                eval_env
            )
            eval_env.training = False
            eval_env.norm_reward = False
        
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        frames = []
        
        # Record video for first few episodes
        record_video = episode < num_videos
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            # Capture frames for video
            if record_video:
                frame = eval_env_base.render()
                if frame is not None:
                    frames.append(frame)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Save video
        if record_video and frames:
            video_path = video_dir / f"rollout{episode}.mp4"
            # Assuming 30 FPS for dm_control environments
            fps = 30
            media.write_video(str(video_path), frames, fps=fps)
            print(f"Video saved to: {video_path}")
        
        eval_env.close()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Evaluation Summary:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")
    print("="*50)


def main(argv):
    """
    Main training and evaluation function.
    """
    del argv # Not used since we're using absl for flags
    
    # Parse environment name
    if _DOMAIN.value and _TASK.value:
        domain = _DOMAIN.value
        task = _TASK.value
        env_name = f"{domain}-{task}"
    else:
        domain, task = parse_env_name(_ENV_NAME.value)
        env_name = _ENV_NAME.value
    
    print(f"Environment: {domain}/{task}")
    
    # Determine if we're loading a checkpoint
    if _LOAD_RUN_NAME.value:
        # Load from existing run
        run_name = _LOAD_RUN_NAME.value
        logdir = Path(_LOGDIR.value) / run_name
        
        if not logdir.exists():
            raise ValueError(f"Run directory not found: {logdir}")
        
        print(f"Loading from run: {run_name}")
    else:
        # Create new experiment
        run_name = create_experiment_name(
            env_name, 
            _ALGORITHM.value, 
            _SUFFIX.value,
            inject_type=_INJECT_TYPE.value if _ALGORITHM.value == "SAC-MPC" else None,
            percentage=_PERCENTAGE.value if _ALGORITHM.value == "SAC-MPC" else None
        )
        logdir = Path(_LOGDIR.value) / run_name
        logdir.mkdir(parents=True, exist_ok=True)
        print(f"Created new run: {run_name}")
    
    print(f"Log directory: {logdir}")
    
    # Set up directories
    checkpoint_dir = logdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    video_dir = logdir / "videos"
    
    # Determine TensorBoard logging path
    if _ENABLE_LOGGING.value:
        tensorboard_log_path = str(logdir / "tensorboard")
    else:
        tensorboard_log_path = None
    
    # Create configuration object (for both new and loaded runs)
    config = AllConfig(
        algorithm=_ALGORITHM.value,
        learning_rate=_LEARNING_RATE.value,
        buffer_size=_BUFFER_SIZE.value,
        learning_starts=_LEARNING_STARTS.value,
        batch_size=_BATCH_SIZE.value,
        tau=_TAU.value,
        gamma=_GAMMA.value,
        seed=_SEED.value,
        tensorboard_log=tensorboard_log_path,
        inject_n_timesteps=_INJECT_N_TIMESTEPS.value,
        inject_type=_INJECT_TYPE.value,
        percentage=_PERCENTAGE.value,
        num_traj=_NUM_TRAJ.value,
        random_select=_RANDOM_SELECT.value,
        data_dir=_DATA_DIR.value,
    )
    
    # Save configuration (only for new runs)
    if not _LOAD_RUN_NAME.value:
        # Convert dataclass to dict and add environment info
        from dataclasses import asdict
        config_dict = asdict(config)
        config_dict.update({
            "env_name": env_name,
            "domain": domain,
            "task": task,
            "total_timesteps": _TOTAL_TIMESTEPS.value,
            "num_envs": _NUM_ENVS.value,
        })
        save_config(logdir, config_dict)
    
    # Create training environment
    print(f"Creating {_NUM_ENVS.value} parallel environments...")
    vec_env = make_vec_env(
        lambda: make_dm_env(domain, task), # lambda fxn so make_vec_env() can make multiple envs
        n_envs=_NUM_ENVS.value,
        seed=_SEED.value,
    )

    # VecNormalize standardizes observations and rewards to ~N(0,1), which is critical for
    # stable learning in continuous control (prevents different-scale features from dominating)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Path for the final model and normalization stats
    model_path = logdir / "final_model"
    vec_normalize_path = logdir / "vec_normalize.pkl"
    replay_buffer_path = logdir / "replay_buffer.pkl"
    
    # Load or create model
    # NOTE: SB3/SBX saves models with .zip extension but load() doesn't require it
    if _LOAD_RUN_NAME.value and (model_path.with_suffix('.zip').exists() or model_path.exists()):
        # Load existing model
        model = load_model(_ALGORITHM.value, model_path, vec_env)
        print("Model loaded successfully")
        
        # Load normalization stats
        if vec_normalize_path.exists():
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            print("Normalization stats loaded")
        
        # Load replay buffer (for off-policy algorithms)
        if replay_buffer_path.exists() and hasattr(model, 'load_replay_buffer'):
            model.load_replay_buffer(replay_buffer_path)
            print(f"Replay buffer loaded (size: {model.replay_buffer.size()})")
    else:
        # Create new model
        print(f"Creating new {_ALGORITHM.value} model...")
        model = create_model(
            env=vec_env,
            cfg=config,
        )
    
    # Training phase
    if not _PLAY_ONLY.value and _TOTAL_TIMESTEPS.value > 0:
        print(f"\nStarting training for {_TOTAL_TIMESTEPS.value} timesteps...")
        
        # Create callbacks using factory function
        callbacks, eval_env, mpc_inject_callback = create_callbacks(
            cfg=config,
            enable_logging=_ENABLE_LOGGING.value,
            logdir=logdir,
            domain=domain,
            task=task,
            seed=_SEED.value,
            checkpoint_freq=_CHECKPOINT_FREQ.value,
            eval_freq=_EVAL_FREQ.value,
        )
        
        # If using SAC-MPC with percentage injection, connect the callback to the model
        if _ALGORITHM.value == "SAC-MPC" and _INJECT_TYPE.value == "percentage" and mpc_inject_callback is not None:
            model.target_mpc_percentage = config.percentage
            model.mpc_inject_callback = mpc_inject_callback
            print(f"Connected MPC injection callback to SAC_MPC (target: {config.percentage}%)")
        
        # Train the model
        # When resuming, reset_num_timesteps=False continues from loaded timestep count
        model.learn(
            total_timesteps=_TOTAL_TIMESTEPS.value,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False if _LOAD_RUN_NAME.value else True,
        )
        
        print("Training complete!")
        
        # Save final model, normalization stats, and replay buffer (only if logging enabled)
        if _ENABLE_LOGGING.value:
            print(f"Saving model to: {model_path}")
            model.save(model_path)
            vec_env.save(vec_normalize_path)
            
            # Save replay buffer (for off-policy algorithms)
            if hasattr(model, 'save_replay_buffer'):
                model.save_replay_buffer(replay_buffer_path)
                print(f"Replay buffer saved (size: {model.replay_buffer.size()})")
            
            print("Model and normalization stats saved")
        
        # Close eval environment if it was created
        if eval_env is not None:
            eval_env.close()
    
    # Evaluation phase (only if logging enabled)
    if _ENABLE_LOGGING.value:
        print(f"\nEvaluating model for {_NUM_EVAL_EPISODES.value} episodes...")
        evaluate_and_record(
            model=model,
            domain=domain,
            task=task,
            num_episodes=_NUM_EVAL_EPISODES.value,
            num_videos=_NUM_VIDEOS.value,
            video_dir=video_dir,
            normalize_env=vec_normalize_path if vec_normalize_path.exists() else None,
        )
    
    vec_env.close()
    print("\nDone!")


if __name__ == "__main__":
    app.run(main)