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
    
    # Configuration flags for GPU
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
# SB3 PyTorch SAC/TD3 for quadruped environments (supports custom asymmetric policies)
# SBX (JAX) is used for other environments for speed; SB3 is used for quadruped because
# asymmetric actor-critic requires custom PyTorch feature extractors.
from stable_baselines3 import SAC as SB3_SAC, TD3 as SB3_TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import mediapy as media
import jax

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

#from mpc_rl.planner.mpc_planner import MPCPlanner
from mpc_rl.common import TaggedReplayBuffer, TaggedDictReplayBuffer
from mpc_rl.common import FixedMPCInjectCallback, PercentMPCInjectCallback, QuadrupedTensorboardCallback
from mpc_rl.sac_mpc.sac_mpc import SAC_MPC
from mpc_rl.td3_mpc.td3_mpc import TD3_MPC
# SB3 (PyTorch) MPC-augmented algorithms for quadruped (supports asymmetric policies + Dict obs)
from mpc_rl.sac_mpc.sb3_sac_mpc import SB3_SAC_MPC
from mpc_rl.td3_mpc.sb3_td3_mpc import SB3_TD3_MPC

# Register custom quadruped velocity tracking environment
import mpc_rl.envs
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.go2_sysid import assert_go2_sysid_joint_dynamics

# Asymmetric actor-critic policies for quadruped sim2real training
from mpc_rl.asym_policies import AsymmetricSACPolicy, AsymmetricTD3Policy

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

# Quadruped environment flags
_ROBOT = flags.DEFINE_string(
    "robot", "go2",
    "Quadruped robot model name (go2, go1, mini_cheetah, aliengo). Only used when env_name starts with 'quadruped-'",
)
_USE_GO2_SYSID = flags.DEFINE_boolean(
    "use_go2_sysid", True,
    "Apply the identified Go2 joint-dynamics patch to quadruped envs and MPX "
    "controllers. Disable this to match pre-sysID data and training runs."
)
_MAX_EPISODE_STEPS = flags.DEFINE_integer(
    "max_episode_steps", 1000, "Maximum number of steps per episode"
)

# Training flags
_ALGORITHM = flags.DEFINE_enum(
    "algorithm", "SAC", ["SAC", "PPO", "TD3", "SAC-MPC", "TD3-MPC"], "RL algorithm to use"
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
_CHECKPOINT_EVALS = flags.DEFINE_string(
    "checkpoint_evals", None,
    "Comma-separated checkpoint steps for additional video evaluations, "
    "for example '400000,500000'. Each checkpoint writes videos to "
    "logdir/video_<step>/."
)

# Experiment flags
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_LOGDIR = flags.DEFINE_string("logdir", "logs", "Base directory for logs")
_ENABLE_LOGGING = flags.DEFINE_boolean(
    "enable_logging", True, "Enable checkpoints, videos, and TensorBoard logging. Set to False for hyperparameter optimization with optuna."
)

# Hyperparameter flags (not optimized yet)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
_BUFFER_SIZE = flags.DEFINE_integer("buffer_size", 1_000_000, "Replay buffer size")
_LEARNING_STARTS = flags.DEFINE_integer(
    "learning_starts", 10_000, "Steps of model to collect transitions before learning starts"
)
_POLICY_DELAY = flags.DEFINE_integer("policy_delay", 2, "TD3 (only) actor update delay. Actor and target networks update once every policy_delay critic updates.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Minibatch size")
_TAU = flags.DEFINE_float("tau", 0.005, "Soft update coefficient")
_GAMMA = flags.DEFINE_float("gamma", 0.99, "Discount factor")
_GRADIENT_STEPS = flags.DEFINE_integer(
    "gradient_steps", -1,
    "Number of gradient steps per environment step. "
    "-1 means as many gradient steps as env steps collected per rollout "
    "(=num_envs). Higher values improve sample efficiency for off-policy "
    "algorithms (SAC/TD3). Default 1 is too conservative for multi-env training."
)

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
    "data_dir", None, "Directory containing pre-generated MPC trajectories (e.g., 'data/cartpole_0_010dt/' or 'data/walker_0_0025dt/')"
)

# Checkpoint flags
_CHECKPOINT_FREQ = flags.DEFINE_integer(
    "checkpoint_freq", 25_000, "Save checkpoint every N steps"
)
_EVAL_FREQ = flags.DEFINE_integer(
    "eval_freq", 10_000, "Evaluate policy every N steps"
)
_SAVE_REPLAY_BUFFER_CHECKPOINTS = flags.DEFINE_boolean(
    "save_replay_buffer_checkpoints", False,
    "Save replay buffer at each checkpoint. Disable to greatly reduce disk usage."
)
_SAVE_REPLAY_BUFFER_FINAL = flags.DEFINE_boolean(
    "save_replay_buffer_final", False,
    "Save replay buffer at end of training for resume support. Disable for model-only artifacts."
)

# Domain randomization flags
_DOMAIN_RAND = flags.DEFINE_boolean(
    "domain_rand", True,
    "Enable domain randomization for quadruped environments. "
    "Randomizes friction, mass, PD gains, observation noise, and applies "
    "periodic push perturbations for improved sim-to-real transfer."
)
_DOMAIN_RAND_CONFIG_TYPE = flags.DEFINE_enum(
    "domain_rand_config_type", "custom",
    [
        "custom",
        "default",
        "default_no_push",
        "half_no_push",
        "quarter_no_push",
        "sysid_dyn10_default",
        "sysid_dyn10_default_no_push",
        "sysid_dyn10_half_no_push",
        "sysid_dyn20_mjlab",
        "sysid_dyn20_mjlab_no_push",
        "sysid_floor_only_no_push",
        "sysid_floor_sensing_no_push",
        "disabled",
    ],
    "Named quadruped domain-randomization preset. "
    "'custom' preserves the legacy flag-driven behavior where only "
    "--domain_rand_obs_noise overrides the default config."
)
_DOMAIN_RAND_OBS_NOISE = flags.DEFINE_float(
    "domain_rand_obs_noise", 1.0,
    "Observation noise level for domain randomization (0.0 = no noise, 1.0 = full)."
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
    gradient_steps: int
    policy_delay: int
    seed: int
    tensorboard_log: str
    inject_n_timesteps: int
    inject_type: str
    percentage: int
    num_traj: int
    random_select: bool
    data_dir: str
    use_go2_sysid: bool


def parse_env_name(env_name: str) -> tuple[str, str]:
    """
    Parse environment name into domain and task.
    
    Args:
        env_name: Environment name in format 'domain-task' or 'domain_task'
    
    Returns:
        Tuple of (domain, task)
    """
    # Split on hyphen only (preserve underscores in task names like 'swingup_sparse')
    parts = env_name.split("-")
    
    if len(parts) < 2:
        raise ValueError(
            f"Invalid env_name format: {env_name}. "
            "Expected format: 'domain-task' (e.g., 'cartpole-swingup')"
        )
    
    domain = parts[0]
    task = "-".join(parts[1:])  # Handle tasks with hyphens like 'stand-and-reach'
    return domain, task


def parse_checkpoint_eval_steps(checkpoint_evals: Optional[str]) -> list[int]:
    """
    Parse a checkpoint evaluation flag into a list of unique positive steps.

    Accepts values like:
    - "400000"
    - "400000,500000"
    - "(400000, 500000)"
    - "[400000,500000]"
    """
    if checkpoint_evals is None:
        return []

    cleaned = checkpoint_evals.strip()
    if not cleaned:
        return []

    cleaned = cleaned.strip("()[]")
    tokens = []
    for chunk in cleaned.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens.extend(part for part in chunk.split() if part)

    checkpoint_steps = []
    seen_steps = set()
    for token in tokens:
        try:
            step = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid checkpoint step '{token}' in --checkpoint_evals={checkpoint_evals!r}. "
                "Use a comma-separated list of integers, e.g. "
                "--checkpoint_evals=400000,500000"
            ) from exc

        if step <= 0:
            raise ValueError(
                f"Checkpoint steps must be positive integers, got {step}."
            )

        if step not in seen_steps:
            checkpoint_steps.append(step)
            seen_steps.add(step)

    return checkpoint_steps


class ExactTimestepCheckpointCallback(BaseCallback):
    """Save checkpoints at exact env-step milestones, even with vectorized envs.

    SB3's built-in CheckpointCallback saves every N callback calls, so the common
    ``checkpoint_freq // num_envs`` conversion floors the desired interval when
    ``checkpoint_freq`` is not divisible by ``num_envs``. That causes drift such as
    25,000-step checkpoints being written every 24,832 steps when ``num_envs=256``.

    This callback instead tracks the next desired env-step milestone directly and
    saves as soon as training crosses it, naming the checkpoint with the requested
    milestone (e.g. ``model_300000_steps.zip``).
    """

    def __init__(
        self,
        save_freq_steps: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if save_freq_steps <= 0:
            raise ValueError(f"save_freq_steps must be positive, got {save_freq_steps}")

        self.save_freq_steps = save_freq_steps
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.next_save_step = save_freq_steps

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        current_steps = int(self.model.num_timesteps)
        self.next_save_step = ((current_steps // self.save_freq_steps) + 1) * self.save_freq_steps

    def _checkpoint_path(self, step: int, checkpoint_type: str = "", extension: str = "") -> Path:
        return self.save_path / f"{self.name_prefix}_{checkpoint_type}{step}_steps.{extension}"

    def _save_checkpoint(self, target_step: int) -> None:
        model_path = self._checkpoint_path(target_step, extension="zip")
        self.model.save(model_path)
        if self.verbose >= 2:
            print(f"Saving model checkpoint to {model_path}")

        if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            replay_buffer_path = self._checkpoint_path(target_step, "replay_buffer_", extension="pkl")
            self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
            if self.verbose > 1:
                print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            vecnormalize_path = self._checkpoint_path(target_step, "vecnormalize_", extension="pkl")
            self.model.get_vec_normalize_env().save(vecnormalize_path)  # type: ignore[union-attr]
            if self.verbose >= 2:
                print(f"Saving model VecNormalize to {vecnormalize_path}")

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_save_step:
            self._save_checkpoint(self.next_save_step)
            self.next_save_step += self.save_freq_steps
        return True


def is_shadow_hand_env(env_name: str) -> bool:
    """
    Check if the environment is a shadow hand environment.
    
    Args:
        env_name: Environment name
    
    Returns:
        True if it's a shadow hand environment
    """
    shadow_hand_envs = [
        "ShadowHandManipulateBlockRotateXYZ-v1",
        "ShadowHandManipulateBlockRotateXYZDense-v1",
    ]
    return env_name in shadow_hand_envs


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


def make_shadow_hand_env(env_name: str, render_mode=None):
    """
    Create a shadow hand gymnasium environment.
    
    Args:
        env_name: Full environment name (e.g., 'ShadowHandManipulateBlockRotateXYZ-v1')
        render_mode: Render mode for the environment
    
    Returns:
        Shadow hand gymnasium environment
    """
    gym_env = gym.make(env_name, render_mode=render_mode)
    gym_env = FlattenObservation(gym_env)
    return gym_env


def is_quadruped_env(env_name: str) -> bool:
    """Check if the environment is a quadruped velocity tracking environment.
    
    Convention: env_name starts with 'quadruped-' (e.g., 'quadruped-velocity_tracking').
    """
    return env_name.lower().startswith("quadruped-")


def make_quadruped_env(robot: str = "go2", render_mode=None, domain_rand_cfg=None,
                       simple_reward: bool = False,
                       use_go2_sysid: bool = True):
    """
    Create a quadruped velocity tracking gymnasium environment.
    
    Returns a Dict observation space for asymmetric actor-critic training:
        "policy" (45-dim): Real-hardware-available sensor observations (actor)
        "privileged" (3-dim): Simulation-only ground truth base_lin_vel (critic)
    
    No FlattenObservation wrapper is applied since Dict obs is required
    for the asymmetric policy architecture.
    
    Args:
        robot: Robot model name (go2, go1, mini_cheetah, aliengo, etc.)
        render_mode: Render mode for the environment
        domain_rand_cfg: Domain randomization config. None uses defaults.
        simple_reward: If True, use simplified reward (velocity tracking + termination only).
            Used when training with MPC injection (SAC-MPC/TD3-MPC).
        use_go2_sysid: If True, apply the identified Go2 joint dynamics.
            Ignored for non-Go2 robots.
    
    Returns:
        QuadrupedVelocityTracking gymnasium environment with Dict obs space
    """
    kwargs = dict(
        robot=robot,
        render_mode=render_mode,
        max_episode_steps=_MAX_EPISODE_STEPS.value,
        use_go2_sysid=use_go2_sysid,
    )
    if domain_rand_cfg is not None:
        kwargs["domain_rand_cfg"] = domain_rand_cfg
    if simple_reward:
        kwargs["simple_reward"] = True
    gym_env = gym.make(
        "QuadrupedVelocityTracking-v0",
        **kwargs,
    )
    if robot.lower() == "go2" and use_go2_sysid:
        dr_cfg = gym_env.unwrapped.domain_rand_cfg
        dynamics_dr_enabled = bool(
            dr_cfg.enable and (
                dr_cfg.joint_damping_scale_range[0] != dr_cfg.joint_damping_scale_range[1]
                or dr_cfg.joint_armature_scale_range[0] != dr_cfg.joint_armature_scale_range[1]
                or dr_cfg.joint_friction_scale_range[0] != dr_cfg.joint_friction_scale_range[1]
                or dr_cfg.joint_friction_range[0] != dr_cfg.joint_friction_range[1]
            )
        )
        # With dynamics DR presets (for example sysid_dyn10_*), exact equality
        # to the canonical sysID table is intentionally broken at startup.
        if not dynamics_dr_enabled:
            assert_go2_sysid_joint_dynamics(gym_env.unwrapped.mjModel)
    return gym_env


def build_quadruped_domain_rand_config() -> DomainRandomizationConfig:
    """Resolve the quadruped domain-randomization config from CLI flags."""
    config_type = _DOMAIN_RAND_CONFIG_TYPE.value

    if not _DOMAIN_RAND.value or config_type == "disabled":
        return DomainRandomizationConfig.disabled()

    if config_type == "custom":
        return DomainRandomizationConfig(
            enable=True,
            obs_noise_level=_DOMAIN_RAND_OBS_NOISE.value,
        )

    return DomainRandomizationConfig.from_preset(config_type)


def create_experiment_name(env_name: str, algorithm: str, suffix: str = None,
                          inject_type: str = None, percentage: int = None) -> str:
    """Create unique experiment name with timestamp and algorithm."""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{algorithm}-{timestamp}"
    
    # Add injection type for SAC-MPC or TD3-MPC
    if algorithm in ["SAC-MPC", "TD3-MPC"] and inject_type:
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


def make_single_env_for_model_loading(domain: str, task: str,
                                      is_quadruped: bool = False,
                                      robot: str = "go2",
                                      simple_reward: bool = False,
                                      use_go2_sysid: bool = True):
    """Create a single-env VecEnv for loading a saved model."""
    is_shadow_hand = (domain == "shadow_hand")

    if is_quadruped:
        return DummyVecEnv([
            lambda: make_quadruped_env(
                robot=robot,
                domain_rand_cfg=DomainRandomizationConfig.disabled(),
                simple_reward=simple_reward,
                use_go2_sysid=use_go2_sysid,
            )
        ])
    if is_shadow_hand:
        return DummyVecEnv([lambda: make_shadow_hand_env(task)])
    return DummyVecEnv([lambda: make_dm_env(domain, task)])


def load_saved_model_for_video_eval(algorithm: str, model_path: Path,
                                    vecnormalize_path: Optional[Path],
                                    domain: str, task: str,
                                    is_quadruped: bool = False,
                                    robot: str = "go2",
                                    simple_reward: bool = False,
                                    use_go2_sysid: bool = True):
    """Load a saved model plus its VecNormalize stats for video evaluation."""
    model_env = make_single_env_for_model_loading(
        domain=domain,
        task=task,
        is_quadruped=is_quadruped,
        robot=robot,
        simple_reward=simple_reward,
        use_go2_sysid=use_go2_sysid,
    )

    if vecnormalize_path is not None and vecnormalize_path.exists():
        model_env = VecNormalize.load(vecnormalize_path, model_env)
        model_env.training = False
        model_env.norm_reward = False

    model = load_model(algorithm, model_path, model_env, is_quadruped=is_quadruped)
    return model, model_env


def load_model(algorithm: str, model_path: Path, env, is_quadruped: bool = False):
    """Load a trained model.
    
    For quadruped environments, uses SB3 (PyTorch) SAC/TD3 with asymmetric policies.
    For other environments, uses SBX (JAX) implementations.
    """
    if is_quadruped:
        # Quadruped uses SB3 PyTorch with asymmetric policy
        algo_class = {
            "SAC": SB3_SAC, "TD3": SB3_TD3,
            "SAC-MPC": SB3_SAC_MPC, "TD3-MPC": SB3_TD3_MPC,
        }[algorithm]
    else:
        algo_class = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "SAC-MPC": SAC_MPC, "TD3-MPC": TD3_MPC}[algorithm]
    print(f"Loading model from: {model_path}")
    return algo_class.load(model_path, env=env)


def create_model(env, cfg, is_quadruped: bool = False):
    """
    Create a new model instance.
    
    Uses a factory pattern: algo_class is a class object (not an instance), selected by
    algorithm string. Calling algo_class(...) invokes the class constructor (__init__) to
    create a new agent instance with the specified hyperparameters.
    
    For quadruped environments, uses SB3 (PyTorch) SAC/TD3 with asymmetric actor-critic
    policies where the actor only sees hardware-available observations and the critic
    also receives privileged simulation data (base linear velocity).
    """
    if is_quadruped:
        # Quadruped: use SB3 PyTorch with asymmetric actor-critic policies
        # Actor sees only "policy" obs (45-dim), critic sees "policy"+"privileged" (48-dim)
        if cfg.algorithm == "SAC":
            model = SB3_SAC(
                AsymmetricSACPolicy,
                env,
                learning_rate=cfg.learning_rate,
                buffer_size=cfg.buffer_size,
                learning_starts=cfg.learning_starts,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                gamma=cfg.gamma,
                gradient_steps=cfg.gradient_steps,
                verbose=1,
                seed=cfg.seed,
                tensorboard_log=cfg.tensorboard_log,
            )
        elif cfg.algorithm == "TD3":
            model = SB3_TD3(
                AsymmetricTD3Policy,
                env,
                learning_rate=cfg.learning_rate,
                buffer_size=cfg.buffer_size,
                learning_starts=cfg.learning_starts,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                gamma=cfg.gamma,
                gradient_steps=cfg.gradient_steps,
                policy_delay=cfg.policy_delay,
                verbose=1,
                seed=cfg.seed,
                tensorboard_log=cfg.tensorboard_log,
            )
        elif cfg.algorithm == "SAC-MPC":
            model = SB3_SAC_MPC(
                AsymmetricSACPolicy,
                env,
                # Intentionally widen only quadruped SAC-MPC; all other algorithms keep their current defaults.
                #policy_kwargs=dict(net_arch=[512, 512]),
                learning_rate=cfg.learning_rate,
                buffer_size=cfg.buffer_size,
                learning_starts=cfg.learning_starts,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                gamma=cfg.gamma,
                gradient_steps=cfg.gradient_steps,
                replay_buffer_class=TaggedDictReplayBuffer,
                verbose=1,
                seed=cfg.seed,
                tensorboard_log=cfg.tensorboard_log,
            )
        elif cfg.algorithm == "TD3-MPC":
            model = SB3_TD3_MPC(
                AsymmetricTD3Policy,
                env,
                learning_rate=cfg.learning_rate,
                buffer_size=cfg.buffer_size,
                learning_starts=cfg.learning_starts,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                gamma=cfg.gamma,
                gradient_steps=cfg.gradient_steps,
                policy_delay=cfg.policy_delay,
                replay_buffer_class=TaggedDictReplayBuffer,
                verbose=1,
                seed=cfg.seed,
                tensorboard_log=cfg.tensorboard_log,
            )
        else:
            raise ValueError(
                f"Algorithm '{cfg.algorithm}' is not supported for quadruped environments. "
                "Use SAC, TD3, SAC-MPC, or TD3-MPC."
            )
        return model
    
    # Non-quadruped: use SBX (JAX) for faster training
    algo_class = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "SAC-MPC": SAC_MPC, "TD3-MPC": TD3_MPC}[cfg.algorithm]
    
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
    elif cfg.algorithm == "TD3-MPC":
        model = algo_class(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            gradient_steps=cfg.gradient_steps,
            policy_delay=cfg.policy_delay,
            replay_buffer_class=TaggedReplayBuffer,
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
            gradient_steps=cfg.gradient_steps,
            policy_delay=cfg.policy_delay,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=cfg.tensorboard_log,
        )
    
    return model


def create_callbacks(cfg: AllConfig, enable_logging: bool, logdir: Path, 
                     domain: str, task: str, seed: int,
                     checkpoint_freq: int, eval_freq: int, num_envs: int,
                     is_quadruped: bool = False, robot: str = "go2",
                     save_replay_buffer_checkpoints: bool = False,
                     simple_reward: bool = False,
                     use_go2_sysid: bool = True,
                     domain_rand_config_type: str = "disabled"):
    """
    Factory function to create all callbacks based on configuration.
    
    Args:
        cfg: AllConfig containing algorithm and MPC injection parameters
        enable_logging: Whether to enable checkpoints and eval callbacks
        logdir: Path to log directory
        domain: Environment domain name
        task: Environment task name
        seed: Random seed for eval environment
        checkpoint_freq: Frequency to save checkpoints (in environment steps)
        eval_freq: Frequency to run evaluation (in environment steps)
        num_envs: Number of parallel environments
        is_quadruped: Whether the environment is a quadruped velocity tracking env
        robot: Quadruped robot model name (only used when is_quadruped=True)
        save_replay_buffer_checkpoints: Whether to save replay buffer at every checkpoint
        simple_reward: If True, use simplified reward for quadruped eval envs
    
    Returns:
        Tuple of (callbacks list, eval_env or None, inject_callback or None)
        eval_env is returned so it can be closed after training
    """
    callbacks = []
    eval_env = None
    inject_callback = None  # Initialize to None for non-SAC-MPC algorithms

    # Add rollout Tensorboard callback for quadruped TD3-MPC or SAC-MPC
    if is_quadruped and cfg.algorithm in ["SAC-MPC", "TD3-MPC"]:
        callbacks.append(QuadrupedTensorboardCallback(log_freq=100))
    
    # Add checkpoint callback if logging is enabled
    if enable_logging:
        # Save checkpoints against exact env-step milestones instead of
        # floor-dividing by num_envs, which drifts when the interval is not
        # divisible by the vectorized env count (e.g. 25,000 // 256 = 97).
        checkpoint_callback = ExactTimestepCheckpointCallback(
            save_freq_steps=checkpoint_freq,
            save_path=str(logdir / "checkpoints"),
            name_prefix="model",
            save_replay_buffer=save_replay_buffer_checkpoints,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
    
    # Add eval callback if logging is enabled
    if enable_logging:
        # Create eval environment for evaluation callback
        # Must be wrapped the same way as training env (with VecNormalize)
        # Determine environment type from domain
        is_shadow_hand = (domain == "shadow_hand")
        if is_quadruped:
            _dr_eval = DomainRandomizationConfig.disabled()
            _sr_eval = simple_reward  # capture for lambda closure
            eval_env = make_vec_env(
                lambda: make_quadruped_env(robot=robot, domain_rand_cfg=_dr_eval,
                                          simple_reward=_sr_eval,
                                          use_go2_sysid=use_go2_sysid),
                n_envs=1,
                seed=seed+1000,
            )
        elif is_shadow_hand:
            eval_env = make_vec_env(
                lambda: make_shadow_hand_env(task),  # task contains the full env name
                n_envs=1,
                seed=seed+1000,
            )
        else:
            eval_env = make_vec_env(
                lambda: make_dm_env(domain, task),
                n_envs=1,
                seed=seed+1000,
            )
        eval_env = VecNormalize(
            eval_env,
            training=False,  # Don't update stats during evaluation
            norm_obs=True,
            norm_reward=True,
        )
        # Reseed after VecNormalize wrapping
        #eval_env.seed(seed + 1000)
        #eval_env.action_space.seed(seed + 1000)
        #eval_env.observation_space.seed(seed + 1000)
        
        # Create callback for evaluating the trained model
        # EvalCallback's eval_freq is also per training step, so divide by num_envs
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(logdir / "best_model"),
            log_path=str(logdir / "eval_logs"),
            eval_freq=eval_freq // num_envs,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )
        callbacks.append(eval_callback)
    
    # Add MPC injection callback if using SAC-MPC or TD3-MPC
    if cfg.algorithm in ["SAC-MPC", "TD3-MPC"]:
        if _INJECT_TYPE.value == "fixed":
            print("\nSetting up FIXED MPC Injection from pre-generated trajectories...")
            inject_callback = FixedMPCInjectCallback(
                domain=domain,
                task=task,
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
                domain=domain,
                task=task,
                target_percentage=cfg.percentage,
                data_dir=cfg.data_dir,
                random_select=cfg.random_select,
                seed=seed,  # Pass seed for reproducible trajectory selection
                robot=robot if is_quadruped else "go2",
                use_go2_sysid=use_go2_sysid,
                expected_dr_config_type=(
                    domain_rand_config_type if is_quadruped else None
                ),
                verbose=1,
            )
        callbacks.append(inject_callback)
        
        # Store reference to callback in list so model can access it later
        return (callbacks if callbacks else None), eval_env, inject_callback if cfg.algorithm in ["SAC-MPC", "TD3-MPC"] else None
    
    return (callbacks if callbacks else None), eval_env, None


def evaluate_and_record(model, domain: str, task: str, num_episodes: int, 
                        num_videos: int, video_dir: Path, normalize_env=None, seed: int = None,
                        is_quadruped: bool = False, robot: str = "go2",
                        simple_reward: bool = False,
                        use_go2_sysid: bool = True):
    """
    Evaluate model and record videos.
    
    Args:
        model: Trained model
        domain: Environment domain
        task: Environment task (for shadow_hand, this is the full env name)
        num_episodes: Number of episodes to evaluate
        num_videos: Number of videos to record
        video_dir: Directory to save videos
        normalize_env: VecNormalize wrapper for observation normalization
        seed: Random seed for reproducible evaluation (uses seed+2000+episode for each episode)
        is_quadruped: Whether the environment is a quadruped velocity tracking env
        robot: Quadruped robot model name (only used when is_quadruped=True)
        simple_reward: If True, use simplified reward for quadruped eval envs
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    
    # Determine environment type
    is_shadow_hand = (domain == "shadow_hand")
    
    # For quadruped evaluation, use fixed x-velocity commands for the recorded videos
    # to systematically test the policy at different speeds
    quadruped_eval_velocities = [0.0, 0.5, 1.0]  # vx for each video
    
    for episode in range(num_episodes):
        # Create evaluation environment with rgb_array render mode for video recording
        if is_quadruped:
            eval_env_base = make_quadruped_env(
                robot=robot, render_mode="rgb_array",
                domain_rand_cfg=DomainRandomizationConfig.disabled(),
                simple_reward=simple_reward,
                use_go2_sysid=use_go2_sysid,
            )
        elif is_shadow_hand:
            eval_env_base = make_shadow_hand_env(task, render_mode="rgb_array")
        else:
            eval_env_base = make_dm_env(domain, task, render_mode="rgb_array")
        
        # Wrap in VecEnv for compatibility with model
        eval_env = DummyVecEnv([lambda: eval_env_base])
        
        # Seed the environment for reproducibility (different seed per episode)
        if seed is not None:
            eval_env.seed(seed + 2000 + episode)
            eval_env.action_space.seed(seed + 2000 + episode)
            eval_env.observation_space.seed(seed + 2000 + episode)
        
        # Apply normalization if available
        if normalize_env is not None:
            eval_env = VecNormalize.load(
                normalize_env, 
                eval_env
            )
            eval_env.training = False
            eval_env.norm_reward = False
        
        obs = eval_env.reset()
        
        # For quadruped video episodes, set fixed velocity commands
        # so each video tests a specific speed
        if is_quadruped and episode < len(quadruped_eval_velocities):
            vx = quadruped_eval_velocities[episode]
            # Unwrap through TimeLimit to reach QuadrupedVelocityTrackingEnv
            eval_env_base.unwrapped.set_commands(vx=vx, vy=0.0, wz=0.0)
            # Re-fetch obs so the command is reflected in the observation
            obs = eval_env.env_method("_get_obs")
            # _get_obs returns a dict per env; repack for VecEnv format
            obs = {k: np.array([obs[0][k]]) for k in obs[0]}
            # If VecNormalize is active, normalize the new observation
            if normalize_env is not None:
                obs = eval_env.normalize_obs(obs)
            print(f"  Quadruped eval episode {episode}: fixed vx={vx:.1f} m/s")
        
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
                if is_quadruped:
                    # Quadruped env supports render_mode="rgb_array" natively
                    frame = eval_env_base.render()
                elif domain == "walker":
                    # Use tracking camera for walker environments
                    frame = eval_env.unwrapped.envs[0].unwrapped._env.physics.render(camera_id='side', height=480, width=640)
                else:
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
            # Include velocity in filename for quadruped
            if is_quadruped and episode < len(quadruped_eval_velocities):
                vx = quadruped_eval_velocities[episode]
                video_path = video_dir / f"rollout{episode}_vx{vx:.1f}.mp4"
            else:
                video_path = video_dir / f"rollout{episode}.mp4"
            # Use 50 FPS for quadruped (matches control frequency), 30 FPS for others
            fps = 50 if is_quadruped else 30
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


def _parse_saved_checkpoint_step(model_zip_path: Path) -> Optional[int]:
    """Extract the labeled step from a checkpoint filename like model_300000_steps.zip."""
    stem = model_zip_path.stem
    prefix = "model_"
    suffix = "_steps"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None

    step_str = stem[len(prefix):-len(suffix)]
    try:
        return int(step_str)
    except ValueError:
        return None


def resolve_checkpoint_paths(checkpoint_dir: Path, checkpoint_step: int) -> tuple[Optional[int], Optional[Path], Optional[Path]]:
    """Resolve the requested checkpoint, falling back to the nearest saved step if needed."""
    model_path = checkpoint_dir / f"model_{checkpoint_step}_steps"
    model_zip_path = model_path.with_suffix(".zip")
    vecnormalize_path = checkpoint_dir / f"model_vecnormalize_{checkpoint_step}_steps.pkl"
    if (model_path.exists() or model_zip_path.exists()) and vecnormalize_path.exists():
        return checkpoint_step, model_path, vecnormalize_path

    candidate_steps = []
    for candidate_zip in checkpoint_dir.glob("model_*_steps.zip"):
        candidate_step = _parse_saved_checkpoint_step(candidate_zip)
        if candidate_step is None:
            continue
        candidate_vecnormalize = checkpoint_dir / f"model_vecnormalize_{candidate_step}_steps.pkl"
        if candidate_vecnormalize.exists():
            candidate_steps.append(candidate_step)

    if not candidate_steps:
        return None, None, None

    resolved_step = min(candidate_steps, key=lambda step: (abs(step - checkpoint_step), step))
    resolved_model_path = checkpoint_dir / f"model_{resolved_step}_steps"
    resolved_vecnormalize_path = checkpoint_dir / f"model_vecnormalize_{resolved_step}_steps.pkl"
    return resolved_step, resolved_model_path, resolved_vecnormalize_path


def evaluate_checkpoint_videos(logdir: Path, checkpoint_steps: list[int],
                               algorithm: str, domain: str, task: str,
                               num_episodes: int, num_videos: int, seed: int,
                               is_quadruped: bool = False, robot: str = "go2",
                               simple_reward: bool = False,
                               use_go2_sysid: bool = True):
    """Load requested checkpoints and record videos for each one."""
    checkpoint_dir = logdir / "checkpoints"

    for checkpoint_step in checkpoint_steps:
        resolved_step, model_path, vecnormalize_path = resolve_checkpoint_paths(
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=checkpoint_step,
        )
        checkpoint_video_dir = logdir / f"video_{checkpoint_step}"

        if resolved_step is None or model_path is None or vecnormalize_path is None:
            model_zip_path = (checkpoint_dir / f"model_{checkpoint_step}_steps").with_suffix(".zip")
            print(
                f"Skipping checkpoint eval at step {checkpoint_step}: "
                f"checkpoint model not found at {model_zip_path}"
            )
            continue

        if resolved_step != checkpoint_step:
            print(
                f"Requested checkpoint {checkpoint_step} not found exactly; "
                f"using nearest saved checkpoint {resolved_step} instead."
            )

        print(
            f"\nEvaluating checkpoint at step {resolved_step} "
            f"and recording videos to {checkpoint_video_dir}..."
        )
        checkpoint_model, checkpoint_env = load_saved_model_for_video_eval(
            algorithm=algorithm,
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            domain=domain,
            task=task,
            is_quadruped=is_quadruped,
            robot=robot,
            simple_reward=simple_reward,
            use_go2_sysid=use_go2_sysid,
        )

        try:
            evaluate_and_record(
                model=checkpoint_model,
                domain=domain,
                task=task,
                num_episodes=num_episodes,
                num_videos=num_videos,
                video_dir=checkpoint_video_dir,
                normalize_env=vecnormalize_path,
                seed=seed,
                is_quadruped=is_quadruped,
                robot=robot,
                simple_reward=simple_reward,
                use_go2_sysid=use_go2_sysid,
            )
        finally:
            checkpoint_env.close()


def main(argv):
    """
    Main training and evaluation function.
    """
    del argv # Not used since we're using absl for flags
    
    # ==================== SEED EVERYTHING FOR REPRODUCIBILITY ====================   
    # 1. NumPy's random number generator (used by callbacks and various operations)
    np.random.seed(_SEED.value)
    
    # 2. Python hash randomization (affects dict/set ordering)
    os.environ['PYTHONHASHSEED'] = str(_SEED.value)
    
    # 3. JAX deterministic operations (crucial for GPU reproducibility)
    os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
    
    print(f"=" * 60)
    print(f"SEEDING: All random number generators set to seed={_SEED.value}")
    print(f"=" * 60)
    # ============================================================================
    
    # Detect environment type
    is_shadow_hand = is_shadow_hand_env(_ENV_NAME.value)
    is_quadruped = is_quadruped_env(_ENV_NAME.value)
    
    # Parse environment name
    if is_quadruped:
        # Quadruped velocity tracking environment
        domain, task = parse_env_name(_ENV_NAME.value)
        env_name = _ENV_NAME.value
        print(f"Environment: Quadruped ({_ROBOT.value}) / {task}")
    elif is_shadow_hand:
        # Shadow hand environments use the full registered name
        env_name = _ENV_NAME.value
        domain = "shadow_hand"
        task = env_name  # Use full name as task for consistency
        print(f"Environment: Shadow Hand - {env_name}")
    elif _DOMAIN.value and _TASK.value:
        domain = _DOMAIN.value
        task = _TASK.value
        env_name = f"{domain}-{task}"
        print(f"Environment: {domain}/{task}")
    else:
        domain, task = parse_env_name(_ENV_NAME.value)
        env_name = _ENV_NAME.value
        print(f"Environment: {domain}/{task}")

    checkpoint_eval_steps = parse_checkpoint_eval_steps(_CHECKPOINT_EVALS.value)
    if checkpoint_eval_steps:
        print(f"Additional checkpoint video evals requested: {checkpoint_eval_steps}")
    
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
            inject_type=_INJECT_TYPE.value if _ALGORITHM.value in ["SAC-MPC", "TD3-MPC"] else None,
            percentage=_PERCENTAGE.value if _ALGORITHM.value in ["SAC-MPC", "TD3-MPC"] else None
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
        gradient_steps=_GRADIENT_STEPS.value,
        policy_delay=_POLICY_DELAY.value,
        seed=_SEED.value,
        tensorboard_log=tensorboard_log_path,
        inject_n_timesteps=_INJECT_N_TIMESTEPS.value,
        inject_type=_INJECT_TYPE.value,
        percentage=_PERCENTAGE.value,
        num_traj=_NUM_TRAJ.value,
        random_select=_RANDOM_SELECT.value,
        data_dir=_DATA_DIR.value,
        use_go2_sysid=_USE_GO2_SYSID.value,
    )
    
    # ── Domain randomization setup (quadruped only) ──────────────────────
    dr_cfg = None
    if is_quadruped:
        print(
            "Go2 sysID joint dynamics: "
            f"{'ENABLED' if _USE_GO2_SYSID.value else 'DISABLED'}"
        )
        dr_cfg = build_quadruped_domain_rand_config()
        if dr_cfg.enable:
            print(
                "Domain randomization: ENABLED "
                f"(config_type={_DOMAIN_RAND_CONFIG_TYPE.value})"
            )
            print(f"Resolved DR config: {dr_cfg.to_dict()}")
        else:
            print(
                "Domain randomization: DISABLED "
                f"(config_type={_DOMAIN_RAND_CONFIG_TYPE.value})"
            )
    # Eval environments never use DR (deterministic evaluation)
    dr_cfg_eval = DomainRandomizationConfig.disabled() if is_quadruped else None

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
            "checkpoint_evals": checkpoint_eval_steps,
            "save_replay_buffer_checkpoints": _SAVE_REPLAY_BUFFER_CHECKPOINTS.value,
            "save_replay_buffer_final": _SAVE_REPLAY_BUFFER_FINAL.value,
            "use_go2_sysid": _USE_GO2_SYSID.value,
        })
        if is_quadruped:
            config_dict["domain_randomization"] = {
                "enabled": dr_cfg.enable,
                "config_type": _DOMAIN_RAND_CONFIG_TYPE.value,
                "legacy_flag_enabled": _DOMAIN_RAND.value,
                "legacy_obs_noise_level": _DOMAIN_RAND_OBS_NOISE.value,
                "resolved_config": dr_cfg.to_dict(),
            }
        save_config(logdir, config_dict)

    # Use simplified reward for quadruped environments when training with MPC injection
    use_simple_reward = is_quadruped and _ALGORITHM.value in ["SAC-MPC", "TD3-MPC"]
    if use_simple_reward:
        print("Using simplified reward function (velocity tracking + termination only)")

    # Create training environment
    print(f"Creating {_NUM_ENVS.value} parallel environments...")
    if is_quadruped:
        robot_name = _ROBOT.value
        _dr = dr_cfg  # capture for lambda closure
        _sr = use_simple_reward  # capture for lambda closure
        vec_env = make_vec_env(
            lambda: make_quadruped_env(robot=robot_name, domain_rand_cfg=_dr,
                                      simple_reward=_sr,
                                      use_go2_sysid=_USE_GO2_SYSID.value),
            n_envs=_NUM_ENVS.value,
            seed=_SEED.value,
        )
    elif is_shadow_hand:
        vec_env = make_vec_env(
            lambda: make_shadow_hand_env(env_name),
            n_envs=_NUM_ENVS.value,
            seed=_SEED.value,
        )
    else:
        vec_env = make_vec_env(
            lambda: make_dm_env(domain, task), # lambda fxn so make_vec_env() can make multiple envs
            n_envs=_NUM_ENVS.value,
            seed=_SEED.value,
        )

    # VecNormalize standardizes observations and rewards to ~N(0,1), which is critical for
    # stable learning in continuous control (prevents different-scale features from dominating)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Reseed after VecNormalize wrapping to ensure deterministic environment behavior
    #vec_env.seed(_SEED.value)
    #vec_env.action_space.seed(_SEED.value)
    #vec_env.observation_space.seed(_SEED.value)
    
    # Path for the final model and normalization stats
    model_path = logdir / "final_model"
    vec_normalize_path = logdir / "vec_normalize.pkl"
    replay_buffer_path = logdir / "replay_buffer.pkl"
    
    # Load or create model
    # NOTE: SB3/SBX saves models with .zip extension but load() doesn't require it
    if _LOAD_RUN_NAME.value and (model_path.with_suffix('.zip').exists() or model_path.exists()):
        # Load existing model
        model = load_model(_ALGORITHM.value, model_path, vec_env, is_quadruped=is_quadruped)
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
            is_quadruped=is_quadruped,
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
            num_envs=_NUM_ENVS.value,
            is_quadruped=is_quadruped,
            robot=_ROBOT.value,
            save_replay_buffer_checkpoints=_SAVE_REPLAY_BUFFER_CHECKPOINTS.value,
            simple_reward=use_simple_reward,
            use_go2_sysid=_USE_GO2_SYSID.value,
            domain_rand_config_type=(
                _DOMAIN_RAND_CONFIG_TYPE.value if is_quadruped else "disabled"
            ),
        )
        
        # If using SAC-MPC with percentage injection, connect the callback to the model
        if (_ALGORITHM.value in ["SAC-MPC", "TD3-MPC"] and
            _INJECT_TYPE.value == "percentage" and
            mpc_inject_callback is not None):
            model.target_mpc_percentage = config.percentage
            model.mpc_inject_callback = mpc_inject_callback
            print(f"Connected MPC injection callback to {_ALGORITHM.value} (target: {config.percentage}%)")
        
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
            if _SAVE_REPLAY_BUFFER_FINAL.value and hasattr(model, 'save_replay_buffer'):
                model.save_replay_buffer(replay_buffer_path)
                print(f"Replay buffer saved (size: {model.replay_buffer.size()})")
            
            print("Model and normalization stats saved")
        
        # Close eval environment if it was created
        if eval_env is not None:
            eval_env.close()
    
    # Evaluation phase (only if logging enabled)
    if _ENABLE_LOGGING.value:
        if checkpoint_eval_steps:
            evaluate_checkpoint_videos(
                logdir=logdir,
                checkpoint_steps=checkpoint_eval_steps,
                algorithm=_ALGORITHM.value,
                domain=domain,
                task=task,
                num_episodes=_NUM_EVAL_EPISODES.value,
                num_videos=_NUM_VIDEOS.value,
                seed=_SEED.value,
                is_quadruped=is_quadruped,
                robot=_ROBOT.value,
                simple_reward=use_simple_reward,
                use_go2_sysid=_USE_GO2_SYSID.value,
            )

        print(f"\nEvaluating model for {_NUM_EVAL_EPISODES.value} episodes...")
        evaluate_and_record(
            model=model,
            domain=domain,
            task=task,
            num_episodes=_NUM_EVAL_EPISODES.value,
            num_videos=_NUM_VIDEOS.value,
            video_dir=video_dir,
            normalize_env=vec_normalize_path if vec_normalize_path.exists() else None,
            seed=_SEED.value,  # Pass seed for reproducible evaluation
            is_quadruped=is_quadruped,
            robot=_ROBOT.value,
            simple_reward=use_simple_reward,
            use_go2_sysid=_USE_GO2_SYSID.value,
        )
    
    vec_env.close()
    print("\nDone!")


if __name__ == "__main__":
    app.run(main)
