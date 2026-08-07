import datetime
import json
import os
import sys
from pathlib import Path
import warnings
import subprocess
from collections import Counter

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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import mediapy as media
import jax
import torch as th

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
from mpc_rl.common.quadruped_tensorboard_callback import failure_reason_metric_name
from mpc_rl.sac_mpc.sac_mpc import SAC_MPC
from mpc_rl.td3_mpc.td3_mpc import TD3_MPC
# SB3 (PyTorch) MPC-augmented algorithms for quadruped (supports asymmetric policies + Dict obs)
from mpc_rl.sac_mpc.sb3_sac_mpc import SB3_SAC_MPC
from mpc_rl.td3_mpc.sb3_td3_mpc import SB3_TD3_MPC

# Register custom quadruped velocity tracking environment
import mpc_rl.envs
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.go2_sysid import assert_go2_sysid_joint_dynamics
from mpc_rl.envs.cheetah3_env import DEFAULT_SPEED_GOAL as CHEETAH3_DEFAULT_SPEED_GOAL
from mpc_rl.envs.barrel_roll_common import (
    ACTION_SCALE as BARREL_ROLL_ACTION_SCALE,
    CONTROL_DT as BARREL_ROLL_CONTROL_DT,
    CONTROL_STEPS as BARREL_ROLL_CONTROL_STEPS,
    ROLL_START_TIME as BARREL_ROLL_START_TIME,
    SCHEMA_VERSION as BARREL_ROLL_SCHEMA_VERSION,
    SPREAD_RANGE as BARREL_ROLL_SPREAD_RANGE,
    SUCCESS_CONFIG as BARREL_ROLL_SUCCESS_CONFIG,
    TASK_ID as BARREL_ROLL_TASK_ID,
)
from mpc_rl.planner.barrel_roll_dataset import expected_effective_config, sha256_file

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
_CHEETAH3_SPEED_GOAL = flags.DEFINE_float(
    "cheetah3_speed_goal",
    CHEETAH3_DEFAULT_SPEED_GOAL,
    "Forward speed target in m/s for the cheetah3 reward.",
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
_QUADRUPED_MPC_REPLAY_MODE = flags.DEFINE_enum(
    "quadruped_mpc_replay_mode",
    "direct",
    ["direct", "torque_saved_pd", "torque_current_pd"],
    "Quadruped MPC data replay mode. 'direct' injects saved RL transitions "
    "when available. 'torque_saved_pd' forces torque replay and uses saved "
    "trajectory PD gains. 'torque_current_pd' forces torque replay but uses "
    "the current QuadrupedVelocityTrackingEnv PD gains for inverse-PD action "
    "conversion.",
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


# Reserved evaluation seeds are intentionally far outside the commissioned and
# planned generation ranges (0+, 100_000+, 200_000+, and 300_000+).  Dataset
# generation must continue to treat this range as reserved.
BARREL_ROLL_EVAL_SEEDS = tuple(range(1_000_000, 1_000_100))
_QUADRUPED_TASKS = ("velocity_tracking", "barrel_roll")


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
    quadruped_mpc_replay_mode: str
    use_go2_sysid: bool
    cheetah3_speed_goal: float


def validate_quadruped_task(task: str) -> str:
    """Return a supported quadruped task or fail instead of misrouting it."""
    if task not in _QUADRUPED_TASKS:
        valid = ", ".join(_QUADRUPED_TASKS)
        raise ValueError(f"Unknown quadruped task {task!r}; expected one of: {valid}")
    return task


def validate_barrel_roll_training_options(
    *,
    robot: str,
    algorithm: str,
    inject_type: str,
    percentage: int,
    replay_mode: str,
    domain_rand_enabled: bool,
    domain_rand_config_type: str,
    use_go2_sysid: bool,
    data_dir: str | None,
) -> None:
    """Reject options that would silently violate the frozen barrel task."""
    incompatible = []
    if robot.lower() != "go2":
        incompatible.append("robot must be 'go2'")
    if algorithm != "SAC-MPC":
        incompatible.append("algorithm must be 'SAC-MPC'")
    if inject_type != "percentage":
        incompatible.append("inject_type must be 'percentage'")
    if percentage != 25:
        incompatible.append("percentage must be 25")
    if replay_mode != "direct":
        incompatible.append("quadruped_mpc_replay_mode must be 'direct'")
    if domain_rand_enabled or domain_rand_config_type != "disabled":
        incompatible.append("domain randomization must be disabled")
    if not use_go2_sysid:
        incompatible.append("use_go2_sysid must be enabled")
    if not data_dir:
        incompatible.append(
            f"data_dir must point to validated schema-v{BARREL_ROLL_SCHEMA_VERSION} "
            "barrel data"
        )
    if incompatible:
        raise ValueError("Incompatible barrel-roll options: " + "; ".join(incompatible))


def barrel_roll_config_snapshot(data_dir: str, target_percentage: int) -> dict:
    """Return the complete frozen task/data/evaluation provenance for config.json."""
    frozen = expected_effective_config()
    return {
        **frozen,
        "reset": {
            "sampler": "symmetric_hip_spread",
            "spread_range_rad": list(BARREL_ROLL_SPREAD_RANGE),
        },
        "dataset": {
            "path": str(Path(data_dir)),
            "replay_mode": "direct",
            "schema_version": BARREL_ROLL_SCHEMA_VERSION,
            "target_mpc_percentage": target_percentage,
        },
        "evaluation": {
            "checkpoint_metric": "success_rate",
            "num_episodes": len(BARREL_ROLL_EVAL_SEEDS),
            "seeds": list(BARREL_ROLL_EVAL_SEEDS),
        },
    }


def _git_repository_snapshot(path: Path) -> dict:
    """Return the exact Git revision and tracked-dirty state for one repository."""
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"invalid Git revision for {path}: {commit!r}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status.strip()),
    }


def barrel_roll_run_provenance(data_dir: str) -> dict:
    """Validate and serialize G8 source and immutable-dataset provenance."""
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = Path(data_dir)
    summary_path = dataset_dir / "dataset_summary.json"
    checksum_path = dataset_dir / "checksums.sha256"
    if not summary_path.is_file() or not checksum_path.is_file():
        raise ValueError(
            f"barrel-roll production data must contain dataset_summary.json and "
            f"checksums.sha256: {dataset_dir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_checksum_hash = sha256_file(checksum_path)
    if summary.get("checksum_index_sha256") != actual_checksum_hash:
        raise ValueError(
            "barrel-roll checksum-index hash does not match dataset_summary.json"
        )
    aggregate_name = summary.get("aggregate_manifest")
    if not isinstance(aggregate_name, str) or not aggregate_name:
        raise ValueError("barrel-roll dataset summary is missing aggregate_manifest")
    aggregate_path = dataset_dir / aggregate_name
    if not aggregate_path.is_file():
        raise ValueError(f"barrel-roll aggregate manifest is missing: {aggregate_path}")
    actual_aggregate_hash = sha256_file(aggregate_path)
    if summary.get("aggregate_manifest_sha256") != actual_aggregate_hash:
        raise ValueError(
            "barrel-roll aggregate-manifest hash does not match dataset_summary.json"
        )

    return {
        "argv": list(sys.argv),
        "source": {
            "root": _git_repository_snapshot(repo_root),
            "mpx": _git_repository_snapshot(repo_root / "deps" / "mpx"),
            "primal_dual_ilqr": _git_repository_snapshot(
                repo_root / "deps" / "mpx" / "mpx" / "primal_dual_ilqr"
            ),
            "gym_quadruped": _git_repository_snapshot(
                repo_root / "deps" / "gym-quadruped"
            ),
            "mujoco_mpc": _git_repository_snapshot(repo_root / "deps" / "mujoco_mpc"),
        },
        "dataset": {
            "path": str(dataset_dir),
            "schema_version": summary.get("schema_version"),
            "file_count": summary.get("file_count"),
            "transition_count": summary.get("transition_count"),
            "effective_config_sha256": summary.get("effective_config_sha256"),
            "checksum_index": checksum_path.name,
            "checksum_index_sha256": actual_checksum_hash,
            "aggregate_manifest": aggregate_name,
            "aggregate_manifest_sha256": actual_aggregate_hash,
            "dataset_summary_sha256": sha256_file(summary_path),
        },
    }


def quadruped_video_filename(
    *, task: str, episode: int, episode_seed: int | None, velocity: float | None
) -> str:
    """Return a task-aware quadruped rollout filename."""
    if task == "barrel_roll":
        if episode_seed is None:
            raise ValueError("barrel-roll videos require an evaluation seed")
        return f"rollout{episode}_seed{episode_seed}.mp4"
    if velocity is not None:
        return f"rollout{episode}_vx{velocity:.1f}.mp4"
    return f"rollout{episode}.mp4"


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


def evaluate_barrel_roll_policy(model, eval_env, seeds=BARREL_ROLL_EVAL_SEEDS) -> dict:
    """Evaluate fixed seeds and retain per-episode G8 outcome diagnostics."""
    episode_rewards = []
    successes = []
    failure_reasons = Counter()
    episodes = []

    for seed in seeds:
        eval_env.seed(int(seed))
        obs = eval_env.reset()
        done = np.array([False])
        episode_reward = 0.0
        terminal_info = None
        control_step = 0
        touchdown_step = None
        stabilization_step = None
        post_start_contact_break = False
        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)
            episode_reward += float(reward[0])
            control_step += 1
            step_info = infos[0]
            if isinstance(step_info, dict):
                contacts = np.asarray(step_info.get("contact_state", []), dtype=bool)
                all_contacts = contacts.shape == (4,) and bool(np.all(contacts))
                if (
                    contacts.shape == (4,)
                    and control_step * BARREL_ROLL_CONTROL_DT >= BARREL_ROLL_START_TIME
                    and not all_contacts
                ):
                    post_start_contact_break = True
                if touchdown_step is None and post_start_contact_break and all_contacts:
                    touchdown_step = control_step
                stability_count = int(step_info.get("stability_count", 0))
                if (
                    stabilization_step is None
                    and touchdown_step is not None
                    and stability_count >= BARREL_ROLL_SUCCESS_CONFIG.stable_control_steps
                ):
                    stabilization_step = control_step
            if bool(done[0]):
                terminal_info = step_info

        if not isinstance(terminal_info, dict) or "is_success" not in terminal_info:
            raise RuntimeError(
                f"barrel-roll evaluation seed {seed} did not return terminal is_success"
            )
        terminal_values = {}
        for key in ("roll_progress", "roll_error"):
            if terminal_info.get(key) is None:
                raise RuntimeError(
                    f"barrel-roll evaluation seed {seed} did not return terminal {key}"
                )
            value = float(terminal_info[key])
            if not np.isfinite(value):
                raise RuntimeError(
                    f"barrel-roll evaluation seed {seed} returned non-finite {key}"
                )
            terminal_values[key] = value
        if not np.isfinite(episode_reward):
            raise RuntimeError(
                f"barrel-roll evaluation seed {seed} returned a non-finite episode return"
            )
        success = bool(terminal_info["is_success"])
        successes.append(success)
        episode_rewards.append(episode_reward)
        failure_reason = None
        if not success:
            failure_reason = str(terminal_info.get("failure_reason") or "unknown")
            failure_reasons[failure_reason] += 1
        episodes.append({
            "seed": int(seed),
            "success": success,
            "failure_reason": failure_reason,
            "return": episode_reward,
            "control_steps": control_step,
            "terminal_roll_progress": terminal_values["roll_progress"],
            "terminal_roll_error": terminal_values["roll_error"],
            "touchdown_step": touchdown_step,
            "touchdown_time_s": (
                touchdown_step * BARREL_ROLL_CONTROL_DT
                if touchdown_step is not None
                else None
            ),
            "stabilization_step": stabilization_step,
            "stabilization_time_s": (
                stabilization_step * BARREL_ROLL_CONTROL_DT
                if stabilization_step is not None
                else None
            ),
            "terminal_stability_count": int(terminal_info.get("stability_count", 0)),
        })

    terminal_errors = [episode["terminal_roll_error"] for episode in episodes]
    touchdown_times = [
        episode["touchdown_time_s"]
        for episode in episodes
        if episode["touchdown_time_s"] is not None
    ]
    stabilization_times = [
        episode["stabilization_time_s"]
        for episode in episodes
        if episode["stabilization_time_s"] is not None
    ]

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_return": float(np.mean(episode_rewards)),
        "min_return": float(np.min(episode_rewards)),
        "max_return": float(np.max(episode_rewards)),
        "episode_rewards": episode_rewards,
        "failure_reasons": dict(failure_reasons),
        "seeds": list(seeds),
        "episodes": episodes,
        "mean_terminal_roll_error": float(np.mean(terminal_errors)),
        "mean_abs_terminal_roll_error": float(np.mean(np.abs(terminal_errors))),
        "touchdown_count": len(touchdown_times),
        "mean_touchdown_time_s": (
            float(np.mean(touchdown_times)) if touchdown_times else None
        ),
        "stabilization_count": len(stabilization_times),
        "mean_stabilization_time_s": (
            float(np.mean(stabilization_times)) if stabilization_times else None
        ),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write one durable JSON artifact without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


class BarrelRollEvalCallback(BaseCallback):
    """Checkpoint barrel policies using fixed-seed success rate."""

    def __init__(
        self,
        eval_env,
        *,
        eval_freq: int,
        best_model_save_path: Path,
        seeds=BARREL_ROLL_EVAL_SEEDS,
        history_path: Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        if len(seeds) != 100 or len(set(seeds)) != 100:
            raise ValueError("barrel-roll evaluation requires 100 unique held-out seeds")
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.best_model_save_path = Path(best_model_save_path)
        self.history_path = (
            Path(history_path)
            if history_path is not None
            else self.best_model_save_path / "evaluation_history.jsonl"
        )
        self.seeds = tuple(int(seed) for seed in seeds)
        self.best_success_rate = -np.inf
        self.last_result = None

    def _init_callback(self) -> None:
        self.best_model_save_path.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _evaluate_and_record(self) -> None:
        sync_envs_normalization(self.training_env, self.eval_env)
        result = evaluate_barrel_roll_policy(self.model, self.eval_env, self.seeds)
        result = {**result, "timesteps": int(self.num_timesteps)}
        self.last_result = result
        with self.history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            history.flush()
        self.logger.record("eval/barrel_roll_success_rate", result["success_rate"])
        self.logger.record("eval/mean_reward", result["mean_reward"])
        metric_failure_reasons = Counter()
        for reason, count in result["failure_reasons"].items():
            metric_failure_reasons[failure_reason_metric_name(reason)] += count
        for reason, count in metric_failure_reasons.items():
            self.logger.record(f"eval/failure_reason/{reason}", float(count))

        # Mean reward is diagnostic only.  A model becomes best strictly when
        # held-out success rate improves.
        if result["success_rate"] > self.best_success_rate:
            self.best_success_rate = result["success_rate"]
            self.model.save(self.best_model_save_path / "best_model")
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(self.best_model_save_path / "vec_normalize.pkl")
            _atomic_write_json(
                self.best_model_save_path / "selection.json",
                {
                    "checkpoint_metric": "success_rate",
                    "success_rate": result["success_rate"],
                    "timesteps": int(self.num_timesteps),
                },
            )

    def _on_training_start(self) -> None:
        # G6 requires evidence that held-out success improves above the exact
        # untrained policy baseline, so evaluate before collecting a transition.
        self._evaluate_and_record()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        self._evaluate_and_record()
        return True


class BarrelRollPilotDiagnosticsCallback(BaseCallback):
    """Fail fast on non-finite G6 signals and write a durable audit summary."""

    _TRAIN_SCALARS = (
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
        "train/ent_coef_loss",
    )

    def __init__(self, summary_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.summary_path = Path(summary_path)
        self.rollout_batches_checked = 0
        self.environment_transitions_checked = 0
        self.q_batches_checked = 0
        self.training_snapshots_checked = 0
        self.last_checked_updates = 0
        self.ranges: dict[str, list[float]] = {}
        self.replay_percentages: list[float] = []
        self.roll_progresses: list[float] = []

    def _update_range(self, name: str, values) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return
        if not np.isfinite(array).all():
            self._fail(f"non-finite {name}")
        low = float(np.min(array))
        high = float(np.max(array))
        if name not in self.ranges:
            self.ranges[name] = [low, high]
        else:
            self.ranges[name][0] = min(self.ranges[name][0], low)
            self.ranges[name][1] = max(self.ranges[name][1], high)

    def _check_observations(self, observations, prefix: str = "observations") -> None:
        if isinstance(observations, dict):
            for key, value in observations.items():
                self._update_range(f"{prefix}/{key}", value)
        else:
            self._update_range(prefix, observations)

    def _check_q_values(self, observations, actions) -> None:
        observation_tensor, _ = self.model.policy.obs_to_tensor(observations)
        action_tensor = th.as_tensor(
            actions, dtype=th.float32, device=self.model.device
        )
        with th.no_grad():
            q_values = self.model.critic(observation_tensor, action_tensor)
        for index, q_value in enumerate(q_values):
            values = q_value.detach().cpu().numpy()
            self._update_range(f"q_value/{index}", values)
        self.q_batches_checked += 1
        q_min = min(bounds[0] for key, bounds in self.ranges.items() if key.startswith("q_value/"))
        q_max = max(bounds[1] for key, bounds in self.ranges.items() if key.startswith("q_value/"))
        self.logger.record("diagnostics/q_value_min", q_min)
        self.logger.record("diagnostics/q_value_max", q_max)

    def _check_training_scalars(self) -> None:
        updates = int(getattr(self.model, "_n_updates", 0))
        if updates <= self.last_checked_updates:
            return
        logged = getattr(self.logger, "name_to_value", {})
        missing = [name for name in self._TRAIN_SCALARS if name not in logged]
        if missing:
            self._fail(
                f"training update {updates} missing diagnostics: {', '.join(missing)}"
            )
        for name in self._TRAIN_SCALARS:
            self._update_range(name, [logged[name]])
        self.last_checked_updates = updates
        self.training_snapshots_checked += 1

    def _check_replay(self) -> None:
        replay_buffer = getattr(self.model, "replay_buffer", None)
        if replay_buffer is None or not hasattr(replay_buffer, "get_mpc_percentage"):
            return
        percentage = float(replay_buffer.get_mpc_percentage())
        self._update_range("replay_buffer/mpc_percentage", [percentage])
        if percentage < 0.0 or percentage > 100.0:
            self._fail(f"invalid MPC replay percentage {percentage}")
        self.replay_percentages.append(percentage)
        sources = getattr(replay_buffer, "transition_sources", None)
        if sources is not None:
            filled = replay_buffer.size()
            active_sources = sources if replay_buffer.full else sources[:filled]
            if active_sources.size and not np.isin(active_sources, (0, 1)).all():
                self._fail("replay buffer contains a source tag other than 0 or 1")

    def _summary(self, status: str, error: str | None = None) -> dict:
        return {
            "status": status,
            "error": error,
            "rollout_batches_checked": self.rollout_batches_checked,
            "environment_transitions_checked": self.environment_transitions_checked,
            "q_batches_checked": self.q_batches_checked,
            "training_snapshots_checked": self.training_snapshots_checked,
            "last_checked_updates": self.last_checked_updates,
            "ranges": self.ranges,
            "replay_percentage_last": (
                self.replay_percentages[-1] if self.replay_percentages else None
            ),
            "roll_progress_min": (
                min(self.roll_progresses) if self.roll_progresses else None
            ),
            "roll_progress_max": (
                max(self.roll_progresses) if self.roll_progresses else None
            ),
        }

    def _write_summary(self, status: str, error: str | None = None) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self._summary(status, error), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.summary_path)

    def _fail(self, error: str) -> None:
        self._write_summary("failed", error)
        raise FloatingPointError(error)

    def _on_step(self) -> bool:
        observations = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        if observations is None or actions is None or rewards is None:
            self._fail("rollout callback is missing observations, actions, or rewards")
        self._check_observations(observations)
        self._update_range("actions", actions)
        self._update_range("rewards", rewards)
        self._check_q_values(observations, actions)
        self._check_training_scalars()
        self._check_replay()
        infos = self.locals.get("infos") or []
        for info in infos:
            if isinstance(info, dict) and info.get("roll_progress") is not None:
                progress = float(info["roll_progress"])
                self._update_range("roll_progress", [progress])
                self.roll_progresses.append(progress)
        self.rollout_batches_checked += 1
        self.environment_transitions_checked += int(np.asarray(rewards).size)
        if self.n_calls % 100 == 0:
            self._write_summary("running")
        return True

    def _on_training_end(self) -> None:
        self._check_training_scalars()
        self._check_replay()
        self._write_summary("complete")


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


def is_cheetah3_env(env_name: str) -> bool:
    """Check if the environment is the local three-legged cheetah task."""
    return env_name.lower() in {"cheetah3", "cheetah3-run"} or env_name.lower().startswith("cheetah3-")


def make_cheetah3_env(render_mode=None, speed_goal: float = CHEETAH3_DEFAULT_SPEED_GOAL):
    """Create the local three-legged cheetah environment wrapped for SBX."""
    time_limit = _MAX_EPISODE_STEPS.value * 0.01
    gym_env = gym.make(
        "Cheetah3-v0",
        render_mode=render_mode,
        speed_goal=speed_goal,
        time_limit=time_limit,
    )
    gym_env = FlattenObservation(gym_env)
    return gym_env


def make_quadruped_env(robot: str = "go2", render_mode=None, domain_rand_cfg=None,
                       simple_reward: bool = False,
                       use_go2_sysid: bool = True,
                       task: str = "velocity_tracking"):
    """
    Create a supported quadruped gymnasium environment.
    
    Returns a Dict observation space for asymmetric actor-critic training:
        "policy" (45-dim): Real-hardware-available sensor observations (actor)
        "privileged": Simulation-only critic input (3D velocity tracking, 4D barrel roll)
    
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
        task: `velocity_tracking` (default) or `barrel_roll`.
    
    Returns:
        Task-specific quadruped gymnasium environment with Dict obs space
    """
    task = validate_quadruped_task(task)
    if task == "barrel_roll":
        if robot.lower() != "go2":
            raise ValueError("barrel-roll requires robot='go2'")
        if not use_go2_sysid:
            raise ValueError("barrel-roll requires use_go2_sysid=True")
        if domain_rand_cfg is not None and domain_rand_cfg.enable:
            raise ValueError("barrel-roll requires disabled domain randomization")
        if simple_reward:
            raise ValueError("barrel-roll requires its frozen task-specific reward")
        return gym.make(
            "QuadrupedBarrelRoll-v0",
            robot="go2",
            render_mode=render_mode,
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
            use_go2_sysid=True,
            action_scale=BARREL_ROLL_ACTION_SCALE,
        )

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
                                      is_cheetah3: bool = False,
                                      robot: str = "go2",
                                      simple_reward: bool = False,
                                      use_go2_sysid: bool = True,
                                      cheetah3_speed_goal: float = CHEETAH3_DEFAULT_SPEED_GOAL):
    """Create a single-env VecEnv for loading a saved model."""
    is_shadow_hand = (domain == "shadow_hand")

    if is_quadruped:
        return DummyVecEnv([
            lambda: make_quadruped_env(
                robot=robot,
                domain_rand_cfg=DomainRandomizationConfig.disabled(),
                simple_reward=simple_reward,
                use_go2_sysid=use_go2_sysid,
                task=task,
            )
        ])
    if is_cheetah3:
        return DummyVecEnv([
            lambda: make_cheetah3_env(speed_goal=cheetah3_speed_goal)
        ])
    if is_shadow_hand:
        return DummyVecEnv([lambda: make_shadow_hand_env(task)])
    return DummyVecEnv([lambda: make_dm_env(domain, task)])


def load_saved_model_for_video_eval(algorithm: str, model_path: Path,
                                    vecnormalize_path: Optional[Path],
                                    domain: str, task: str,
                                    is_quadruped: bool = False,
                                    is_cheetah3: bool = False,
                                    robot: str = "go2",
                                    simple_reward: bool = False,
                                    use_go2_sysid: bool = True,
                                    cheetah3_speed_goal: float = CHEETAH3_DEFAULT_SPEED_GOAL):
    """Load a saved model plus its VecNormalize stats for video evaluation."""
    model_env = make_single_env_for_model_loading(
        domain=domain,
        task=task,
        is_quadruped=is_quadruped,
        is_cheetah3=is_cheetah3,
        robot=robot,
        simple_reward=simple_reward,
        use_go2_sysid=use_go2_sysid,
        cheetah3_speed_goal=cheetah3_speed_goal,
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
        # Actor sees only the 45D policy vector. The critic dimensions are
        # derived from the task space (48D state for velocity, 49D for barrel).
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
                     is_cheetah3: bool = False,
                     save_replay_buffer_checkpoints: bool = False,
                     simple_reward: bool = False,
                     use_go2_sysid: bool = True,
                     domain_rand_config_type: str = "disabled",
                     cheetah3_speed_goal: float = CHEETAH3_DEFAULT_SPEED_GOAL):
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

    # Add rollout Tensorboard callback for quadruped off-policy training.
    if is_quadruped and cfg.algorithm in ["SAC", "TD3", "SAC-MPC", "TD3-MPC"]:
        callbacks.append(QuadrupedTensorboardCallback(log_freq=100, task=task))
        if task == "barrel_roll" and enable_logging:
            callbacks.append(
                BarrelRollPilotDiagnosticsCallback(
                    logdir / "barrel_roll_pilot_diagnostics.json"
                )
            )
    
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
                                          use_go2_sysid=use_go2_sysid,
                                          task=task),
                n_envs=1,
                seed=seed+1000,
            )
        elif is_cheetah3:
            eval_env = make_vec_env(
                lambda: make_cheetah3_env(speed_goal=cheetah3_speed_goal),
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
        if is_quadruped and task == "barrel_roll":
            eval_callback = BarrelRollEvalCallback(
                eval_env,
                eval_freq=max(eval_freq // num_envs, 1),
                best_model_save_path=logdir / "best_model",
                seeds=BARREL_ROLL_EVAL_SEEDS,
                history_path=logdir / "barrel_roll_eval_history.jsonl",
            )
        else:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(logdir / "best_model"),
                log_path=str(logdir / "eval_logs"),
                eval_freq=max(eval_freq // num_envs, 1),
                deterministic=True,
                render=False,
                n_eval_episodes=5,
            )
        callbacks.append(eval_callback)
    
    # Add MPC injection callback if using SAC-MPC or TD3-MPC
    if cfg.algorithm in ["SAC-MPC", "TD3-MPC"]:
        if cfg.inject_type == "fixed":
            print("\nSetting up FIXED MPC Injection from pre-generated trajectories...")
            inject_callback = FixedMPCInjectCallback(
                domain=domain,
                task=task,
                inject_every_n_timesteps=cfg.inject_n_timesteps,
                num_mpc_trajectories=cfg.num_traj,
                data_dir=cfg.data_dir,
                random_select=cfg.random_select,
                seed=seed,  # Pass seed for reproducible trajectory selection
                cheetah3_speed_goal=cheetah3_speed_goal,
                verbose=1,
            )
        elif cfg.inject_type == "percentage":
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
                expected_quadruped_task=(
                    BARREL_ROLL_TASK_ID
                    if is_quadruped and task == "barrel_roll"
                    else None
                ),
                expected_quadruped_schema_version=(
                    BARREL_ROLL_SCHEMA_VERSION
                    if is_quadruped and task == "barrel_roll"
                    else None
                ),
                quadruped_mpc_replay_mode=cfg.quadruped_mpc_replay_mode,
                cheetah3_speed_goal=cheetah3_speed_goal,
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
                        use_go2_sysid: bool = True,
                        is_cheetah3: bool = False,
                        cheetah3_speed_goal: float = CHEETAH3_DEFAULT_SPEED_GOAL,
                        barrel_roll_seeds: tuple[int, ...] | None = None,
                        barrel_roll_video_labels: dict[int, str] | None = None):
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
    episode_successes = []
    failure_reasons = Counter()
    
    # Determine environment type
    is_shadow_hand = (domain == "shadow_hand")
    
    # For quadruped evaluation, use fixed x-velocity commands for the recorded videos
    # to systematically test the policy at different speeds
    quadruped_eval_velocities = [0.0, 0.5, 1.0]  # vx for each video
    is_barrel_roll = is_quadruped and task == "barrel_roll"
    if is_barrel_roll:
        evaluation_seeds = (
            tuple(int(value) for value in barrel_roll_seeds)
            if barrel_roll_seeds is not None
            else BARREL_ROLL_EVAL_SEEDS
        )
        if not evaluation_seeds or len(evaluation_seeds) != len(set(evaluation_seeds)):
            raise ValueError("barrel-roll recording seeds must be non-empty and unique")
        num_episodes = len(evaluation_seeds)
        invalid_labels = set((barrel_roll_video_labels or {}).values()) - {
            "success", "failure"
        }
        if invalid_labels:
            raise ValueError(f"invalid barrel-roll video labels: {sorted(invalid_labels)}")
    
    for episode in range(num_episodes):
        # Create evaluation environment with rgb_array render mode for video recording
        if is_quadruped:
            eval_env_base = make_quadruped_env(
                robot=robot, render_mode="rgb_array",
                domain_rand_cfg=DomainRandomizationConfig.disabled(),
                simple_reward=simple_reward,
                use_go2_sysid=use_go2_sysid,
                task=task,
            )
        elif is_cheetah3:
            eval_env_base = make_cheetah3_env(
                render_mode="rgb_array",
                speed_goal=cheetah3_speed_goal,
            )
        elif is_shadow_hand:
            eval_env_base = make_shadow_hand_env(task, render_mode="rgb_array")
        else:
            eval_env_base = make_dm_env(domain, task, render_mode="rgb_array")
        
        # Wrap in VecEnv for compatibility with model
        eval_env = DummyVecEnv([lambda: eval_env_base])
        
        # Seed the environment for reproducibility (different seed per episode)
        episode_seed = (
            evaluation_seeds[episode]
            if is_barrel_roll
            else (seed + 2000 + episode if seed is not None else None)
        )
        if episode_seed is not None:
            eval_env.seed(episode_seed)
            eval_env.action_space.seed(episode_seed)
            eval_env.observation_space.seed(episode_seed)
        
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
        if is_quadruped and not is_barrel_roll and episode < len(quadruped_eval_velocities):
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
                elif is_cheetah3:
                    frame = eval_env_base.unwrapped._env.physics.render(
                        camera_id="side", height=480, width=640
                    )
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
        if is_barrel_roll:
            terminal_info = info[0]
            success = bool(terminal_info.get("is_success", False))
            episode_successes.append(success)
            if not success:
                failure_reasons[str(terminal_info.get("failure_reason") or "unknown")] += 1
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Save video
        if record_video and frames:
            # Include velocity in filename for quadruped
            if is_barrel_roll:
                outcome_label = (barrel_roll_video_labels or {}).get(int(episode_seed))
                if outcome_label is not None:
                    video_path = video_dir / f"{outcome_label}_seed{episode_seed}.mp4"
                else:
                    video_path = video_dir / quadruped_video_filename(
                        task=task,
                        episode=episode,
                        episode_seed=episode_seed,
                        velocity=None,
                    )
            elif is_quadruped and episode < len(quadruped_eval_velocities):
                vx = quadruped_eval_velocities[episode]
                video_path = video_dir / quadruped_video_filename(
                    task=task,
                    episode=episode,
                    episode_seed=episode_seed,
                    velocity=vx,
                )
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
    if is_barrel_roll:
        print(f"Success rate: {np.mean(episode_successes):.1%} over {len(episode_successes)} held-out seeds")
        print(f"Failure reasons: {dict(failure_reasons)}")
    print("="*50)
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
        "failure_reasons": dict(failure_reasons),
    }


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
                               use_go2_sysid: bool = True,
                               is_cheetah3: bool = False,
                               cheetah3_speed_goal: float = CHEETAH3_DEFAULT_SPEED_GOAL):
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
            is_cheetah3=is_cheetah3,
            robot=robot,
            simple_reward=simple_reward,
            use_go2_sysid=use_go2_sysid,
            cheetah3_speed_goal=cheetah3_speed_goal,
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
                is_cheetah3=is_cheetah3,
                cheetah3_speed_goal=cheetah3_speed_goal,
            )
        finally:
            checkpoint_env.close()


def evaluate_selected_barrel_roll_checkpoint(
    *,
    logdir: Path,
    algorithm: str,
    domain: str,
    task: str,
    robot: str,
    use_go2_sysid: bool,
) -> dict:
    """Evaluate the success-selected checkpoint and record outcome videos."""
    best_model_dir = Path(logdir) / "best_model"
    model_path = best_model_dir / "best_model"
    vecnormalize_path = best_model_dir / "vec_normalize.pkl"
    selection_path = best_model_dir / "selection.json"
    required_paths = (model_path.with_suffix(".zip"), vecnormalize_path, selection_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "selected barrel-roll checkpoint artifacts are missing: " + ", ".join(missing)
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    selected_model, selected_env = load_saved_model_for_video_eval(
        algorithm=algorithm,
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        domain=domain,
        task=task,
        is_quadruped=True,
        robot=robot,
        simple_reward=False,
        use_go2_sysid=use_go2_sysid,
    )
    try:
        evaluation = evaluate_barrel_roll_policy(
            selected_model,
            selected_env,
            BARREL_ROLL_EVAL_SEEDS,
        )
        report = {
            "selection": selection,
            "checkpoint_model": str(model_path.with_suffix(".zip")),
            "vecnormalize": str(vecnormalize_path),
            "evaluation": evaluation,
        }
        _atomic_write_json(best_model_dir / "evaluation.json", report)

        representative = {}
        for episode in evaluation["episodes"]:
            label = "success" if episode["success"] else "failure"
            representative.setdefault(label, int(episode["seed"]))
        video_seeds = tuple(representative.values())
        if video_seeds:
            labels_by_seed = {seed: label for label, seed in representative.items()}
            video_result = evaluate_and_record(
                model=selected_model,
                domain=domain,
                task=task,
                num_episodes=len(video_seeds),
                num_videos=len(video_seeds),
                video_dir=best_model_dir / "videos",
                normalize_env=vecnormalize_path,
                seed=None,
                is_quadruped=True,
                robot=robot,
                simple_reward=False,
                use_go2_sysid=use_go2_sysid,
                barrel_roll_seeds=video_seeds,
                barrel_roll_video_labels=labels_by_seed,
            )
            for seed, observed_success in zip(
                video_seeds, video_result["episode_successes"], strict=True
            ):
                expected_success = labels_by_seed[seed] == "success"
                if bool(observed_success) != expected_success:
                    raise RuntimeError(
                        f"representative video outcome changed for seed {seed}: "
                        f"expected {expected_success}, got {bool(observed_success)}"
                    )
                video_path = best_model_dir / "videos" / (
                    f"{labels_by_seed[seed]}_seed{seed}.mp4"
                )
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise RuntimeError(f"representative video was not written: {video_path}")
        report["representative_video_seeds"] = representative
        _atomic_write_json(best_model_dir / "evaluation.json", report)
        return report
    finally:
        selected_env.close()


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
    is_cheetah3 = is_cheetah3_env(_ENV_NAME.value)
    
    # Parse environment name
    if is_quadruped:
        domain, task = parse_env_name(_ENV_NAME.value)
        task = validate_quadruped_task(task)
        env_name = _ENV_NAME.value
        print(f"Environment: Quadruped ({_ROBOT.value}) / {task}")
    elif is_cheetah3:
        env_name = "cheetah3-run"
        domain = "cheetah3"
        task = "run"
        print(f"Environment: Three-Legged Cheetah / run")
        print(f"Cheetah3 speed goal: {_CHEETAH3_SPEED_GOAL.value:.3f} m/s")
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

    is_barrel_roll = is_quadruped and task == "barrel_roll"
    if is_barrel_roll:
        validate_barrel_roll_training_options(
            robot=_ROBOT.value,
            algorithm=_ALGORITHM.value,
            inject_type=_INJECT_TYPE.value,
            percentage=_PERCENTAGE.value,
            replay_mode=_QUADRUPED_MPC_REPLAY_MODE.value,
            domain_rand_enabled=_DOMAIN_RAND.value,
            domain_rand_config_type=_DOMAIN_RAND_CONFIG_TYPE.value,
            use_go2_sysid=_USE_GO2_SYSID.value,
            data_dir=_DATA_DIR.value,
        )
        if (
            flags.FLAGS["max_episode_steps"].present
            and _MAX_EPISODE_STEPS.value != BARREL_ROLL_CONTROL_STEPS
        ):
            raise ValueError(
                f"barrel-roll requires max_episode_steps={BARREL_ROLL_CONTROL_STEPS}"
            )

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
        quadruped_mpc_replay_mode=_QUADRUPED_MPC_REPLAY_MODE.value,
        use_go2_sysid=_USE_GO2_SYSID.value,
        cheetah3_speed_goal=_CHEETAH3_SPEED_GOAL.value,
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
            "cheetah3_speed_goal": _CHEETAH3_SPEED_GOAL.value,
        })
        if is_quadruped:
            config_dict["domain_randomization"] = {
                "enabled": dr_cfg.enable,
                "config_type": _DOMAIN_RAND_CONFIG_TYPE.value,
                "legacy_flag_enabled": _DOMAIN_RAND.value,
                "legacy_obs_noise_level": _DOMAIN_RAND_OBS_NOISE.value,
                "resolved_config": dr_cfg.to_dict(),
            }
        if is_barrel_roll:
            barrel_roll_snapshot = barrel_roll_config_snapshot(
                _DATA_DIR.value,
                _PERCENTAGE.value,
            )
            barrel_roll_snapshot["run_provenance"] = barrel_roll_run_provenance(
                _DATA_DIR.value
            )
            config_dict["barrel_roll"] = barrel_roll_snapshot
        save_config(logdir, config_dict)

    # Use simplified reward for quadruped environments when training with MPC injection
    use_simple_reward = (
        is_quadruped
        and not is_barrel_roll
        and _ALGORITHM.value in ["SAC-MPC", "TD3-MPC"]
    )
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
                                      use_go2_sysid=_USE_GO2_SYSID.value,
                                      task=task),
            n_envs=_NUM_ENVS.value,
            seed=_SEED.value,
        )
    elif is_cheetah3:
        speed_goal = _CHEETAH3_SPEED_GOAL.value
        vec_env = make_vec_env(
            lambda: make_cheetah3_env(speed_goal=speed_goal),
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
            is_cheetah3=is_cheetah3,
            save_replay_buffer_checkpoints=_SAVE_REPLAY_BUFFER_CHECKPOINTS.value,
            simple_reward=use_simple_reward,
            use_go2_sysid=_USE_GO2_SYSID.value,
            domain_rand_config_type=(
                _DOMAIN_RAND_CONFIG_TYPE.value if is_quadruped else "disabled"
            ),
            cheetah3_speed_goal=_CHEETAH3_SPEED_GOAL.value,
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

        if is_barrel_roll and _ENABLE_LOGGING.value:
            selected_report = evaluate_selected_barrel_roll_checkpoint(
                logdir=logdir,
                algorithm=_ALGORITHM.value,
                domain=domain,
                task=task,
                robot=_ROBOT.value,
                use_go2_sysid=_USE_GO2_SYSID.value,
            )
            print(
                "Selected checkpoint success rate: "
                f"{selected_report['evaluation']['success_rate']:.1%}"
            )
    
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
                is_cheetah3=is_cheetah3,
                cheetah3_speed_goal=_CHEETAH3_SPEED_GOAL.value,
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
            is_cheetah3=is_cheetah3,
            cheetah3_speed_goal=_CHEETAH3_SPEED_GOAL.value,
        )
    
    vec_env.close()
    print("\nDone!")


if __name__ == "__main__":
    app.run(main)
