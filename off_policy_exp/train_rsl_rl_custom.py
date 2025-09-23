"""
Test script to train a PPO agent using RSL-RL for a specified mujoco playground environment.
"""

import os

from datetime import datetime
import json

from typing import Optional
from absl import app
from absl import flags
from absl import logging
import jax
import mediapy as media
from ml_collections import config_dict
import mujoco
from rsl_rl.runners import OnPolicyRunner
import torch
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

# Configure XLA compiler for optimized GPU operations
xla_flags = os.environ.get("XLA_FLAGS", "")  # Get existing XLA flags
xla_flags += " --xla_gpu_triton_gemm_any=True"  # Enable Triton GPU kernels for matrix ops
os.environ["XLA_FLAGS"] = xla_flags  # Apply the updated flags
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for headless GPU rendering

# Suppress info logs if you want
logging.set_verbosity(logging.WARNING)

# Define flags similar to the JAX script
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "CartpoleSwingup",
    (
        "Name of the environment. One of: "
        f"{', '.join(registry.ALL_ENVS)}"
    ),
)
_LOAD_RUN_NAME = flags.DEFINE_string(
    "load_run_name", None, "Run name to load from (for checkpoint restoration)."
)
_CHECKPOINT_NUM = flags.DEFINE_integer(
    "checkpoint_num", -1, "Checkpoint number to load from."
)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train."
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode).",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name.")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 4096, "Number of parallel envs.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training.")
_MULTI_GPU = flags.DEFINE_boolean(
    "multi_gpu", False, "If true, use multi-GPU training (distributed)."
)
_CAMERA = flags.DEFINE_string(
    "camera", None, "Camera name to use for rendering."
)

def rsl_rl_custom_config(
        env_name: str, unused_impl: Optional[str] = None
        ) -> config_dict.ConfigDict:
    """Returns tuned RSL-RL PPO config for the given environment."""

    rl_config = config_dict.create(
        seed=1,
        runner_class_name="OnPolicyRunner",
        policy=config_dict.create(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            activation="elu",
            class_name="ActorCritic",
        ),
        algorithm=config_dict.create(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            # mini batch size = num_envs*nsteps / nminibatches
            num_mini_batches=4,
            learning_rate=3.0e-4,  # 5.e-4
            schedule="fixed",  # could be adaptive, fixed
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        num_steps_per_env=24,  # per iteration
        max_iterations=100000,  # number of policy updates
        empirical_normalization=True,
        # logging
        save_interval=50,  # check for potential saves every this many iterations
        experiment_name="test",
        run_name="",
        # load and resume
        resume=False,
        load_run="-1",  # -1 = last run
        checkpoint=-1,  # -1 = last saved model
        resume_path=None,  # updated from load_run and chkpt
    )

    if env_name in (
        "Go1Getup",
        "BerkeleyHumanoidJoystickFlatTerrain",
        "G1Joystick",
        "Go1JoystickFlatTerrain",
        "CartpoleSwingup"
    ):
        rl_config.max_iterations = 1000
    if env_name == "Go1JoystickFlatTerrain":
        rl_config.algorithm.learning_rate = 3e-4
        rl_config.algorithm.schedule = "fixed"

    return rl_config


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    """Get default RL config for a given environment."""
    # if env_name in registry.manipulation._envs:
    #    return manipulation_params.rsl_rl_config(env_name)
    # elif env_name in registry.locomotion._envs:
    #    return locomotion_params.rsl_rl_config(env_name)
    # elif env_name in rsl_rl_custom_config(env_name):
    #    return rsl_rl_custom_config(env_name)
    # else:
    #    raise ValueError(f"No RL config for environment: {env_name}")
    return rsl_rl_custom_config(env_name)


def main(argv):
    """Run training & eval for the specified environment using RSL-RL."""
    del argv  # unused

    # Possibly parse the device for multi-GPU
    if _MULTI_GPU.value:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_rank = local_rank
        device = f"cuda:{local_rank}"
        print(f"Using multi-GPU: local_rank={local_rank}, device={device}")
    else:
        device = _DEVICE.value
        device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

    # If play-only, use fewer envs
    num_envs = 1 if _PLAY_ONLY.value else _NUM_ENVS.value

    # Load default config from registry
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    print(f"Environment config:\n{env_cfg}")

    # Generate unique experiment name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-{timestamp}"
    if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}")

    # Logging directory
    logdir = os.path.abspath(os.path.join("logs", exp_name))
    os.makedirs(logdir, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Checkpoint directory
    ckpt_path = os.path.join(logdir, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Initialize Weights & Biases if required
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        wandb.tensorboard.patch(root_logdir=logdir)
        wandb.init(project="mjxrl", name=exp_name)
        wandb.config.update(env_cfg.to_dict())
        wandb.config.update({"env_name": _ENV_NAME.value})

    # Save environment config to JSON
    with open(
        os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8"
    ) as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    # Domain randomization
    randomizer = registry.get_domain_randomizer(_ENV_NAME.value)

    # We'll store environment states during rendering
    render_trajectory = []

    # Callback to gather states for rendering
    def render_callback(_, state):
        render_trajectory.append(state)

    # Create the environment
    raw_env = registry.load(_ENV_NAME.value, config=env_cfg)
    brax_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_envs,
        _SEED.value,
        env_cfg.episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )

    # Build RSL-RL config
    train_cfg = get_rl_config(_ENV_NAME.value)

    obs_size = raw_env.observation_size
    if isinstance(obs_size, dict):
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
    else:
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}

    # Overwrite default config with flags
    train_cfg.seed = _SEED.value
    train_cfg.run_name = exp_name
    train_cfg.resume = _LOAD_RUN_NAME.value is not None
    train_cfg.load_run = _LOAD_RUN_NAME.value if _LOAD_RUN_NAME.value else "-1"
    train_cfg.checkpoint = _CHECKPOINT_NUM.value

    train_cfg_dict = train_cfg.to_dict()
    runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)

    # If resume, load from checkpoint
    if train_cfg.resume:
        resume_path = wrapper_torch.get_load_path(
            os.path.abspath("logs"),
            load_run=train_cfg.load_run,
            checkpoint=train_cfg.checkpoint,
        )
        print(f"Loading model from checkpoint: {resume_path}")
        runner.load(resume_path)

    if not _PLAY_ONLY.value:
        # Perform training
        runner.learn(
            num_learning_iterations=train_cfg.max_iterations,
            init_at_random_ep_len=False,
        )
        print("Done training.")
        return

    # If just playing (no training)
    policy = runner.get_inference_policy(device=device)

    # Example: run a single rollout
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(_SEED.value)
    state = jit_reset(rng)
    rollout = [state]

    # We’ll assume your environment’s observation is in state.obs["state"].
    obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])

    for _ in range(env_cfg.episode_length):
        with torch.no_grad():
            actions = policy(obs_torch)
        # Step environment
        state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
        rollout.append(state)
        obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])
        if state.done:
            break

    # Render
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    render_every = 2
    # If your environment is wrapped multiple times, adjust as needed:
    base_env = eval_env  # or brax_env.env.env.env
    fps = 1.0 / base_env.dt / render_every
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj,
        camera=_CAMERA.value,
        height=480,
        width=640,
        scene_option=scene_option,
    )
    media.write_video("rollout.mp4", frames, fps=fps)
    print("Rollout video saved as 'rollout.mp4'.")


if __name__ == "__main__":
    app.run(main)