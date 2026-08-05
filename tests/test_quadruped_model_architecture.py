"""Regression tests for quadruped model architectures built via train.py.

Run:
    conda run -n mpc-rl python -m pytest tests/test_quadruped_model_architecture.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from absl import flags
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure local imports resolve when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.action_interfaces import (
    DEFAULT_ACTION_INTERFACE_ID,
    MPX_BOUND_ACTION_INTERFACE_ID,
)
from mpc_rl.envs.go2_sysid import assert_go2_sysid_joint_dynamics
from mpc_rl.common import QuadrupedTensorboardCallback
from mpc_rl.train import AllConfig, create_callbacks, create_model, make_quadruped_env

FLAGS = flags.FLAGS


if not FLAGS.is_parsed():
    FLAGS(["test_quadruped_model_architecture"])


def _build_cfg(algorithm: str) -> AllConfig:
    return AllConfig(
        algorithm=algorithm,
        learning_rate=3e-4,
        buffer_size=1_000,
        learning_starts=10,
        batch_size=32,
        tau=0.005,
        gamma=0.99,
        gradient_steps=1,
        policy_delay=2,
        seed=1,
        tensorboard_log="",
        inject_n_timesteps=5_000,
        inject_type="percentage",
        percentage=25,
        num_traj=10,
        random_select=True,
        data_dir="",
        quadruped_mpc_replay_mode="direct",
        quadruped_action_interface=DEFAULT_ACTION_INTERFACE_ID,
        use_go2_sysid=True,
        cheetah3_speed_goal=3.0,
    )


@pytest.fixture
def quadruped_vec_env():
    env = DummyVecEnv([
        lambda: make_quadruped_env(
            robot="go2",
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
            simple_reward=False,
        )
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    yield env
    env.close()


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def test_quadruped_sac_mpc_uses_512_hidden_layers(quadruped_vec_env):
    model = create_model(quadruped_vec_env, _build_cfg("SAC-MPC"), is_quadruped=True)

    assert model.policy_kwargs["net_arch"] == [512, 512]

    actor_layers = _linear_layers(model.policy.actor.latent_pi)
    assert [layer.in_features for layer in actor_layers] == [45, 512]
    assert [layer.out_features for layer in actor_layers] == [512, 512]
    assert model.policy.actor.mu.in_features == 512
    assert model.policy.actor.mu.out_features == 12

    critic_q0_layers = _linear_layers(model.policy.critic.qf0)
    assert [layer.in_features for layer in critic_q0_layers] == [60, 512, 512]
    assert [layer.out_features for layer in critic_q0_layers] == [512, 512, 1]


def test_quadruped_sac_keeps_default_hidden_layers(quadruped_vec_env):
    model = create_model(quadruped_vec_env, _build_cfg("SAC"), is_quadruped=True)

    assert model.policy_kwargs.get("net_arch") is None

    actor_layers = _linear_layers(model.policy.actor.latent_pi)
    assert [layer.in_features for layer in actor_layers] == [45, 256]
    assert [layer.out_features for layer in actor_layers] == [256, 256]
    assert model.policy.actor.mu.in_features == 256
    assert model.policy.actor.mu.out_features == 12


def test_quadruped_td3_mpc_keeps_existing_architecture(quadruped_vec_env):
    model = create_model(quadruped_vec_env, _build_cfg("TD3-MPC"), is_quadruped=True)

    assert model.policy_kwargs.get("net_arch") is None

    actor_layers = _linear_layers(model.policy.actor.mu)
    assert [layer.in_features for layer in actor_layers] == [45, 400, 300]
    assert [layer.out_features for layer in actor_layers] == [400, 300, 12]


def test_make_quadruped_env_go2_uses_sysid_joint_dynamics():
    env = make_quadruped_env(
        robot="go2",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        simple_reward=False,
    )
    try:
        assert_go2_sysid_joint_dynamics(env.unwrapped.mjModel)
    finally:
        env.close()


def test_make_quadruped_env_go2_can_disable_sysid_joint_dynamics():
    env = make_quadruped_env(
        robot="go2",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        simple_reward=False,
        use_go2_sysid=False,
    )
    try:
        with pytest.raises(AssertionError, match="Go2 sysID joint dynamics"):
            assert_go2_sysid_joint_dynamics(env.unwrapped.mjModel)
    finally:
        env.close()


def test_make_quadruped_env_uses_selected_mpx_bound_action_interface():
    env = make_quadruped_env(
        robot="go2",
        domain_rand_cfg=DomainRandomizationConfig.disabled(),
        simple_reward=True,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )
    try:
        assert env.unwrapped.action_scale == 1.0
        assert env.unwrapped.action_lpf_cutoff_hz is None
        assert env.unwrapped.action_lpf_alpha == 1.0
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["SAC", "TD3"])
def test_quadruped_pure_off_policy_uses_tensorboard_callback(tmp_path, algorithm):
    callbacks, eval_env, inject_callback = create_callbacks(
        cfg=_build_cfg(algorithm),
        enable_logging=False,
        logdir=tmp_path,
        domain="quadruped",
        task="velocity_tracking",
        seed=1,
        checkpoint_freq=25_000,
        eval_freq=10_000,
        num_envs=1,
        is_quadruped=True,
        robot="go2",
        save_replay_buffer_checkpoints=False,
        simple_reward=False,
        use_go2_sysid=True,
        domain_rand_config_type="disabled",
        cheetah3_speed_goal=3.0,
    )

    assert any(isinstance(callback, QuadrupedTensorboardCallback) for callback in callbacks)
    assert eval_env is None
    assert inject_callback is None
