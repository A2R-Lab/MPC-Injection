"""Asymmetric actor-critic policies for sim2real quadruped training.

Implements custom SB3 (PyTorch) SAC and TD3 policies where the actor and critic
receive different observations:

    - **Actor** (deployed on real hardware): Only "policy" observations (45-dim)
      consisting of IMU data, joint encoders, commands, and previous actions.
    - **Critic** (training only, in simulation): Both "policy" AND "privileged"
      observations (48-dim total), where privileged includes ground-truth base
      linear velocity from the simulator.

This asymmetric design:
    1. Gives the critic better value estimates (it sees the velocity that drives
       the reward signal), leading to more accurate policy gradients.
    2. Ensures the actor never learns to depend on privileged information,
       enabling zero-gap sim2real transfer without teacher-student distillation.

Usage with SB3:
    from stable_baselines3 import SAC, TD3
    from mpc_rl.policies.asymmetric_policy import AsymmetricSACPolicy, AsymmetricTD3Policy

    model = SAC(AsymmetricSACPolicy, env, verbose=1)
    model = TD3(AsymmetricTD3Policy, env, verbose=1)

References:
    - IsaacLab: Separate "policy" and "critic" observation groups
    - Pinto et al. 2017: "Asymmetric Actor Critic for Image-Based Robot Learning"
    - Kumar et al. 2021: "RMA: Rapid Motor Adaptation for Legged Robots"
"""

from __future__ import annotations

from typing import Any, Optional

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractors
# ═══════════════════════════════════════════════════════════════════════════════


class PolicyFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for the **actor** network.

    Only extracts the "policy" key from the Dict observation space,
    providing the actor with real-hardware-available observations only.

    Input:  Dict{"policy": Tensor(batch, 45), "privileged": Tensor(batch, 3)}
    Output: Tensor(batch, 45)
    """

    def __init__(self, observation_space: spaces.Dict):
        assert isinstance(observation_space, spaces.Dict), (
            f"PolicyFeaturesExtractor requires Dict observation space, got {type(observation_space)}"
        )
        assert "policy" in observation_space.spaces, (
            "Dict observation space must contain 'policy' key"
        )
        policy_space = observation_space["policy"]
        features_dim = int(policy_space.shape[0])
        super().__init__(observation_space, features_dim)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        return observations["policy"]


class CriticFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for the **critic** network.

    Extracts and concatenates both "policy" and "privileged" keys from the
    Dict observation space, giving the critic full information for accurate
    value estimation.

    Input:  Dict{"policy": Tensor(batch, 45), "privileged": Tensor(batch, 3)}
    Output: Tensor(batch, 48)
    """

    def __init__(self, observation_space: spaces.Dict):
        assert isinstance(observation_space, spaces.Dict), (
            f"CriticFeaturesExtractor requires Dict observation space, got {type(observation_space)}"
        )
        assert "policy" in observation_space.spaces, (
            "Dict observation space must contain 'policy' key"
        )
        assert "privileged" in observation_space.spaces, (
            "Dict observation space must contain 'privileged' key"
        )
        policy_dim = int(observation_space["policy"].shape[0])
        privileged_dim = int(observation_space["privileged"].shape[0])
        features_dim = policy_dim + privileged_dim
        super().__init__(observation_space, features_dim)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        return th.cat([observations["policy"], observations["privileged"]], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Asymmetric SAC Policy
# ═══════════════════════════════════════════════════════════════════════════════


class AsymmetricSACPolicy(SACPolicy):
    """SAC policy with asymmetric actor-critic observations.

    The actor receives only "policy" observations (45-dim, real-hardware sensors).
    The critic receives both "policy" + "privileged" observations (48-dim total).

    This is achieved by overriding make_actor() and make_critic() to provide
    different feature extractors while reusing SB3's standard Actor and
    ContinuousCritic classes unchanged.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule,
        **kwargs: Any,
    ):
        # Force separate feature extractors (no sharing between actor and critic)
        kwargs["share_features_extractor"] = False
        # Set default features_extractor_class to CriticFeaturesExtractor
        # This is used as fallback by make_features_extractor() if called directly
        kwargs.setdefault("features_extractor_class", CriticFeaturesExtractor)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ):
        """Create actor with PolicyFeaturesExtractor (only policy observations)."""
        if features_extractor is None:
            features_extractor = PolicyFeaturesExtractor(self.observation_space)
        return super().make_actor(features_extractor=features_extractor)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ):
        """Create critic with CriticFeaturesExtractor (all observations)."""
        if features_extractor is None:
            features_extractor = CriticFeaturesExtractor(self.observation_space)
        return super().make_critic(features_extractor=features_extractor)


# ═══════════════════════════════════════════════════════════════════════════════
# Asymmetric TD3 Policy
# ═══════════════════════════════════════════════════════════════════════════════


class AsymmetricTD3Policy(TD3Policy):
    """TD3 policy with asymmetric actor-critic observations.

    The actor (and actor_target) receive only "policy" observations (45-dim).
    The critic (and critic_target) receive "policy" + "privileged" (48-dim).

    Same approach as AsymmetricSACPolicy, adapted for TD3's architecture
    (which includes actor_target in addition to critic_target).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule,
        **kwargs: Any,
    ):
        kwargs["share_features_extractor"] = False
        kwargs.setdefault("features_extractor_class", CriticFeaturesExtractor)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ):
        """Create actor with PolicyFeaturesExtractor (only policy observations)."""
        if features_extractor is None:
            features_extractor = PolicyFeaturesExtractor(self.observation_space)
        return super().make_actor(features_extractor=features_extractor)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ):
        """Create critic with CriticFeaturesExtractor (all observations)."""
        if features_extractor is None:
            features_extractor = CriticFeaturesExtractor(self.observation_space)
        return super().make_critic(features_extractor=features_extractor)
