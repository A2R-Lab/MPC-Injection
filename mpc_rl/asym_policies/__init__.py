"""Custom policy classes for asymmetric actor-critic training.

Provides SB3-based (PyTorch) SAC and TD3 policies where the actor only sees
real-hardware-available observations ("policy" key) while the critic also
receives privileged simulation ground truth ("privileged" key).

This enables sim2real transfer without teacher-student distillation:
the deployed actor policy is identical to the trained one, with zero
observation gap.
"""

from mpc_rl.asym_policies.asymmetric_policy import (
    AsymmetricSACPolicy,
    AsymmetricTD3Policy,
    CriticFeaturesExtractor,
    PolicyFeaturesExtractor,
)

__all__ = [
    "AsymmetricSACPolicy",
    "AsymmetricTD3Policy",
    "PolicyFeaturesExtractor",
    "CriticFeaturesExtractor",
]
