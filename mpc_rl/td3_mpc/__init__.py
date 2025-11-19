"""
TD3 with MPC-augmented replay buffer.
"""

from .td3_mpc import TD3_MPC
from .policies import TD3_MPCPolicy

__all__ = [
    'TD3_MPC',
    'TD3_MPCPolicy'
]