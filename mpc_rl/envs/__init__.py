"""Quadruped RL environments for velocity tracking.

This package provides gymnasium environments for training quadruped robots
to track user-commanded velocities using only real-hardware-available sensors.
"""

import gymnasium as gym

from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.envs.barrel_roll_env import QuadrupedBarrelRollEnv
from mpc_rl.envs.cheetah3_env import Cheetah3Env
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig

gym.register(
    id="QuadrupedVelocityTracking-v0",
    entry_point="mpc_rl.envs.velocity_tracking_env:QuadrupedVelocityTrackingEnv",
    max_episode_steps=1000,
)

gym.register(
    id="QuadrupedBarrelRoll-v0",
    entry_point="mpc_rl.envs.barrel_roll_env:QuadrupedBarrelRollEnv",
    max_episode_steps=50,
)

gym.register(
    id="Cheetah3-v0",
    entry_point="mpc_rl.envs.cheetah3_env:Cheetah3Env",
    max_episode_steps=1000,
)
