from functools import partial
from typing import Any, ClassVar, Literal, Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax.typing import ArrayLike
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.sac.policies import SACPolicy, SimbaSACPolicy

"""
NOTE: This file is derived from SBX's implementation of SAC. We are modifying it to be able to
inject MPC trajectories into the replay buffer during training to evaluate the effect on learning
since MPC is essentially solving an approximate solution to the MDP.
"""