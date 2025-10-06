from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import (
    BaseJaxPolicy,
    SimbaSquashedGaussianActor,
    SimbaVectorCritic,
    SquashedGaussianActor,
    VectorCritic,
)
from sbx.common.type_aliases import RLTrainState

"""
NOTE: This file is derived from SBX's implementation of SAC. We are modifying it to be able to
inject MPC trajectories into the replay buffer during training to evaluate the effect on learning
since MPC is essentially solving an approximate solution to the MDP.
"""