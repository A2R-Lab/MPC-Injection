#!/usr/bin/env python3
"""
Script to plot the state distributions of a trained SAC-MPC walker model.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize