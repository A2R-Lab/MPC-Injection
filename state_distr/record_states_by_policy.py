#!/usr/bin/env python3
"""
Script to load a trained SAC-MPC model and record all the unique states that a walker visits during an episode.

The recorded states will be saved in a subdirectory to here and a separate plotting (UMAP or t-SNE) script will visualize them.
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