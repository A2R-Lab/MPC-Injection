"""
TODO: Test script to verify that the mpc_planner is working as expected for certain trajectories and environments.

The comparisons will be hardcoded for now.

The main purpose is to ensure the functions are working as expected before integrating with SAC-MPC.
"""

import numpy as np
from mpc_rl.sac_mpc.mpc_planner import MPCPlanner