"""
Script to visualize the trajectories of the walker environment using viser.

Inspired by Se Hwan Jeon's script used in his Residual MPC paper: https://arxiv.org/abs/2510.12717

For us the plan is as follows:
1) Convert the walker MuJoCo model to URDF using cnvrt_mjcf_to_urdf.py
2) Use this URDF in viser to visualize the recorded trajectories (which now includes joint angles)
"""

import numpy as np
import viser
from viser.extras import ViserUrdf
import time
from pathlib import Path

