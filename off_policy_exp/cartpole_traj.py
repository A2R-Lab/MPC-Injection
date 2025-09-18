import matplotlib.pyplot as plt
import mediapy as media
import mujoco
from mujoco import viewer
from mujoco_mpc import agent as agent_lib
import numpy as np

import pathlib

"""
This script is just for testing and eventually collecting trajectories
from mujoco_mpc for off-policy RL
"""

# Simple access to the XML file
# TODO: should make this proper by having mujoco_mpc as a submodule in this repo
xml_path = "/home/roy/mujoco_mpc/build/mjpc/tasks/cartpole/task.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

viewer.launch_passive(model, data)


viewer.close()