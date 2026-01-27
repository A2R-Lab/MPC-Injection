"""
Quick script to save a MuJoCo MJCF model as a URDF file.

NOTE: The xmls come from ~/MPC-RL/mpc_rl/tasks/walker/walker_modified.xml and are saved there too
"""

import mujoco
import os

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("../mpc_rl/tasks/walker/walker_modified.xml")

# Save as URDF
mujoco.mj_saveLastXML("../mpc_rl/tasks/walker/walker_modified.urdf", model)