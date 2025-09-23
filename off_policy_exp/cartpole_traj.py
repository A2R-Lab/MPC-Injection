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

# Create mujoco_mpc agent (contains planner & policy)
agent = agent_lib.Agent(task_id="Cartpole", model=model) # Cartpole default only has one task (swingup)

agent.set_cost_weights({"Velocity": 0.15})
print("Cost weights: ", agent.get_cost_weights())

agent.set_task_parameter("Goal", 1.5) # set cart's goal position
print("Task parameters: ", agent.get_task_parameters())

# Rollout Horizon
T = 2000

# Trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T - 1))
time = np.zeros(T)

# Costs
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

# Rollout
mujoco.mj_resetData(model, data) # reset to initial state

# Launch viewer for visualization
mj_viewer = viewer.launch_passive(model, data)

# Cache the initial state
qpos[:, 0] = data.qpos
qvel[:, 0] = data.qvel
time[0]    = data.time

# Simulate
for t in range(T - 1):
    if t % 100 == 0:
        print(f"Step {t}/{T}")
    
    # Check if viewer is still running
    if not mj_viewer.is_running():
        print("Viewer closed, stopping simulation early.")
        # Trim arrays to actual length
        T = t + 1
        qpos = qpos[:, :T]
        qvel = qvel[:, :T]
        ctrl = ctrl[:, :T-1]
        time = time[:T]
        cost_total = cost_total[:T-1]
        cost_terms = cost_terms[:, :T-1]
        break
    
    # Set the planner state
    agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=data.mocap_pos,
        mocap_quat=data.mocap_quat,
        userdata=data.userdata
    )

    # Run planner for num_steps
    num_steps = 10
    for _ in range(num_steps):
        agent.planner_step()
    
    # Set ctrl from agent policy
    data.ctrl = agent.get_action()
    ctrl[:, t] = data.ctrl

    # Get costs
    cost_total[t] = agent.get_total_cost()
    for i, c in enumerate(agent.get_cost_term_values().items()):
        cost_terms[i, t] = c[1]

    # Step the simulation
    mujoco.mj_step(model, data)
    
    # Sync viewer (this updates the visual display)
    mj_viewer.sync()

    # Cache the next state
    qpos[:, t + 1] = data.qpos
    qvel[:, t + 1] = data.qvel
    time[t + 1]    = data.time

print("Simulation completed!")

# Close viewer
mj_viewer.close()

# Reset the agent
agent.reset()

# plot position
fig = plt.figure()

plt.plot(time, qpos[0, :], label="q0", color="blue")
plt.plot(time, qpos[1, :], label="q1", color="orange")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Configuration")

# plot velocity
fig = plt.figure()

plt.plot(time, qvel[0, :], label="v0", color="blue")
plt.plot(time, qvel[1, :], label="v1", color="orange")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity")

# plot control
fig = plt.figure()

plt.plot(time[:-1], ctrl[0, :], color="blue")

plt.xlabel("Time (s)")
plt.ylabel("Control")

plt.show()