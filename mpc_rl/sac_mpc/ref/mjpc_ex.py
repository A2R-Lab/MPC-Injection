import matplotlib.pyplot as plt
from matplotlib import animation
import mediapy as media
import mujoco
# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib
import numpy as np

import pathlib

"""
Test script for mujoco_mpc's python API.
"""

# model
model_path = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "mpc_rl/tasks/cartpole/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# data
data = mujoco.MjData(model)

# renderer
renderer = mujoco.Renderer(model)

# agent
agent = agent_lib.Agent(task_id="Cartpole", model=model)

# weights
weights = {
  "Vertical": 10.0,      # Pendulum verticality cost
  "Centered": 10.0,      # Cart position cost
  "Velocity": 0.1,       # Velocity cost
  "Control": 0.1         # Control cost
}
agent.set_cost_weights(weights)
print("Cost weights:", agent.get_cost_weights())

# parameters - Goal is the target cart position (0.0 = center)
task_params = {"Goal": 0.0}
agent.set_task_parameters(task_params)
print("Parameters:", agent.get_task_parameters())

# rollout horizon
# NOTE: the planner runs at 100 Hz but the sim itself is 0.001 dt
#       so 10,000 steps = 10 seconds
T = 5000*2

# trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T - 1))
time = np.zeros(T)

# costs
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

# rollout
mujoco.mj_resetData(model, data)

# reset to "home" keyframe seen in the task.xml
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if home_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, home_id)
else:
    print("Warning: 'home' keyframe not found in model")

if False:
  # Add random perturbations to initial state to match DM Control behavior
  # Set seed for reproducibility (comment out for different random states each run)
  np.random.seed(42)

  # Add small random offsets to position and velocity
  # These ranges match the typical DM Control cartpole swingup initialization
  qpos_noise = np.random.uniform(-0.02, 0.02, size=model.nq)  # +-0.02 for positions
  qvel_noise = np.random.uniform(-0.02, 0.02, size=model.nv)  # +-0.02 for velocities

  data.qpos[:] += qpos_noise
  data.qvel[:] += qvel_noise

print(f"Initial state with random perturbations:")
print(f"  qpos: {data.qpos}")
print(f"  qvel: {data.qvel}")

# cache initial state
qpos[:, 0] = data.qpos
qvel[:, 0] = data.qvel
time[0] = data.time

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# simulate
for t in range(T - 1):
  if t % 100 == 0:
    print("t = ", t)

  # set planner state
  agent.set_state(
      time=data.time,
      qpos=data.qpos,
      qvel=data.qvel,
      act=data.act,
      mocap_pos=data.mocap_pos,
      mocap_quat=data.mocap_quat,
      userdata=data.userdata,
  )

  # run planner for num_steps - # iterations for optimization
  # more iterations for faster/better plans
  num_steps = 10
  
  for _ in range(num_steps):
    agent.planner_step()

  # set ctrl from agent policy
  data.ctrl = agent.get_action()
  ctrl[:, t] = data.ctrl

  # get costs
  cost_total[t] = agent.get_total_cost()
  for i, c in enumerate(agent.get_cost_term_values().items()):
    cost_terms[i, t] = c[1]

  # step
  mujoco.mj_step(model, data)

  # cache
  qpos[:, t + 1] = data.qpos
  qvel[:, t + 1] = data.qvel
  time[t + 1] = data.time

  # render and save frames
  renderer.update_scene(data)
  pixels = renderer.render()
  frames.append(pixels)

# reset
agent.reset()

# display frames as animation
print("Creating animation...")
fig_anim = plt.figure(figsize=(8, 6))
img = plt.imshow(frames[0])
plt.axis('off')
plt.title("Cartpole Rollout")

def animate(i):
    img.set_data(frames[i])
    return [img]

anim = animation.FuncAnimation(fig_anim, animate, frames=len(frames), 
                               interval=1000/FPS, blit=True, repeat=True)
print(f"Animation created with {len(frames)} frames at {FPS:.1f} FPS")

# plot position
fig = plt.figure()

plt.plot(time, qpos[0, :], label="q0", color="blue")
plt.plot(time, qpos[1, :], label="q1", color="orange")
print("qpos size:", qpos.shape)
print("qvel size:", qvel.shape)
print("control size:", ctrl.shape)

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

# plot costs
fig = plt.figure()

for i, c in enumerate(agent.get_cost_term_values().items()):
  plt.plot(time[:-1], cost_terms[i, :], label=c[0])

plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Costs")
plt.show()