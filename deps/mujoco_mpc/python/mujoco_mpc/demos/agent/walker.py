import numpy as np
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from pathlib import Path
from mujoco_mpc import agent as agent_lib
import mujoco

if __name__ == "__main__":
    """
    TODO: Ok, let's start simple with the walker to make sure everything is working properly
    instead of going through the MPCPlanner
    """
    model_path = (
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "mpc_rl/tasks/walker/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    
    data = mujoco.MjData(model)
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Set camera to use the "floating", "sideon", "float_far", or "egocentric" camera
    renderer.enable_depth_rendering = False
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "sideon")
    print(f"Using camera: {camera_id}")

    agent = agent_lib.Agent(task_id="Walker", model=model)

    weights = {
        'Speed': 1.0,
        'Height': 10.0,
        'Rotation': 3.0,
        'Control': 0.1
    }
    agent.set_cost_weights(weights)
    print("Cost weights:", agent.get_cost_weights())

    task_params = {
        'Speed Goal': 1.0,
        'Height Goal': 1.2
    }
    agent.set_task_parameters(task_params)
    print("Task params:", agent.get_task_parameters())

    # Print model information - masses and damping
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    # Print body masses
    print("\nBody Masses:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:  # Skip world body
            mass = model.body_mass[i]
            print(f"  {body_name}: {mass:.4f} kg")
    
    # Print joint damping
    print("\nJoint Damping:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        damping = model.dof_damping[model.jnt_dofadr[i]]
        print(f"  {joint_name}: {damping:.4f}")
    
    # Print geom masses (these contribute to body mass)
    print("\nGeom Masses (contributes to body mass):")
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        # Get geom mass from model
        # Note: geom masses are not directly stored in mjModel after compilation
        # They are used to compute body masses during compilation
        print(f"  {geom_name} (body: {body_name})")
    
    print("="*60 + "\n")

    # rollout horizon
    T = 1000

    # trajectories
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T-1))
    time = np.zeros((T,))

    # costs
    cost_total = np.zeros(T-1)
    cost_terms = np.zeros((len(agent.get_cost_term_values()), T-1))

    #
    # Rollout
    #
    mujoco.mj_resetData(model, data)

    # cache init state
    qpos[:, 0] = data.qpos.copy()
    qvel[:, 0] = data.qvel.copy()
    time[0] = data.time

    # frames for animation
    frames = []
    FPS = 1.0 / model.opt.timestep

    # simulate
    for t in range(T - 1):
        if t % 100 == 0:
            print(f"Step {t}/{T}")
        
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

        # run planner for num_steps
        num_steps = 2
        for _ in range(num_steps):
            agent.planner_step()
        
        # set control from agent policy
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

        # render and save frames with the tracking camera
        renderer.update_scene(data, camera=camera_id)
        pixels = renderer.render()
        frames.append(pixels)
    
    agent.reset()
    
    # Plot trajectories
    print("\nPlotting trajectories...")
    
    # Plot key joint positions
    fig1 = plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, qpos[0, :], label="rootz (height)", color="blue")
    plt.plot(time, qpos[1, :], label="rootx (forward)", color="orange")
    plt.legend()
    plt.ylabel("Root Position")
    plt.grid(True)
    plt.title("Walker Root Position Trajectories")
    
    plt.subplot(3, 1, 2)
    plt.plot(time, qpos[3, :], label="right_hip", color="red")
    plt.plot(time, qpos[4, :], label="right_knee", color="darkred")
    plt.plot(time, qpos[5, :], label="right_ankle", color="lightcoral")
    plt.legend()
    plt.ylabel("Right Leg Joints (rad)")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, qpos[6, :], label="left_hip", color="green")
    plt.plot(time, qpos[7, :], label="left_knee", color="darkgreen")
    plt.plot(time, qpos[8, :], label="left_ankle", color="lightgreen")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Left Leg Joints (rad)")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Plot velocities
    fig2 = plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, qvel[0, :], label="rootz velocity", color="blue")
    plt.plot(time, qvel[1, :], label="rootx velocity", color="orange")
    plt.legend()
    plt.ylabel("Root Velocity")
    plt.grid(True)
    plt.title("Walker Root Velocity Trajectories")
    
    plt.subplot(2, 1, 2)
    plt.plot(time, qvel[3, :], label="right_hip", color="red")
    plt.plot(time, qvel[6, :], label="left_hip", color="green")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Hip Velocities")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Plot controls (6 actuators)
    fig3 = plt.figure(figsize=(12, 8))
    
    control_names = ["right_hip", "right_knee", "right_ankle", 
                     "left_hip", "left_knee", "left_ankle"]
    colors = ["red", "darkred", "lightcoral", "green", "darkgreen", "lightgreen"]
    
    for i, (name, color) in enumerate(zip(control_names, colors)):
        plt.subplot(3, 2, i+1)
        plt.plot(time[:-1], ctrl[i, :], color=color)
        plt.ylabel("Control")
        plt.title(name)
        plt.grid(True)
        if i >= 4:
            plt.xlabel("Time (s)")
    
    plt.tight_layout()
    
    # Plot costs
    fig4 = plt.figure(figsize=(10, 6))
    
    # Get cost term names from walker task.xml
    cost_names = ["Control", "Height", "Rotation", "Speed"]
    colors_cost = ["blue", "orange", "green", "red"]
    
    for i, (name, color) in enumerate(zip(cost_names, colors_cost)):
        plt.plot(time[:-1], cost_terms[i, :], label=name, color=color)
    
    plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black", linewidth=2)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Costs")
    plt.title("Walker Walk Cost Terms")
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
    # Animate the frames
    print(f"\nCreating animation with {len(frames)} frames...")
    fig_anim = plt.figure(figsize=(10, 6))
    img = plt.imshow(frames[0])
    plt.axis('off')
    plt.title("Walker MPC Rollout")
    
    def animate(i):
        img.set_data(frames[i])
        return [img]
    
    # Display at real-time speed based on model timestep
    display_fps = 1.0 / model.opt.timestep
    
    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frames), 
                                   interval=1000/display_fps, blit=True, repeat=True)
    print(f"Animation created at {display_fps:.1f} FPS (model timestep: {model.opt.timestep}s)")
    
    plt.show()