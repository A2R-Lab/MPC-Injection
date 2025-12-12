import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
import pathlib
import mujoco
from mujoco_mpc import agent as agent_lib

def gui_main():
    # Use local task_flat.xml (not the installed package version)
    # This allows us to use the modified go2 robot
    model_path = (
        pathlib.Path(__file__).parent.parent
        / "mpc_rl/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)


    with agent_lib.Agent(
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat",
        model=model,
    ) as agent:
        while True:
            pass


if __name__ == "__main__":

    gui_main()

    exit(0)


    """
    Quadruped MPC trajectory planning and animation.
    Uses the task_flat.xml configuration which includes:
    - Cost terms: Upright, Height, Position, Gait, Balance, Effort, Posture, Orientation, Angmom
    - 12 actuators (3 per leg: hip, thigh, calf)
    - Flat terrain with obstacles (box, ramp, hill)
    """
    model_path = (
        pathlib.Path(__file__).parent.parent
        / "mpc_rl/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    
    data = mujoco.MjData(model)
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Set camera - use default camera (0) or find a good tracking camera
    renderer.enable_depth_rendering = False
    # Try to find a camera, otherwise use default
    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    except:
        camera_id = 0  # Use default camera
    print(f"Using camera: {camera_id}")

    # Create agent for Quadruped task with larger gRPC message limits
    # The quadruped model with meshes exceeds default 40MB gRPC limit
    # Pass extra flags to the server to increase its message size limits
    #extra_server_flags = [
    #    '--grpc_max_send_message_length=104857600',  # 100MB
    #    '--grpc_max_receive_message_length=104857600',  # 100MB
    #]
    
    agent = agent_lib.Agent(
        task_id="Quadruped Flat", 
        model=model,
    #    extra_flags=extra_server_flags
    )

    # Set cost weights based on task_flat.xml cost sensors
    # The task includes: Upright, Height, Position, Gait, Balance, Effort, Posture, Orientation, Angmom
    weights = {
        'Upright': 3.0,
        'Height': 2.0,
        'Position': 1.0,
        'Gait': 5.0,
        'Balance': 2.0,
        'Effort': 0.1,
        'Posture': 0.5,
        'Orientation': 1.0,
        'Angmom': 0.5
    }
    agent.set_cost_weights(weights)
    print("Cost weights:", agent.get_cost_weights())

    # Set task parameters - adjust based on quadruped gait
    task_params = {
        'Cadence': 2.0,
        'Amplitude': 0.06,
        'Duty ratio': 0.5,
        'Walk speed': 1.0,
        'Walk turn': 0.0,
    }
    agent.set_task_parameters(task_params)
    print("Task params:", agent.get_task_parameters())

    # Print model information
    print("\n" + "="*60)
    print("QUADRUPED MODEL INFORMATION")
    print("="*60)
    
    print(f"\nDegrees of freedom: {model.nv}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of bodies: {model.nbody}")
    
    # Print actuator names
    print("\nActuators:")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")
    
    print("="*60 + "\n")

    # Rollout horizon
    T = 1000

    # Trajectories
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T-1))
    time = np.zeros((T,))

    # Costs
    cost_total = np.zeros(T-1)
    cost_terms = np.zeros((len(agent.get_cost_term_values()), T-1))

    #
    # Rollout
    #
    mujoco.mj_resetData(model, data)

    # Set initial state to "home" keyframe if it exists
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        print("Starting from 'home' keyframe")
    else:
        print("Starting from default initial state")

    # Cache init state
    qpos[:, 0] = data.qpos.copy()
    qvel[:, 0] = data.qvel.copy()
    time[0] = data.time

    # Frames for animation
    frames = []
    FPS = 1.0 / model.opt.timestep

    # Simulate
    print(f"Running MPC simulation for {T} steps...")
    for t in range(T - 1):
        if t % 100 == 0:
            print(f"Step {t}/{T}")
        
        # Set planner state
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        # Run planner for num_steps
        num_steps = 2
        for _ in range(num_steps):
            agent.planner_step()
        
        # Set control from agent policy
        data.ctrl = agent.get_action()
        ctrl[:, t] = data.ctrl

        # Get costs
        cost_total[t] = agent.get_total_cost()
        for i, c in enumerate(agent.get_cost_term_values().items()):
            cost_terms[i, t] = c[1]

        # Step simulation
        mujoco.mj_step(model, data)

        # Cache state
        qpos[:, t + 1] = data.qpos
        qvel[:, t + 1] = data.qvel
        time[t + 1] = data.time

        # Render and save frames
        renderer.update_scene(data, camera=camera_id)
        pixels = renderer.render()
        frames.append(pixels)
    
    agent.reset()
    
    print(f"\nSimulation complete. Total steps: {T}")
    print(f"Final position: x={qpos[0, -1]:.2f}, y={qpos[1, -1]:.2f}, z={qpos[2, -1]:.2f}")
    
    # Animate the frames
    print(f"\nCreating animation with {len(frames)} frames...")
    fig_anim = plt.figure(figsize=(10, 6))
    img = plt.imshow(frames[0])
    plt.axis('off')
    plt.title("Quadruped MPC Rollout")
    
    def animate(i):
        img.set_data(frames[i])
        return [img]
    
    # Display at real-time speed based on model timestep
    display_fps = 1.0 / model.opt.timestep
    
    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frames), 
                                   interval=1000/display_fps, blit=True, repeat=True)
    print(f"Animation created at {display_fps:.1f} FPS (model timestep: {model.opt.timestep}s)")
    
    plt.show()
