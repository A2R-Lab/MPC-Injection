"""
Test script to verify that the mpc_planner is working as expected for certain trajectories and environments.

The comparisons will be hardcoded for now.

The main purpose is to ensure the functions are working as expected before integrating with SAC-MPC.
"""

import numpy as np
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from pathlib import Path

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.planner.mpc_planner import MPCPlanner

# For walker-walk-v0
def planner_test_walker_walk():
    print("Testing MPCPlanner for Walker Walk...")
    
    model_path = (Path(__file__).parent.parent
                  / "mpc_rl/tasks/walker/task.xml")

    # Create MPC planner instance
    # Based on walker/task.xml configuration:
    # - agent_horizon: 0.8s
    # - agent_timestep: 0.01s
    # 
    # Cost terms (from XML sensor definitions):
    #   <user name="Control" dim="6" user="0 0.1 0.0 1.0" />
    #     - Format: user="norm_type weight min max"
    #     - Weight: 0.1 applies to all 6 control dimensions
    #   <user name="Height" dim="1" user="0 10.0 0.0 10.0" />
    #     - Weight: 10.0
    #   <user name="Rotation" dim="1" user="0 3.0 0.0 5.0" />
    #     - Weight: 3.0
    #   <user name="Speed" dim="1" user="0 1.0 0.0 1.0" />
    #     - Weight: 1.0
    #
    # Task parameters (from XML residual definitions):
    #   <numeric name="residual_Height Goal" data="1.2 0.5 1.2" />
    #     - Format: data="default min max"
    #     - Default: 1.2
    #   <numeric name="residual_Speed Goal" data="0 -5.0 5.0" />
    #     - Format: data="default min max"
    #     - Default: 0
    planner = MPCPlanner(
        model_path=model_path,
        task_id="Walker",
        rollout_horizon=5000,
        opt_steps=10,
        weights={
            "Control": 0.1,
            "Height": 10.0,
            "Rotation": 3.0,
            "Speed": 1.0
        },
        task_params={
            "Height Goal": 1.2,
            "Speed Goal": 1.0  # Default is 0 (stationary), not 1.0
        },
        init_state_noise_flag=False,
        qpos_noise_rnge=(-0.0, 0.0),
        qvel_noise_rnge=(-0.0, 0.0),
        verbose=1
    )
    
    print(f"Rollout horizon: {planner.get_rollout_horizon()}")
    print(f"Initial state noise: {planner.get_init_state_noise_flag()}")
    print(f"qpos noise range: {planner.get_qpos_noise_range()}")
    print(f"qvel noise range: {planner.get_qvel_noise_range()}")
    
    # Run MPC planning
    print("\nRunning MPC trajectory optimization for Walker...")
    #planner.plan(keyframe="home")
    planner.plan_receding_horizon(keyframe="home", plan_frequency=10)
    print("Planning complete!")
    
    # Get trajectories and costs
    qpos, qvel, ctrl, time = planner.get_trajectories()
    cost_total, cost_terms = planner.get_costs()
    
    print(f"\nTrajectory shapes:")
    print(f"  qpos: {qpos.shape}")
    print(f"  qvel: {qvel.shape}")
    print(f"  ctrl: {ctrl.shape}")
    print(f"  time: {time.shape}")
    
    # Walker has 9 joints: rootz, rootx, rooty, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle
    # And 6 actuators: right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle
    
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
    
    # Create animation of walker using the MPC controls
    print("\nCreating walker animation with MPC controls...")
    
    # Downsample controls to match dm_control timestep
    # Walker MPC physics timestep: 0.0025s (from walker_modified.xml)
    # DM Control walker control_timestep: typically 0.0025s (matches physics)
    # So we can use controls directly without downsampling, or downsample slightly for smoother animation
    # For smoother animation, we'll downsample by 2 to get ~0.005s per frame
    downsample_factor = 2  # Adjust for animation smoothness
    ctrl_downsampled = ctrl[:, ::downsample_factor]
    
    # Create walker environment from dm_control with fixed random seed for consistency
    dm_env = suite.load(domain_name="walker", task_name="walk", 
                        task_kwargs={'random': np.random.RandomState(42)})
    env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
    env = FlattenObservation(env)
    
    # Reset environment and set to the same initial state as MPC planner
    obs, info = env.reset()
    
    # Set walker to the same initial state used by MPC planner (from keyframe "home")
    physics = env.unwrapped._env.physics
    with physics.reset_context():
        physics.data.qpos[:] = qpos[:, 0]  # Use the initial qpos from MPC trajectory
        physics.data.qvel[:] = qvel[:, 0]  # Use the initial qvel from MPC trajectory
    physics.forward()
    
    # Collect frames by applying downsampled controls
    print("Applying MPC controls to walker environment...")
    frames = []
    
    # Get initial frame
    frame = env.render()
    frames.append(frame)
    
    # Apply each control action
    num_steps = min(ctrl_downsampled.shape[1], 1000)  # Limit to 1000 steps for animation
    for t in range(num_steps):
        action = ctrl_downsampled[:, t]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and save frame
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            print(f"Episode ended at step {t}")
            break
    
    env.close()
    print(f"Collected {len(frames)} frames")
    
    # Create animation
    print("Creating animation...")
    fig_anim = plt.figure(figsize=(10, 6))
    img = plt.imshow(frames[0])
    plt.axis('off')
    plt.title("Walker with MPC Controls")
    
    def animate(i):
        img.set_data(frames[i])
        return [img]
    
    # Display at 30 FPS for smooth playback (regardless of actual simulation rate)
    display_fps = 30
    
    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frames), 
                                   interval=1000/display_fps, blit=True, repeat=True)
    print(f"Animation created with {len(frames)} frames at {display_fps} FPS")
    
    plt.show()
    
    print("\nWalker Walk Test complete!")


# For cartpole swingup
def planner_test_cartpole_swingup():
    print("Testing MPCPlanner...")
    
    # Create MPC planner instance
    planner = MPCPlanner(
        rollout_horizon=10000,
        opt_steps=10,
        weights={
            "Vertical": 10.0,
            "Centered": 10.0,
            "Velocity": 0.1,
            "Control": 0.1
        },
        task_params={"Goal": 0.0},
        init_state_noise_flag=False,
        qpos_noise_rnge=(-0.02, 0.02),
        qvel_noise_rnge=(-0.02, 0.02)
    )
    
    print(f"Rollout horizon: {planner.get_rollout_horizon()}")
    print(f"Initial state noise: {planner.get_init_state_noise_flag()}")
    print(f"qpos noise range: {planner.get_qpos_noise_range()}")
    print(f"qvel noise range: {planner.get_qvel_noise_range()}")
    
    # Run MPC planning
    print("\nRunning MPC trajectory optimization...")
    planner.plan(keyframe="home")
    print("Planning complete!")
    
    # Get trajectories and costs
    qpos, qvel, ctrl, time = planner.get_trajectories()
    cost_total, cost_terms = planner.get_costs()
    
    print(f"\nTrajectory shapes:")
    print(f"  qpos: {qpos.shape}")
    print(f"  qvel: {qvel.shape}")
    print(f"  ctrl: {ctrl.shape}")
    print(f"  time: {time.shape}")
    
    # Plot position
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(time, qpos[0, :], label="q0 (cart position)", color="blue")
    plt.plot(time, qpos[1, :], label="q1 (pole angle)", color="orange")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("States")
    plt.title("State Trajectories")
    plt.grid(True)
    plt.tight_layout()
    
    # Plot velocity
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(time, qvel[0, :], label="v0 (cart velocity)", color="blue")
    plt.plot(time, qvel[1, :], label="v1 (pole velocity)", color="orange")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Trajectories")
    plt.grid(True)
    plt.tight_layout()
    
    # Plot control
    fig3 = plt.figure(figsize=(10, 4))
    plt.plot(time[:-1], ctrl[0, :], color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Control")
    plt.title("Control Signal")
    plt.grid(True)
    plt.tight_layout()
    
    # Plot costs
    fig4 = plt.figure(figsize=(10, 6))
    
    # Get cost term names (matching mjpc_ex.py style)
    cost_names = ["Vertical", "Centered", "Velocity", "Control"]
    for i, name in enumerate(cost_names):
        plt.plot(time[:-1], cost_terms[i, :], label=name)
    
    plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black", linewidth=2)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Costs")
    plt.title("Cost Terms")
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
    print("\nTest complete!")

def planner_receding_horizon_test():
    print("Testing MPCPlanner with Receding Horizon...")
    
    # Create MPC planner instance
    planner = MPCPlanner(
        rollout_horizon=10000,
        opt_steps=10,
        weights={
            "Vertical": 10.0,
            "Centered": 10.0,
            "Velocity": 0.1,
            "Control": 0.1
        },
        task_params={"Goal": 0.0},
        init_state_noise_flag=False,
        qpos_noise_rnge=(-0.02, 0.02),
        qvel_noise_rnge=(-0.02, 0.02),
        verbose=1
    )
    
    print(f"Rollout horizon: {planner.get_rollout_horizon()}")
    print(f"Initial state noise: {planner.get_init_state_noise_flag()}")
    print(f"qpos noise range: {planner.get_qpos_noise_range()}")
    print(f"qvel noise range: {planner.get_qvel_noise_range()}")
    
    # Run MPC planning with receding horizon (re-plan every 50 steps for speedup)
    print("\nRunning MPC receding horizon trajectory optimization...")
    print("(Re-planning every 50 steps for faster execution)")
    import time
    start_time = time.time()
    planner.plan_receding_horizon(keyframe="home", plan_frequency=10)
    end_time = time.time()
    print(f"Planning complete! Took {end_time - start_time:.2f} seconds")
    
    # Get trajectories
    qpos, qvel, ctrl, time_array = planner.get_trajectories()
    
    print(f"\nTrajectory shapes:")
    print(f"  qpos: {qpos.shape}")
    print(f"  qvel: {qvel.shape}")
    print(f"  ctrl: {ctrl.shape}")
    print(f"  time: {time_array.shape}")
    
    # Plot position
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(time_array, qpos[0, :], label="q0 (cart position)", color="blue")
    plt.plot(time_array, qpos[1, :], label="q1 (pole angle)", color="orange")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("States")
    plt.title("State Trajectories (Receding Horizon)")
    plt.grid(True)
    plt.tight_layout()
    
    # Plot velocity
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(time_array, qvel[0, :], label="v0 (cart velocity)", color="blue")
    plt.plot(time_array, qvel[1, :], label="v1 (pole velocity)", color="orange")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Trajectories (Receding Horizon)")
    plt.grid(True)
    plt.tight_layout()
    
    # Plot control
    fig3 = plt.figure(figsize=(10, 4))
    plt.plot(time_array[:-1], ctrl[0, :], color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Control")
    plt.title("Control Signal (Receding Horizon)")
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
    print("\nReceding Horizon Test complete!")

def plan_and_save_traj():
    """
    Simple function to save a planned trajectory to load and use for later
    """
    # Create MPC planner instance
    planner = MPCPlanner(
        rollout_horizon=10000,
        opt_steps=10,
        weights={
            "Vertical": 10.0,
            "Centered": 10.0,
            "Velocity": 0.1,
            "Control": 0.1
        },
        task_params={"Goal": 0.0},
        init_state_noise_flag=False,
        qpos_noise_rnge=(-0.02, 0.02),
        qvel_noise_rnge=(-0.02, 0.02),
        verbose=1
    )
    
    # Run MPC planning
    print("\nRunning MPC trajectory optimization...")
    planner.plan(keyframe="home")
    print("Planning complete!")
    
    # Get trajectories and costs
    qpos, qvel, ctrl, time = planner.get_trajectories()
    cost_total, cost_terms = planner.get_costs()

    # Save trajs to file
    save_path = Path(__file__).parent / "mpc_trajectories.npz"
    np.savez_compressed(save_path, qpos=qpos, qvel=qvel, ctrl=ctrl, time=time)
    print(f"\nTrajs saved to: {save_path}")

def downsample_test():
    """
    This is a sanity check test to verify that downsampling the action trajectory from the planner
    results in the same expected trajectory in the RL environment.
    """
    # Create MPC planner instance
    planner = MPCPlanner(
        rollout_horizon=10000,
        opt_steps=10,
        weights={
            "Vertical": 10.0,
            "Centered": 10.0,
            "Velocity": 0.1,
            "Control": 0.1
        },
        task_params={"Goal": 0.0},
        init_state_noise_flag=False,
        qpos_noise_rnge=(-0.02, 0.02),
        qvel_noise_rnge=(-0.02, 0.02)
    )

    # Load trajectories from saved file
    load_path = Path(__file__).parent / "mpc_trajectories.npz"
    data = np.load(load_path)
    qpos = data["qpos"]
    qvel = data["qvel"]
    ctrl = data["ctrl"]
    time = data["time"]

    # Downsampling code taken from mpc_inject_callbacks.py
    downsample_factor = 10  # MPC at 0.001s, RL at 0.01s
    # Hack to set planner's internal ctrl to loaded ctrl for downsampling
    planner.ctrl = ctrl
    ctrl_downsampled = planner.get_ctrl_downsampled(downsample_factor)

    # Create downsampled time array
    time_downsampled = time[:-1:downsample_factor]

    # Plot original control
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(time[:-1], ctrl[0, :], color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Control")
    plt.title("Original Control Signal")
    plt.grid(True)
    plt.tight_layout()
    
    # Plot downsampled control
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(time_downsampled, ctrl_downsampled[0, :], color="red", marker='o', 
             linestyle='-', markersize=4)
    plt.xlabel("Time (s)")
    plt.ylabel("Control")
    plt.title(f"Downsampled Control Signal (Factor: {downsample_factor})")
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
    # Create cartpole environment from dm_control
    print("\nCreating cartpole environment...")
    dm_env = suite.load(domain_name="cartpole", task_name="swingup")
    env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
    env = FlattenObservation(env)
    
    # Reset environment to initial state
    obs, info = env.reset()
    
    # Collect frames by applying downsampled controls
    print("Applying downsampled controls to environment...")
    frames = []
    
    # Get initial frame
    frame = env.render()
    frames.append(frame)
    
    # Apply each control action
    num_steps = ctrl_downsampled.shape[1]
    for t in range(num_steps):
        action = ctrl_downsampled[:, t]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and save frame
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            print(f"Episode ended at step {t}")
            break
    
    env.close()
    print(f"Collected {len(frames)} frames")
    
    # Create animation
    print("Creating animation...")
    fig_anim = plt.figure(figsize=(8, 6))
    img = plt.imshow(frames[0])
    plt.axis('off')
    plt.title("Cartpole with Downsampled MPC Controls")
    
    def animate(i):
        img.set_data(frames[i])
        return [img]
    
    # Calculate FPS based on environment timestep (0.01s for dm_control cartpole)
    env_dt = 0.01  # 10ms per step for dm_control cartpole
    FPS = 1.0 / env_dt
    
    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frames), 
                                   interval=1000/FPS, blit=True, repeat=True)
    print(f"Animation created with {len(frames)} frames at {FPS:.1f} FPS")
    
    plt.show()


if __name__ == "__main__":
    # Uncomment the test you want to run:
    
    planner_test_walker_walk()  # Test Walker MPC planning
    #planner_test_cartpole_swingup()  # Test standard MPC planning
    #planner_receding_horizon_test()  # Test receding horizon MPC planning (faster!)
    #plan_and_save_traj()  # Save a trajectory for later use
    #downsample_test()  # Test downsampling and visualization