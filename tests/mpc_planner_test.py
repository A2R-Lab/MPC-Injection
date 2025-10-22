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

def planner_test():
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
    #planner_test()
    #plan_and_save_traj()
    downsample_test()