"""
Test script to verify that the mpc_planner is working as expected for certain trajectories and environments.

The comparisons will be hardcoded for now.

The main purpose is to ensure the functions are working as expected before integrating with SAC-MPC.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from pathlib import Path

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.sac_mpc.mpc_planner import MPCPlanner

if __name__ == "__main__":
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