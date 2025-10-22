import numpy as np
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from pathlib import Path
import random

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.planner.mpc_planner import MPCPlanner


def select_trajectory(data_dir=None, random_select=True, filename=None):
    """
    Select a trajectory file from the data directory.
    
    Args:
        data_dir: Path to data directory (defaults to ../data/)
        random_select: If True, randomly select a file. If False, use filename argument.
        filename: Specific filename to load (used when random_select=False)
    
    Returns:
        Path to selected trajectory file
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    # Get all trajectory files
    traj_files = list(data_dir.glob("*.npz"))
    
    if len(traj_files) == 0:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    
    if random_select:
        selected_file = random.choice(traj_files)
        print(f"Randomly selected: {selected_file.name}")
    else:
        if filename is None:
            raise ValueError("filename must be provided when random_select=False")
        selected_file = data_dir / filename
        if not selected_file.exists():
            raise FileNotFoundError(f"File not found: {selected_file}")
        print(f"Selected: {selected_file.name}")
    
    return selected_file


def test_trajectory_in_environment():
    """
    Test a saved MPC trajectory by running its downsampled controls through a cartpole environment.
    This verifies that the MPC-planned trajectory works in the RL environment.
    """
    # Select a trajectory file from data directory
    filename="qpos_[0.00,3.64]_qvel_[-1.00,-0.50]_rh_10000.npz"
    traj_file = select_trajectory(random_select=False, filename=filename)
    
    # Load the trajectory data
    print(f"\nLoading trajectory from: {traj_file}")
    data = np.load(traj_file)
    qpos = data["qpos"]
    qvel = data["qvel"]
    ctrl = data["ctrl"]
    time = data["time"]
    init_qpos = data["init_qpos"]
    init_qvel = data["init_qvel"]
    
    print(f"Initial state: qpos={init_qpos}, qvel={init_qvel}")
    print(f"Trajectory length: {qpos.shape[1]} steps")
    
    # Downsample the control trajectory
    downsample_factor = 10  # MPC at 0.001s, RL at 0.01s
    ctrl_downsampled = ctrl[:, ::downsample_factor]
    time_downsampled = time[:-1:downsample_factor]
    
    print(f"Downsampled control length: {ctrl_downsampled.shape[1]} steps")
    
    # Plot original vs downsampled control
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(time[:-1], ctrl[0, :], color="blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Control")
    ax1.set_title("Original Control Signal (MPC)")
    ax1.grid(True)
    
    ax2.plot(time_downsampled, ctrl_downsampled[0, :], color="red", marker='o', 
             linestyle='-', markersize=3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Control")
    ax2.set_title(f"Downsampled Control Signal (Factor: {downsample_factor})")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create cartpole environment from dm_control
    print("\nCreating cartpole environment...")
    dm_env = suite.load(domain_name="cartpole", task_name="swingup")
    env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
    env = FlattenObservation(env)
    
    # Reset environment first
    obs, info = env.reset()
    
    # Set the environment to the MPC initial state
    # Access the underlying dm_control physics and set qpos/qvel directly
    env.unwrapped._env.physics.data.qpos[:] = init_qpos
    env.unwrapped._env.physics.data.qvel[:] = init_qvel
    
    # Forward the physics to ensure consistent state
    env.unwrapped._env.physics.forward()
    
    # Get the observation after setting the state
    obs = np.array([
        init_qpos[0],  # cart position
        np.cos(init_qpos[1]),  # cos(pole angle)
        np.sin(init_qpos[1]),  # sin(pole angle)
        init_qvel[0],  # cart velocity
        init_qvel[1]   # pole angular velocity
    ], dtype=np.float32)
    
    print(f"Environment initialized to: qpos={init_qpos}, qvel={init_qvel}")
    
    # Collect frames by applying downsampled controls
    print("Applying downsampled controls to environment...")
    frames = []
    states_from_env = []
    
    # Get initial frame
    frame = env.render()
    frames.append(frame)
    
    # Apply each control action
    num_steps = min(ctrl_downsampled.shape[1], 500)  # Limit to 500 steps for visualization
    for t in range(num_steps):
        action = ctrl_downsampled[:, t]
        
        # Stop if control becomes all zeros
        if np.allclose(action, 0.0, atol=1e-3):
            print(f"Control became zero at step {t}, stopping...")
            break
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and save frame
        frame = env.render()
        frames.append(frame)
        states_from_env.append(obs)
        
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
    plt.title(f"Trajectory Test: qpos={init_qpos}, qvel={init_qvel}")
    
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
    # Test a single trajectory in the environment
    test_trajectory_in_environment()