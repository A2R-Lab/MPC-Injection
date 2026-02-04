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
import gymnasium_robotics
from gymnasium_robotics.utils.rotations import quat_mul
import argparse


def generate_filename_shadow(cube_quat, goal_quat, rollout_horizon):
    """
    Generate filename for Shadow Hand trajectories from cube and goal orientations.
    Format: cube_[w,x,y,z]_goal_[w,x,y,z]_rh_XXXXX.npz
    Quaternion values are rounded to 2 decimal places.
    
    Args:
        cube_quat: Initial cube orientation quaternion (w,x,y,z)
        goal_quat: Goal cube orientation quaternion (w,x,y,z)
        rollout_horizon: Trajectory length
    """
    cube_str = ",".join([f"{q:.2f}" for q in cube_quat])
    goal_str = ",".join([f"{q:.2f}" for q in goal_quat])
    filename = f"cube_[{cube_str}]_goal_[{goal_str}]_rh_{rollout_horizon}.npz"
    return filename


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Generate single Shadow Hand MPC trajectory",
        epilog="""
Examples:
  # Basic usage (default seed 42)
  python shadow.py
  
  # With custom seed for reproducibility
  python shadow.py --seed 123
  
  # Save trajectory without visualization
  python shadow.py --seed 456 --no-plot --no-animate
  
  # Custom output directory
  python shadow.py --output-dir /path/to/output
  
Note: The random seed ensures reproducible initial states (cube/goal poses),
but MPC trajectories may have slight numerical variations due to iterative optimization.
        """
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Output directory (default: ../../data/shadow_hand_<dt>/)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving trajectory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--no-animate", action="store_true", help="Skip animation")
    args = parser.parse_args()
    
    # Seed for reproducibility
    np.random.seed(args.seed)
    
    model_path = (
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "mpc_rl/tasks/shadow_reorient/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    
    data = mujoco.MjData(model)
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Set camera to use the "floating", "sideon", "float_far", or "egocentric" camera
    renderer.enable_depth_rendering = False
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "sideon")
    print(f"Using camera: {camera_id}")

    # Set output directory
    if args.output_dir is None:
        dt = model.opt.timestep
        output_dir = Path(__file__).parent.parent.parent.parent.parent.parent / f"data/shadow_hand_{dt:.3f}dt".replace(".", "_")
    else:
        output_dir = Path(args.output_dir)
    
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    agent = agent_lib.Agent(task_id="Shadow", model=model)

    weights = {
        'In Hand': 20.0,        # Distance b/w cube pos and palm pos
        'Orientation': 10.0,    # Orientation err b/w cube and goal
        'Cube Vel.': 10.0,      # Linear velocity of cube (penalizing fast movements)
        'Actuator': 0.1,        # Control effort
        'Grasp': 1.0,           # Deviation of hand joints from keyframe "gasp" config
        'Joint Vel.': 1.0e-4,   # Angular velocities of hand joints
    }
    agent.set_cost_weights(weights)
    print("Cost weights:", agent.get_cost_weights())

    # task_params = {
    # }
    # agent.set_task_parameters(task_params)
    # print("Task params:", agent.get_task_parameters())

    # rollout horizon
    T = 500

    # trajectories
    qpos = np.zeros((model.nq, T))
    qvel = np.zeros((model.nv, T))
    ctrl = np.zeros((model.nu, T-1))
    time = np.zeros((T,))

    # costs
    cost_total = np.zeros(T-1)
    cost_terms = np.zeros((len(agent.get_cost_term_values()), T-1))

    # Rollout
    mujoco.mj_resetData(model, data)
    
    # Use a local RNG for reproducibility (doesn't affect global np.random state after this point)
    rng = np.random.RandomState(args.seed)
    
    # Randomize manipulated cube's initial pose (following manipulate_block.py)
    # Find the manipulated cube body and its qpos indices
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_jnt_addr = model.body_jntadr[cube_body_id]
    cube_qpos_start = model.jnt_qposadr[cube_jnt_addr]
    
    # Randomize position: base position + Gaussian noise (mean=0, std=0.005)
    base_pos = np.array([0.325, 0.0, 0.075])
    pos_noise = rng.normal(0, 0.005, size=3)
    random_pos = base_pos + pos_noise
    
    # Randomize orientation: start with identity quat (1,0,0,0) + random rotation around random axis
    random_angle = rng.uniform(-np.pi, np.pi)
    
    # Select a random axis (x, y, or z)
    axis_idx = rng.randint(0, 3)
    axis = np.zeros(3)
    axis[axis_idx] = 1.0
    
    # Convert axis-angle to quaternion
    half_angle = random_angle / 2
    sin_half = np.sin(half_angle)
    cube_init_quat = np.array([
        np.cos(half_angle),       # w
        axis[0] * sin_half,       # x
        axis[1] * sin_half,       # y
        axis[2] * sin_half        # z
    ])
    
    # Set the randomized pose in qpos
    data.qpos[cube_qpos_start:cube_qpos_start+3] = random_pos
    data.qpos[cube_qpos_start+3:cube_qpos_start+7] = cube_init_quat
    
    # Randomize goal cube's orientation (following manipulate_block.py)
    # The target orientation is obtained by adding a random offset to the initial cube orientation
    goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
    goal_jnt_addr = model.body_jntadr[goal_body_id]
    goal_qpos_start = model.jnt_qposadr[goal_jnt_addr]
    
    # Sample orientation offset: random angle on a random axis
    goal_angle_offset = rng.uniform(-np.pi, np.pi)
    goal_axis_idx = rng.randint(0, 3)
    goal_axis = np.zeros(3)
    goal_axis[goal_axis_idx] = 1.0
    
    # Convert offset axis-angle to quaternion
    half_offset = goal_angle_offset / 2
    sin_half_offset = np.sin(half_offset)
    offset_quat = np.array([
        np.cos(half_offset),
        goal_axis[0] * sin_half_offset,
        goal_axis[1] * sin_half_offset,
        goal_axis[2] * sin_half_offset
    ])
    
    # Multiply quaternions: goal_quat = offset_quat * cube_init_quat
    # This applies the offset rotation to the initial cube orientation
    goal_quat = quat_mul(offset_quat, cube_init_quat)
    data.qpos[goal_qpos_start:goal_qpos_start+4] = goal_quat
    
    # Forward kinematics to update both cubes
    mujoco.mj_forward(model, data)
    
    print(f"Randomized cube initial pose:")
    print(f"  Position: {random_pos}")
    print(f"  Quaternion: {cube_init_quat}")
    print(f"  Rotation axis: {['X', 'Y', 'Z'][axis_idx]}, angle: {random_angle:.3f} rad ({np.degrees(random_angle):.1f}°)")
    print(f"Goal cube orientation:")
    print(f"  Quaternion: {goal_quat}")
    print(f"  Offset axis: {['X', 'Y', 'Z'][goal_axis_idx]}, offset angle: {goal_angle_offset:.3f} rad ({np.degrees(goal_angle_offset):.1f}°)")

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
        num_steps = 1
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
    
    # Save trajectory
    if not args.no_save:
        print("\nSaving trajectory...")
        
        # Generate filename using cube and goal quaternions
        filename = generate_filename_shadow(cube_init_quat, goal_quat, T)
        filepath = output_dir / filename
        
        np.savez_compressed(
            filepath,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            time=time,
            cost_total=cost_total,
            cost_terms=cost_terms,
            init_qpos=qpos[:, 0],
            init_qvel=qvel[:, 0]
        )
        
        print(f"Trajectory saved to: {filepath}")
    
    # Plot trajectories
    if not args.no_plot:
        print("\nPlotting trajectories...")
    
    # Find cube body ID to extract its state
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
    
    # Extract cube position and orientation from qpos
    # Cube is a free body, so it has 7 DOFs: 3 pos + 4 quat
    # Find the cube's qpos address
    cube_qpos_addr = model.body_jntadr[cube_body_id]
    cube_qpos_start = model.jnt_qposadr[cube_qpos_addr]
    
    goal_qpos_addr = model.body_jntadr[goal_body_id]
    goal_qpos_start = model.jnt_qposadr[goal_qpos_addr]
    
    # Extract cube and goal trajectories
    cube_pos = qpos[cube_qpos_start:cube_qpos_start+3, :]     # x, y, z
    cube_quat = qpos[cube_qpos_start+3:cube_qpos_start+7, :]  # w, x, y, z
    
    goal_pos = qpos[goal_qpos_start:goal_qpos_start+3, :]     # x, y, z
    goal_quat = qpos[goal_qpos_start+3:goal_qpos_start+7, :]  # w, x, y, z
    
    # Convert quaternions to euler angles for easier interpretation
    def quat_to_euler(quat):
        """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)"""
        w, x, y, z = quat
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1,
                        np.copysign(np.pi / 2, sinp),
                        np.arcsin(sinp))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    # Convert cube and goal orientations to euler angles
    cube_euler = np.array([quat_to_euler(cube_quat[:, i]) for i in range(T)]).T
    goal_euler = np.array([quat_to_euler(goal_quat[:, i]) for i in range(T)]).T
    
    if not args.no_plot:
        # Plot 1: Cube Position Tracking
        fig1 = plt.figure(figsize=(14, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time, cube_pos[0, :], label="Cube X", color="blue", linewidth=2)
        plt.plot(time, goal_pos[0, :], label="Goal X", color="blue", linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylabel("X Position (m)")
        plt.grid(True)
        plt.title("Cube Position Tracking")
        
        plt.subplot(3, 1, 2)
        plt.plot(time, cube_pos[1, :], label="Cube Y", color="orange", linewidth=2)
        plt.plot(time, goal_pos[1, :], label="Goal Y", color="orange", linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylabel("Y Position (m)")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time, cube_pos[2, :], label="Cube Z", color="green", linewidth=2)
        plt.plot(time, goal_pos[2, :], label="Goal Z", color="green", linestyle='--', alpha=0.7)
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Z Position (m)")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Plot 2: Cube Orientation Tracking
        fig2 = plt.figure(figsize=(14, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time, cube_euler[0, :], label="Cube Roll", color="red", linewidth=2)
        plt.plot(time, goal_euler[0, :], label="Goal Roll", color="red", linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylabel("Roll (rad)")
        plt.grid(True)
        plt.title("Cube Orientation Tracking")
        
        plt.subplot(3, 1, 2)
        plt.plot(time, cube_euler[1, :], label="Cube Pitch", color="purple", linewidth=2)
        plt.plot(time, goal_euler[1, :], label="Goal Pitch", color="purple", linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylabel("Pitch (rad)")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time, cube_euler[2, :], label="Cube Yaw", color="brown", linewidth=2)
        plt.plot(time, goal_euler[2, :], label="Goal Yaw", color="brown", linestyle='--', alpha=0.7)
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw (rad)")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Plot 3: Position and Orientation Errors
        fig3 = plt.figure(figsize=(14, 6))
        
        # Compute errors
        pos_error = np.linalg.norm(cube_pos - goal_pos, axis=0)
        orient_error = np.linalg.norm(cube_euler - goal_euler, axis=0)
        
        plt.subplot(2, 1, 1)
        plt.plot(time, pos_error, color="blue", linewidth=2)
        plt.ylabel("Position Error (m)")
        plt.grid(True)
        plt.title("Cube Tracking Errors")
        
        plt.subplot(2, 1, 2)
        plt.plot(time, orient_error, color="red", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Orientation Error (rad)")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Plot 4: Control Signals (20 actuators for Shadow Hand)
        fig4 = plt.figure(figsize=(16, 10))
        
        # Shadow Hand has 20 actuators
        actuator_names = [
            "WRJ2 (wrist)", "WRJ1 (wrist)", 
            "THJ5 (thumb)", "THJ4 (thumb)", "THJ3 (thumb)", "THJ2 (thumb)", "THJ1 (thumb)",
            "FFJ4 (index)", "FFJ3 (index)", "FFJ0 (index)",
            "MFJ4 (middle)", "MFJ3 (middle)", "MFJ0 (middle)",
            "RFJ4 (ring)", "RFJ3 (ring)", "RFJ0 (ring)",
            "LFJ5 (pinky)", "LFJ4 (pinky)", "LFJ3 (pinky)", "LFJ0 (pinky)"
        ]
        
        for i in range(min(20, model.nu)):
            plt.subplot(5, 4, i+1)
            plt.plot(time[:-1], ctrl[i, :], linewidth=1.5)
            plt.ylabel("Control")
            if i < len(actuator_names):
                plt.title(actuator_names[i], fontsize=8)
            else:
                plt.title(f"Actuator {i}", fontsize=8)
            plt.grid(True, alpha=0.3)
            if i >= 16:
                plt.xlabel("Time (s)", fontsize=8)
            plt.tick_params(labelsize=7)
        
        plt.suptitle("Shadow Hand Control Signals", fontsize=14, y=0.995)
        plt.tight_layout()
        
        # Plot 5: Cost Terms
        fig5 = plt.figure(figsize=(14, 6))
        
        # Get cost term names from shadow task
        cost_names = list(agent.get_cost_term_values().keys())
        colors_cost = ["blue", "orange", "green", "red", "purple", "brown"]
        
        for i, (name, color) in enumerate(zip(cost_names, colors_cost)):
            if i < len(cost_terms):
                plt.plot(time[:-1], cost_terms[i, :], label=name, color=color, alpha=0.7)
        
        plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black", linewidth=2)
        plt.legend(loc='upper right')
        plt.xlabel("Time (s)")
        plt.ylabel("Cost")
        plt.title("Shadow Hand Manipulation Cost Terms")
        plt.grid(True)
        plt.tight_layout()
        
        plt.show()
    
    # Animate the frames
    if not args.no_animate:
        print(f"\nCreating animation with {len(frames)} frames...")
        fig_anim = plt.figure(figsize=(10, 6))
        img = plt.imshow(frames[0])
        plt.axis('off')
        plt.title("Shadow Hand MPC Rollout")
        
        def animate(i):
            img.set_data(frames[i])
            return [img]
        
        # Display at real-time speed based on model timestep
        display_fps = 1.0 / model.opt.timestep
        
        anim = animation.FuncAnimation(fig_anim, animate, frames=len(frames), 
                                       interval=1000/display_fps, blit=True, repeat=True)
        print(f"Animation created at {display_fps:.1f} FPS (model timestep: {model.opt.timestep}s)")
        
        plt.show()
    
    # Calculate orientation error after animation is displayed
    # Extract final cube orientation and goal orientation
    cube_quat_final = cube_quat[:, -1]  # Final cube quaternion (w,x,y,z)
    goal_quat_final = goal_quat[:, -1]  # Goal quaternion (should be constant, but use final)
    
    # Calculate quaternion error: q_error = goal_quat * conj(cube_quat_final)
    cube_quat_final_conj = np.array([cube_quat_final[0], -cube_quat_final[1], 
                                      -cube_quat_final[2], -cube_quat_final[3]])
    q_error = quat_mul(goal_quat_final, cube_quat_final_conj)
    
    # Orientation error = 2 * arcsin(||vec(q_error)||)
    orientation_error = 2.0 * np.arcsin(np.clip(np.linalg.norm(q_error[1:4]), 0, 1))
    
    # Define threshold (same as gen_traj_data.py default)
    orientation_error_threshold = 0.5  # radians
    
    # Print result
    print("\n" + "="*60)
    print("ORIENTATION ERROR ANALYSIS")
    print("="*60)
    print(f"Final cube orientation:  [{cube_quat_final[0]:.4f}, {cube_quat_final[1]:.4f}, {cube_quat_final[2]:.4f}, {cube_quat_final[3]:.4f}]")
    print(f"Goal orientation:        [{goal_quat_final[0]:.4f}, {goal_quat_final[1]:.4f}, {goal_quat_final[2]:.4f}, {goal_quat_final[3]:.4f}]")
    print(f"Orientation error:       {orientation_error:.4f} rad ({np.degrees(orientation_error):.2f}°)")
    print(f"Threshold:               {orientation_error_threshold:.4f} rad ({np.degrees(orientation_error_threshold):.2f}°)")
    print()
    if orientation_error < orientation_error_threshold:
        print("SUCCESS: Orientation error is BELOW threshold!")
    else:
        print("FAILURE: Orientation error is ABOVE threshold")
    print("="*60)