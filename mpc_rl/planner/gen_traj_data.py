import numpy as np
from pathlib import Path
import sys
from itertools import product
import mujoco

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mpc_rl.planner.mpc_planner import MPCPlanner


def generate_filename(qpos, qvel, rollout_horizon):
    """
    Generate filename from initial state and rollout horizon.
    Format: qpos_[x.xx,x.xx]_qvel_[x.xx,x.xx]_rh_XXXXX.npz
    Values are rounded to 2 decimal places for ease of comparison.
    """
    qpos_str = ",".join([f"{q:.2f}" for q in qpos])
    qvel_str = ",".join([f"{v:.2f}" for v in qvel])
    filename = f"qpos_[{qpos_str}]_qvel_[{qvel_str}]_rh_{rollout_horizon}.npz"
    return filename


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

def calculate_shadow_trajectories(angle_interval, 
                                   cube_angle_range=(-np.pi, np.pi),
                                   goal_angle_range=(-np.pi, np.pi)):
    """
    Calculate total number of trajectories for shadow hand environment.
    
    The shadow hand scans through all combinations of:
    - Initial cube rotation: angle from cube_angle_range on X, Y, or Z axis
    - Goal rotation offset: angle from goal_angle_range on X, Y, or Z axis
    
    Args:
        angle_interval: Step size for angle sampling (radians)
        cube_angle_range: Tuple (min, max) for initial cube rotation angle (radians)
        goal_angle_range: Tuple (min, max) for goal rotation angle offset (radians)
    
    Returns:
        Total number of trajectory combinations
    """
    import numpy as np
    
    # Calculate number of angles for each range
    num_cube_angles = len(np.arange(cube_angle_range[0], 
                                     cube_angle_range[1] + angle_interval/2, 
                                     angle_interval))
    num_goal_angles = len(np.arange(goal_angle_range[0], 
                                     goal_angle_range[1] + angle_interval/2, 
                                     angle_interval))
    
    # 3 axes for cube (X, Y, Z)
    num_cube_axes = 3
    
    # 3 axes for goal (X, Y, Z)
    num_goal_axes = 3
    
    # Total combinations: cube_angles × cube_axes × goal_angles × goal_axes
    total = num_cube_angles * num_cube_axes * num_goal_angles * num_goal_axes
    
    return total

def gen_traj_shadow_hand(
    # Scan through axis-angle space for cube initial orientation and goal orientation
    cube_angle_range=(-np.pi, np.pi),  # Rotation angle for initial cube orientation
    goal_angle_range=(-np.pi, np.pi),  # Rotation angle offset for goal orientation
    angle_interval=0.05,  # Step size for angle sampling (radians)
    rollout_horizon=500,
    opt_steps=1,
    weights=None,
    task_params=None,
    output_dir=None,
    verbose=2,
    orientation_error_threshold=1.0,  # Threshold for successful trajectory
    partition_idx=None,
    num_partitions=None
    ):
    """
    Generate Shadow Hand reorient trajectories by scanning over initial cube orientations
    and goal orientations.
    
    Similar to cartpole and walker, this systematically explores the space of:
    - Initial cube orientations: rotation around X, Y, or Z axis
    - Goal orientations: rotation offset around X, Y, or Z axis
    
    Only successful trajectories (orientation error < threshold) are saved.
    
    Args:
        cube_angle_range: Tuple (min, max) for initial cube rotation angle (radians)
        goal_angle_range: Tuple (min, max) for goal rotation angle offset (radians)
        angle_interval: Step size for scanning angles (radians)
        rollout_horizon: Total length of each trajectory
        opt_steps: Number of optimization steps for MPC
        weights: Cost weights dictionary
        task_params: Task parameters dictionary
        output_dir: Directory to save trajectories (defaults to ../../data/shadow_hand_<dt>/)
        verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
        orientation_error_threshold: Max orientation error for success (radians)
        partition_idx: Index of partition for parallel execution (0-based)
        num_partitions: Total number of partitions for parallel execution
    """
    if weights is None:
        weights = {
            'In Hand': 20.0,        # Distance b/w cube pos and palm pos
            'Orientation': 10.0,    # Orientation err b/w cube and goal
            'Cube Vel.': 10.0,      # Linear velocity of cube
            'Actuator': 0.1,        # Control effort
            'Grasp': 1.0,           # Deviation of hand joints from keyframe
            'Joint Vel.': 1.0e-4,   # Angular velocities of hand joints
        }
    
    if task_params is None:
        task_params = {}
    
    if output_dir is None:
        # Get model timestep for directory naming
        model_path = Path(__file__).parent.parent / "tasks/shadow_reorient/task.xml"
        temp_planner = MPCPlanner(model_path=model_path, task_id="Shadow", 
                                  rollout_horizon=100, opt_steps=1, verbose=0)
        dt = temp_planner.physics_timestep
        output_dir = Path(__file__).parent.parent.parent / f"data/shadow_hand_{dt:.3f}dt".replace(".", "_")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create planner instance with Shadow task - disable randomization, we'll set poses manually
    model_path = Path(__file__).parent.parent / "tasks/shadow_reorient/task.xml"
    planner = MPCPlanner(
        model_path=model_path,
        task_id="Shadow",
        rollout_horizon=rollout_horizon,
        opt_steps=opt_steps,
        weights=weights,
        task_params=task_params,
        init_state_noise_flag=False,  # Disable randomization - we set poses manually
        qpos_noise_rnge=(0.0, 0.0),
        qvel_noise_rnge=(0.0, 0.0),
        verbose=0  # Let this function handle verbosity
    )
    
    # Import quat_mul for quaternion operations
    from gymnasium_robotics.utils.rotations import quat_mul
    
    # Get body and joint addresses for setting cube/goal orientations
    model = planner.model
    data = planner.data
    
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_jnt_addr = model.body_jntadr[cube_body_id]
    cube_qpos_start = model.jnt_qposadr[cube_jnt_addr]
    
    goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
    goal_jnt_addr = model.body_jntadr[goal_body_id]
    goal_qpos_start = model.jnt_qposadr[goal_jnt_addr]
    
    # Generate ranges for angles and axes
    cube_angles = np.arange(cube_angle_range[0], cube_angle_range[1] + angle_interval/2, angle_interval)
    goal_angles = np.arange(goal_angle_range[0], goal_angle_range[1] + angle_interval/2, angle_interval)
    axes = [0, 1, 2]  # X, Y, Z axes
    
    # Calculate total combinations
    total_combinations = len(cube_angles) * len(axes) * len(goal_angles) * len(axes)
    
    if verbose > 0:
        print(f"Generating Shadow Hand trajectories by scanning orientation space:")
        print(f"  Cube angles: {len(cube_angles)} values from {cube_angle_range[0]:.2f} to {cube_angle_range[1]:.2f} rad")
        print(f"  Cube axes: {len(axes)} (X, Y, Z)")
        print(f"  Goal angles: {len(goal_angles)} values from {goal_angle_range[0]:.2f} to {goal_angle_range[1]:.2f} rad")
        print(f"  Goal axes: {len(axes)} (X, Y, Z)")
        print(f"  Total combinations: {total_combinations}")
        print(f"  Angle interval: {angle_interval:.3f} rad ({np.degrees(angle_interval):.1f}°)")
        print(f"  Rollout horizon: {rollout_horizon}")
        print(f"  Orientation error threshold: {orientation_error_threshold:.3f} rad ({np.degrees(orientation_error_threshold):.1f}°)")
        print(f"  Output directory: {output_dir}")
        print()
    
    # Generate all combinations
    all_combinations = list(product(cube_angles, axes, goal_angles, axes))
    
    # Apply partitioning if specified
    if partition_idx is not None and num_partitions is not None:
        if partition_idx >= num_partitions:
            raise ValueError(f"partition_idx ({partition_idx}) must be less than num_partitions ({num_partitions})")
        
        # Split combinations into partitions
        partition_size = len(all_combinations) // num_partitions
        start_idx = partition_idx * partition_size
        
        # Last partition gets any remainder
        if partition_idx == num_partitions - 1:
            end_idx = len(all_combinations)
        else:
            end_idx = start_idx + partition_size
        
        all_combinations = all_combinations[start_idx:end_idx]
        
        if verbose > 0:
            print(f"Running partition {partition_idx + 1}/{num_partitions}")
            print(f"Processing combinations {start_idx + 1} to {end_idx} of {total_combinations}")
            print()
    
    # Generate trajectories for assigned combinations
    traj_count = 0
    successful_count = 0
    axis_names = ['X', 'Y', 'Z']
    
    for cube_angle, cube_axis_idx, goal_angle, goal_axis_idx in all_combinations:
        traj_count += 1
        
        # Reset to grasp keyframe
        keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "grasp")
        if keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        
        # Set initial cube orientation: rotation around specified axis
        cube_axis = np.zeros(3)
        cube_axis[cube_axis_idx] = 1.0
        half_angle = cube_angle / 2
        sin_half = np.sin(half_angle)
        cube_quat = np.array([
            np.cos(half_angle),
            cube_axis[0] * sin_half,
            cube_axis[1] * sin_half,
            cube_axis[2] * sin_half
        ])
        
        # Set cube position (base position from gymnasium_robotics)
        base_pos = np.array([0.325, 0.0, 0.075])
        data.qpos[cube_qpos_start:cube_qpos_start+3] = base_pos
        data.qpos[cube_qpos_start+3:cube_qpos_start+7] = cube_quat
        
        # Set goal orientation: rotation offset from cube initial orientation
        goal_axis = np.zeros(3)
        goal_axis[goal_axis_idx] = 1.0
        half_offset = goal_angle / 2
        sin_half_offset = np.sin(half_offset)
        offset_quat = np.array([
            np.cos(half_offset),
            goal_axis[0] * sin_half_offset,
            goal_axis[1] * sin_half_offset,
            goal_axis[2] * sin_half_offset
        ])
        
        # goal_quat = offset_quat * cube_quat
        goal_quat = quat_mul(offset_quat, cube_quat)
        data.qpos[goal_qpos_start:goal_qpos_start+4] = goal_quat
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        #if verbose > 1:
            #print(f"[{traj_count}/{len(all_combinations)}] Cube: {axis_names[cube_axis_idx]}-axis, {np.degrees(cube_angle):.1f}°  |  Goal: {axis_names[goal_axis_idx]}-axis, {np.degrees(goal_angle):.1f}° offset")
        
        # Plan trajectory from this initial state
        init_qpos = data.qpos.copy()
        init_qvel = data.qvel.copy()
        planner.plan(keyframe="grasp", init_qpos=init_qpos, init_qvel=init_qvel)
        
        # Get trajectories
        qpos, qvel, ctrl, time = planner.get_trajectories()
        cost_total, cost_terms = planner.get_costs()
        
        # Extract final cube orientation
        cube_quat_final = qpos[cube_qpos_start+3:cube_qpos_start+7, -1]
        
        # Calculate orientation error
        cube_quat_final_conj = np.array([cube_quat_final[0], -cube_quat_final[1], 
                                         -cube_quat_final[2], -cube_quat_final[3]])
        q_error = quat_mul(goal_quat, cube_quat_final_conj)
        orientation_error = 2.0 * np.arcsin(np.clip(np.linalg.norm(q_error[1:4]), 0, 1))
        
        # Check if trajectory is successful
        if orientation_error < orientation_error_threshold:
            successful_count += 1
            
            # Generate filename and save
            filename = generate_filename_shadow(cube_quat, goal_quat, rollout_horizon)
            filepath = output_dir / filename
            
            np.savez_compressed(
                filepath,
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                time=time,
                cost_total=cost_total,
                cost_terms=cost_terms,
                cube_quat_init=cube_quat,
                goal_quat=goal_quat,
                orientation_error=orientation_error,
                cube_angle=cube_angle,
                cube_axis=cube_axis_idx,
                goal_angle=goal_angle,
                goal_axis=goal_axis_idx
            )
            
            if verbose > 0:
                print(f"[{traj_count}/{len(all_combinations)}] SUCCESS! Error: {orientation_error:.4f} rad ({np.degrees(orientation_error):.2f}°)")
                #if verbose > 1:
                #    print(f"  Saved to: {filepath}")
        else:
            if verbose > 1:
                print(f"  Failed: error {orientation_error:.4f} rad ({np.degrees(orientation_error):.2f}°) > threshold")
    
    if verbose > 0:
        success_rate = (successful_count / traj_count * 100) if traj_count > 0 else 0
        print(f"\nCompleted! Processed {traj_count} combinations")
        print(f"Successful trajectories: {successful_count} ({success_rate:.1f}%)")
        print(f"Saved to: {output_dir}")


def gen_traj_cartpole(
    qpos0_range=(-0.02, 0.02),  # dm_control: 0.01 * randn() ≈ ±0.02
    qpos1_range=None,  # Will default to (pi-0.02, pi+0.02) for swingup
    qvel0_range=(-0.02, 0.02),  # dm_control: 0.01 * randn() ≈ ±0.02
    qvel1_range=(-0.02, 0.02),  # dm_control: 0.01 * randn() ≈ ±0.02
    interval=0.01,  # Finer interval for small range
    rollout_horizon=10000,
    opt_steps=10,
    weights=None,
    task_params=None,
    output_dir=None,
    verbose=1,
    partition_idx=None,
    num_partitions=None
    ):
    """
    Generate cartpole trajectories by scanning over initial states.
    
    Defaults match dm_control cartpole swingup initialization:
    - Cart position: 0.01 * randn() ≈ ±0.02 range
    - Pole angle: π + 0.01 * randn() ≈ π ± 0.02 (pointing down)
    - Velocities: 0.01 * randn() ≈ ±0.02 range
    
    Args:
        qpos0_range: Tuple (min, max) for qpos[0] (cart position)
        qpos1_range: Tuple (min, max) for qpos[1] (pole angle), defaults to (pi-0.02, pi+0.02)
        qvel0_range: Tuple (min, max) for qvel[0] (cart velocity)
        qvel1_range: Tuple (min, max) for qvel[1] (pole velocity)
        interval: Step size for scanning over the ranges
        rollout_horizon: Total length of each trajectory
        opt_steps: Number of optimization steps for MPC
        weights: Cost weights dictionary
        task_params: Task parameters dictionary
        output_dir: Directory to save trajectories (defaults to ../../data/cartpole_<dt>/)
        verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
        partition_idx: Index of partition for parallel execution (0-based)
        num_partitions: Total number of partitions for parallel execution
    """
    # Set defaults to match dm_control cartpole swingup initialization
    if qpos1_range is None:
        qpos1_range = (np.pi - 0.02, np.pi + 0.02)
    
    if weights is None:
        weights = {
            "Vertical": 10.0,
            "Centered": 10.0,
            "Velocity": 0.1,
            "Control": 0.1
        }
    
    if task_params is None:
        task_params = {"Goal": 0.0}
    
    if output_dir is None:
        # Get model timestep for directory naming
        model_path = Path(__file__).parent.parent / "tasks/cartpole/task.xml"
        temp_planner = MPCPlanner(model_path=model_path, task_id="Cartpole", rollout_horizon=100, opt_steps=1, verbose=0)
        dt = temp_planner.physics_timestep
        output_dir = Path(__file__).parent.parent.parent / f"data/cartpole_{dt:.3f}dt".replace(".", "_")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create planner instance
    planner = MPCPlanner(
        rollout_horizon=rollout_horizon,
        opt_steps=opt_steps,
        weights=weights,
        task_params=task_params,
        init_state_noise_flag=False,
        verbose=0  # Let this function handle verbosity
    )
    
    # Generate ranges for each state variable
    qpos0_vals = np.arange(qpos0_range[0], qpos0_range[1] + interval/2, interval)
    qpos1_vals = np.arange(qpos1_range[0], qpos1_range[1] + interval/2, interval)
    qvel0_vals = np.arange(qvel0_range[0], qvel0_range[1] + interval/2, interval)
    qvel1_vals = np.arange(qvel1_range[0], qvel1_range[1] + interval/2, interval)
    
    # Calculate total number of trajectories
    total_trajs = len(qpos0_vals) * len(qpos1_vals) * len(qvel0_vals) * len(qvel1_vals)
    
    if verbose > 0:
        print(f"Generating {total_trajs} trajectories:")
        print(f"  qpos[0]: {len(qpos0_vals)} values from {qpos0_range[0]:.2f} to {qpos0_range[1]:.2f}")
        print(f"  qpos[1]: {len(qpos1_vals)} values from {qpos1_range[0]:.2f} to {qpos1_range[1]:.2f}")
        print(f"  qvel[0]: {len(qvel0_vals)} values from {qvel0_range[0]:.2f} to {qvel0_range[1]:.2f}")
        print(f"  qvel[1]: {len(qvel1_vals)} values from {qvel1_range[0]:.2f} to {qvel1_range[1]:.2f}")
        print(f"  Interval: {interval:.2f}")
        print(f"  Rollout horizon: {rollout_horizon}")
        print(f"  Output directory: {output_dir}")
        print()
    
    # Generate all combinations
    all_combinations = list(product(qpos0_vals, qpos1_vals, qvel0_vals, qvel1_vals))
    
    # Apply partitioning if specified
    if partition_idx is not None and num_partitions is not None:
        if partition_idx >= num_partitions:
            raise ValueError(f"partition_idx ({partition_idx}) must be less than num_partitions ({num_partitions})")
        
        # Split combinations into partitions
        partition_size = len(all_combinations) // num_partitions
        start_idx = partition_idx * partition_size
        
        # Last partition gets any remainder
        if partition_idx == num_partitions - 1:
            end_idx = len(all_combinations)
        else:
            end_idx = start_idx + partition_size
        
        all_combinations = all_combinations[start_idx:end_idx]
        
        if verbose > 0:
            print(f"Running partition {partition_idx + 1}/{num_partitions}")
            print(f"Processing trajectories {start_idx + 1} to {end_idx} of {total_trajs}")
            print()
    
    # Generate trajectories for assigned combinations
    traj_count = 0
    for qp0, qp1, qv0, qv1 in all_combinations:
        traj_count += 1
        
        # Create initial state
        init_qpos = np.array([qp0, qp1])
        init_qvel = np.array([qv0, qv1])
        
        if verbose > 0:
            total_in_partition = len(all_combinations)
            print(f"[{traj_count}/{total_in_partition}] Planning trajectory with qpos={init_qpos}, qvel={init_qvel}")
        
        # Plan trajectory
        planner.plan(keyframe="home", init_qpos=init_qpos, init_qvel=init_qvel)
        
        # Get trajectories
        qpos, qvel, ctrl, time = planner.get_trajectories()
        cost_total, cost_terms = planner.get_costs()
        
        # Generate filename and save
        filename = generate_filename(init_qpos, init_qvel, rollout_horizon)
        filepath = output_dir / filename
        
        np.savez_compressed(
            filepath,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            time=time,
            cost_total=cost_total,
            cost_terms=cost_terms,
            init_qpos=init_qpos,
            init_qvel=init_qvel
        )
        
        if verbose > 1:
            print(f"  Saved to: {filepath}")
            print(f"  Final qpos: {qpos[:, -1]}")
            print(f"  Total cost: {cost_total.sum():.2f}")
        
    if verbose > 0:
        print(f"\nCompleted! Generated {traj_count} trajectories in {output_dir}")


def gen_traj_walker(
    # Walker has 9 qpos: rootz, rootx, rooty, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle
    # And 9 qvel corresponding to each qpos
    qpos_ranges=None,  # List of 9 tuples (min, max) for each joint position
    qvel_ranges=None,  # List of 9 tuples (min, max) for each joint velocity
    interval=0.1,  # Coarser interval for walker (9D state space)
    rollout_horizon=1000,
    opt_steps=1,
    weights=None,
    task_params=None,
    output_dir=None,
    verbose=1,
    partition_idx=None,
    num_partitions=None
    ):
    """
    Generate walker trajectories by scanning over initial states.
    
    Walker has 9 joints and 9 corresponding velocities. Defaults match dm_control 
    walker initialization:
    - rootz, rootx: Fixed at 0 (not randomized in dm_control)
    - rooty: Uniform in [-pi, pi] (unlimited hinge)
    - right_hip, left_hip: Uniform in [-0.349, 1.745] rad (joint limits)
    - right_knee, left_knee: Uniform in [-2.618, 0] rad (joint limits)
    - right_ankle, left_ankle: Uniform in [-0.785, 0.785] rad (joint limits)
    - All velocities: Fixed at 0 (dm_control initializes to 0)
    
    Args:
        qpos_ranges: List of 9 tuples (min, max) for each joint position
                    Order: rootz, rootx, rooty, right_hip, right_knee, right_ankle, 
                           left_hip, left_knee, left_ankle
                    Defaults to match dm_control walker initialization
        qvel_ranges: List of 9 tuples (min, max) for each joint velocity
                    Defaults to single point at zero (dm_control initialization)
        interval: Step size for scanning over the ranges
        rollout_horizon: Total length of each trajectory
        opt_steps: Number of optimization steps for MPC
        weights: Cost weights dictionary
        task_params: Task parameters dictionary  
        output_dir: Directory to save trajectories (defaults to ../../data/walker_<dt>/)
        verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
        partition_idx: Index of partition for parallel execution (0-based)
        num_partitions: Total number of partitions for parallel execution
    """
    # Set defaults to match dm_control walker initialization
    # Based on randomize_limited_and_rotational_joints() in dm_control:
    # - Unlimited slides (rootz, rootx): NOT randomized, stay at 0
    # - Unlimited hinge (rooty): uniform in [-pi, pi]
    # - Limited hinges: uniform within joint limits
    if qpos_ranges is None:
        qpos_ranges = [
            (0.0, 0.0),                     # rootz - NOT randomized in dm_control
            (0.0, 0.0),                     # rootx - NOT randomized in dm_control
            (-np.pi, np.pi),                # rooty - unlimited hinge: [-pi, pi]
            (-0.34906585, 1.74532925),      # right_hip - limited: [-20, 100] degrees
            (-2.61799388, 0.0),             # right_knee - limited: [-150, 0] degrees
            (-0.78539816, 0.78539816),      # right_ankle - limited: [-45, 45] degrees
            (-0.34906585, 1.74532925),      # left_hip - limited: [-20, 100] degrees
            (-2.61799388, 0.0),             # left_knee - limited: [-150, 0] degrees
            (-0.78539816, 0.78539816),      # left_ankle - limited: [-45, 45] degrees
        ]
    
    if qvel_ranges is None:
        # dm_control initializes all velocities to 0
        qvel_ranges = [(0.0, 0.0)] * 9
    
    if weights is None:
        weights = {
            'Speed': 1.0,
            'Height': 10.0,
            'Rotation': 3.0,
            'Control': 0.1
        }
    
    if task_params is None:
        task_params = {
            'Speed Goal': 1.0,
            'Height Goal': 1.2
        }
    
    if output_dir is None:
        # Get model timestep for directory naming
        model_path = Path(__file__).parent.parent / "tasks/walker/task.xml"
        temp_planner = MPCPlanner(model_path=model_path, task_id="Walker", rollout_horizon=100, opt_steps=1, verbose=0)
        dt = temp_planner.physics_timestep
        output_dir = Path(__file__).parent.parent.parent / f"data/walker_{dt:.4f}dt".replace(".", "_")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create planner instance with walker task
    model_path = Path(__file__).parent.parent / "tasks/walker/task.xml"
    planner = MPCPlanner(
        model_path=model_path,
        task_id="Walker",
        rollout_horizon=rollout_horizon,
        opt_steps=opt_steps,
        weights=weights,
        task_params=task_params,
        init_state_noise_flag=False,
        verbose=0  # Let this function handle verbosity
    )
    
    # Generate ranges for each state variable
    qpos_vals = [np.arange(r[0], r[1] + interval/2, interval) for r in qpos_ranges]
    qvel_vals = [np.arange(r[0], r[1] + interval/2, interval) for r in qvel_ranges]
    
    # Calculate total number of trajectories
    total_trajs = np.prod([len(v) for v in qpos_vals + qvel_vals])
    
    if verbose > 0:
        print(f"Generating walker trajectories:")
        print(f"  Total combinations: {total_trajs:,}")
        for i, (qp_range, vals) in enumerate(zip(qpos_ranges, qpos_vals)):
            joint_names = ["rootz", "rootx", "rooty", "right_hip", "right_knee", 
                          "right_ankle", "left_hip", "left_knee", "left_ankle"]
            print(f"  qpos[{i}] ({joint_names[i]}): {len(vals)} values from {qp_range[0]:.2f} to {qp_range[1]:.2f}")
        for i, (qv_range, vals) in enumerate(zip(qvel_ranges, qvel_vals)):
            print(f"  qvel[{i}]: {len(vals)} values from {qv_range[0]:.2f} to {qv_range[1]:.2f}")
        print(f"  Interval: {interval:.2f}")
        print(f"  Rollout horizon: {rollout_horizon}")
        print(f"  Output directory: {output_dir}")
        print()
    
    # Generate all combinations
    all_combinations = list(product(*qpos_vals, *qvel_vals))
    
    # Apply partitioning if specified
    if partition_idx is not None and num_partitions is not None:
        if partition_idx >= num_partitions:
            raise ValueError(f"partition_idx ({partition_idx}) must be less than num_partitions ({num_partitions})")
        
        # Split combinations into partitions
        partition_size = len(all_combinations) // num_partitions
        start_idx = partition_idx * partition_size
        
        # Last partition gets any remainder
        if partition_idx == num_partitions - 1:
            end_idx = len(all_combinations)
        else:
            end_idx = start_idx + partition_size
        
        all_combinations = all_combinations[start_idx:end_idx]
        
        if verbose > 0:
            print(f"Running partition {partition_idx + 1}/{num_partitions}")
            print(f"Processing trajectories {start_idx + 1} to {end_idx} of {total_trajs}")
            print()
    
    # Generate trajectories for assigned combinations
    traj_count = 0
    for state_combo in all_combinations:
        traj_count += 1
        
        # Split into qpos and qvel (first 9 values are qpos, next 9 are qvel)
        init_qpos = np.array(state_combo[:9])
        init_qvel = np.array(state_combo[9:])
        
        if verbose > 0:
            total_in_partition = len(all_combinations)
            if traj_count % 10 == 0 or traj_count == 1:  # Print every 10th to reduce clutter
                print(f"[{traj_count}/{total_in_partition}] Planning trajectory")
                if verbose > 1:
                    print(f"  qpos={init_qpos}")
                    print(f"  qvel={init_qvel}")
        
        # Plan trajectory
        planner.plan(keyframe="home", init_qpos=init_qpos, init_qvel=init_qvel)
        
        # Get trajectories
        qpos, qvel, ctrl, time = planner.get_trajectories()
        cost_total, cost_terms = planner.get_costs()
        
        # Generate filename and save
        filename = generate_filename(init_qpos, init_qvel, rollout_horizon)
        filepath = output_dir / filename
        
        np.savez_compressed(
            filepath,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            time=time,
            cost_total=cost_total,
            cost_terms=cost_terms,
            init_qpos=init_qpos,
            init_qvel=init_qvel
        )
        
        if verbose > 1:
            print(f"  Saved to: {filepath}")
            print(f"  Final qpos: {qpos[:, -1]}")
            print(f"  Total cost: {cost_total.sum():.2f}")
        
    if verbose > 0:
        print(f"\nCompleted! Generated {traj_count} trajectories in {output_dir}")


def main():
    """
    Main function with default parameters.
    Modify these parameters as needed or use command-line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MPC trajectories with various initial conditions")
    parser.add_argument("--env", type=str, required=True, choices=["cartpole", "walker", "shadow"],
                        help="Environment to generate trajectories for")
    parser.add_argument("--interval", type=float, default=None, 
                        help="Step size for scanning state space (default: 0.01 for cartpole, 0.1 for walker, 0.5 rad for shadow)")
    parser.add_argument("--rollout-horizon", type=int, default=None,
                        help="Trajectory length (default: 10000 for cartpole, 1000 for walker, 500 for shadow)")
    parser.add_argument("--opt-steps", type=int, default=None,
                        help="MPC optimization steps (default: 10 for cartpole, 1 for walker/shadow)")
    parser.add_argument("--orientation-threshold", type=float, default=None,
                        help="Orientation error threshold in radians for shadow (default: 0.1)")
    parser.add_argument("--partition-idx", type=int, default=None,
                        help="Partition index (0-based) for parallel execution")
    parser.add_argument("--num-partitions", type=int, default=None,
                        help="Total number of partitions for parallel execution")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ../../data/<env>_<dt>/)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level (0=quiet, 1=progress, 2=detailed)")
    
    args = parser.parse_args()
    
    # Validate partition arguments
    if (args.partition_idx is not None) != (args.num_partitions is not None):
        parser.error("--partition-idx and --num-partitions must be used together")
    
    # Set environment-specific defaults
    if args.env == "cartpole":
        interval = args.interval if args.interval is not None else 0.01
        rollout_horizon = args.rollout_horizon if args.rollout_horizon is not None else 10000
        opt_steps = args.opt_steps if args.opt_steps is not None else 10
        
        gen_traj_cartpole(
            qpos0_range=(-0.02, 0.02),
            qpos1_range=(np.pi - 0.02, np.pi + 0.02),
            qvel0_range=(-0.02, 0.02),
            qvel1_range=(-0.02, 0.02),
            interval=interval,
            rollout_horizon=rollout_horizon,
            opt_steps=opt_steps,
            output_dir=args.output_dir,
            verbose=args.verbose,
            partition_idx=args.partition_idx,
            num_partitions=args.num_partitions
        )
    
    elif args.env == "walker":
        interval = args.interval if args.interval is not None else 0.1
        rollout_horizon = args.rollout_horizon if args.rollout_horizon is not None else 1000
        opt_steps = args.opt_steps if args.opt_steps is not None else 1
        
        gen_traj_walker(
            qpos_ranges=None,  # Use defaults
            qvel_ranges=None,  # Use defaults
            interval=interval,
            rollout_horizon=rollout_horizon,
            opt_steps=opt_steps,
            output_dir=args.output_dir,
            verbose=args.verbose,
            partition_idx=args.partition_idx,
            num_partitions=args.num_partitions
        )
    
    elif args.env == "shadow":
        angle_interval = args.interval if args.interval is not None else 0.05
        rollout_horizon = args.rollout_horizon if args.rollout_horizon is not None else 500
        opt_steps = args.opt_steps if args.opt_steps is not None else 1
        orientation_threshold = args.orientation_threshold if args.orientation_threshold is not None else 0.7
        
        gen_traj_shadow_hand(
            cube_angle_range=(-np.pi, np.pi),
            goal_angle_range=(-np.pi, np.pi),
            angle_interval=angle_interval,
            rollout_horizon=rollout_horizon,
            opt_steps=opt_steps,
            orientation_error_threshold=orientation_threshold,
            output_dir=args.output_dir,
            verbose=args.verbose,
            partition_idx=args.partition_idx,
            num_partitions=args.num_partitions
        )


if __name__ == "__main__":
    """
    How to Run:

    ===== CARTPOLE =====
    Basic usage (generates trajectories matching dm_control cartpole swingup initialization):
    python mpc_rl/planner/gen_traj_data.py --env cartpole

    With custom interval:
    python mpc_rl/planner/gen_traj_data.py --env cartpole --interval 0.01

    Running in Parallel - Split into 4 partitions (4 terminals):
    
    Terminal 1:
    python mpc_rl/planner/gen_traj_data.py --env cartpole --partition-idx 0 --num-partitions 4

    Terminal 2:
    python mpc_rl/planner/gen_traj_data.py --env cartpole --partition-idx 1 --num-partitions 4
    
    Terminal 3:
    python mpc_rl/planner/gen_traj_data.py --env cartpole --partition-idx 2 --num-partitions 4

    Terminal 4:
    python mpc_rl/planner/gen_traj_data.py --env cartpole --partition-idx 3 --num-partitions 4

    NOTE: Cartpole defaults:
    - Cart position: ±0.02 (0.01 * randn())
    - Pole angle: π ± 0.02 (π + 0.01 * randn())
    - Velocities: ±0.02 each (0.01 * randn())
    - Interval: 0.01 (finer granularity)
    - Rollout horizon: 10000
    - Opt steps: 10

    ===== WALKER =====
    Basic usage:
    python mpc_rl/planner/gen_traj_data.py --env walker

    With custom settings:
    python mpc_rl/planner/gen_traj_data.py --env walker --interval 0.1 --rollout-horizon 1000

    Running in Parallel - Split into 4 partitions:
    
    Terminal 1:
    python mpc_rl/planner/gen_traj_data.py --env walker --partition-idx 0 --num-partitions 4

    Terminal 2:
    python mpc_rl/planner/gen_traj_data.py --env walker --partition-idx 1 --num-partitions 4
    
    Terminal 3:
    python mpc_rl/planner/gen_traj_data.py --env walker --partition-idx 2 --num-partitions 4

    Terminal 4:
    python mpc_rl/planner/gen_traj_data.py --env walker --partition-idx 3 --num-partitions 4

    NOTE: Walker defaults (matching dm_control initialization):
    - rootz, rootx: Fixed at 0 (not randomized)
    - rooty: Uniform in [-π, π] (unlimited rotation)
    - Hip joints: Uniform in [-0.349, 1.745] rad = [-20°, 100°]
    - Knee joints: Uniform in [-2.618, 0] rad = [-150°, 0°]
    - Ankle joints: Uniform in [-0.785, 0.785] rad = [-45°, 45°]
    - All velocities: Fixed at 0 (dm_control initialization)
    - Interval: 0.1 (coarser due to higher dimensionality)
    - Rollout horizon: 1000
    - Opt steps: 1 (faster for walker)

    ===== SHADOW HAND =====
    Basic usage (scan orientation space with default interval):
    python mpc_rl/planner/gen_traj_data.py --env shadow

    With custom angle interval (finer sampling):
    python mpc_rl/planner/gen_traj_data.py --env shadow --interval 0.25

    Adjust success threshold:
    python mpc_rl/planner/gen_traj_data.py --env shadow --orientation-threshold 0.15

    Running in Parallel - Split into 4 partitions:
    
    Terminal 1:
    python mpc_rl/planner/gen_traj_data.py --env shadow --partition-idx 0 --num-partitions 4

    Terminal 2:
    python mpc_rl/planner/gen_traj_data.py --env shadow --partition-idx 1 --num-partitions 4
    
    Terminal 3:
    python mpc_rl/planner/gen_traj_data.py --env shadow --partition-idx 2 --num-partitions 4

    Terminal 4:
    python mpc_rl/planner/gen_traj_data.py --env shadow --partition-idx 3 --num-partitions 4

    NOTE: Shadow Hand defaults:
    - Scans through all combinations of:
      * Initial cube rotation: angle from -π to π on X, Y, or Z axis
      * Goal rotation offset: angle from -π to π on X, Y, or Z axis
    - Only saves successful trajectories (orientation error < threshold)
    - Orientation error threshold: 1.0
    - Rollout horizon: 500
    - Opt steps: 1
    - Filenames contain cube and goal quaternions
    """
    #print(f"Interval 0.05 rad: {calculate_shadow_trajectories(0.05)} trajectories")
    main()