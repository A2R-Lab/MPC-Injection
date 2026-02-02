#!/usr/bin/env python3
"""
Script to load trained SAC-MPC/TD3-MPC models and record the trajectories of each body/limb 
of the walker across different checkpoints.

The recorded trajectories will be saved in model_traj_data subdirectory for later plotting.
"""

import sys
from pathlib import Path
import json
import numpy as np
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path to import from mpc_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpc_rl.sac_mpc.sac_mpc import SAC_MPC
from mpc_rl.td3_mpc.td3_mpc import TD3_MPC


def load_config(run_dir: Path):
    """Load the configuration from config.json"""
    config_path = run_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def make_dm_env(domain: str, task: str, render_mode='rgb_array', seed=None):
    """Create a dm_control environment wrapped for gymnasium."""
    dm_env = suite.load(domain_name=domain, task_name=task)
    gym_env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    gym_env = FlattenObservation(gym_env)
    if seed is not None:
        gym_env.reset(seed=seed)
    return gym_env


def load_model_and_vecnormalize(run_dir: Path, config: dict, checkpoint_step: int, algorithm: str = 'SAC-MPC'):
    """
    Load the trained model and VecNormalize wrapper.
    
    Args:
        run_dir: Path to the run directory
        config: Configuration dictionary
        checkpoint_step: Which checkpoint to load (e.g., 500000)
        algorithm: 'SAC-MPC' or 'TD3-MPC'
    
    Returns:
        Tuple of (model, vec_env)
    """
    # Create the environment
    domain = config['domain']
    task = config['task']

    def env_fn():
        return make_dm_env(domain, task, render_mode=None)
    
    vec_env = DummyVecEnv([env_fn])

    # Load VecNormalize stats
    vecnormalize_path = run_dir / "checkpoints" / f"model_vecnormalize_{checkpoint_step}_steps.pkl"
    model_path = run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip"

    if not vecnormalize_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnormalize_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"  Loading VecNormalize from: {vecnormalize_path.name}")
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)

    # Set VecNormalize to not update stats during evaluation
    vec_env.training = False
    vec_env.norm_reward = False

    # Load the model based on algorithm
    print(f"  Loading {algorithm} model from: {model_path.name}")
    if algorithm.upper() == 'SAC-MPC':
        model = SAC_MPC.load(model_path, env=vec_env)
    elif algorithm.upper() == 'TD3-MPC':
        model = TD3_MPC.load(model_path, env=vec_env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model, vec_env


def run_episode_and_record_trajectories(model, vec_env, body_names: list, max_steps: int = 1000, seed: int = 42):
    """
    Run one episode and collect trajectories of each body part.

    Args:
        model: Trained model
        vec_env: Vectorized environment with normalization
        body_names: List of body part names to track
        max_steps: Maximum number of steps per episode
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
            - 'body_positions': Dict mapping body names to arrays of shape (timesteps, 3) for xyz positions
            - 'body_orientations': Dict mapping body names to arrays of shape (timesteps, 4) for quaternion orientations
            - 'timesteps': Number of timesteps in episode
            - 'rewards': Array of rewards at each timestep
            - 'done': Whether episode terminated early
    """
    # Set seed for reproducibility
    # Must call vec_env.seed() before reset() to properly seed the environment
    vec_env.seed(seed)
    obs = vec_env.reset()
    
    # Access the underlying dm_control physics
    # VecEnv -> DummyVecEnv -> list of envs -> first env -> unwrapped dm_control env
    base_env = vec_env.envs[0].unwrapped
    physics = base_env._env.physics
    
    # Initialize storage
    body_positions = {name: [] for name in body_names}
    body_orientations = {name: [] for name in body_names}
    rewards = []
    observations = []  # Store observations at each timestep
    actions = []  # Store actions at each timestep
    joint_angles = []  # Store joint angles/positions at each timestep
    
    # Track foot-specific data
    foot_clearances = {'right_foot': [], 'left_foot': []}  # Height above ground
    foot_contacts = {'right_foot': [], 'left_foot': []}  # Binary contact state
    torso_height = []  # Torso height above ground
    
    # Get joint names for reference
    joint_names = [physics.model.id2name(i, 'joint') for i in range(physics.model.njnt)]
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # Record the observation (what the policy sees)
        observations.append(obs[0].copy())
        
        # Record body states
        for body_name in body_names:
            # Get position (x, y, z)
            pos = physics.named.data.xpos[body_name].copy()
            body_positions[body_name].append(pos)
            
            # Get orientation as quaternion (w, x, y, z)
            quat = physics.named.data.xquat[body_name].copy()
            body_orientations[body_name].append(quat)
        
        # Record foot clearance (z-coordinate, height above ground)
        foot_clearances['right_foot'].append(physics.named.data.xpos['right_foot'][2])
        foot_clearances['left_foot'].append(physics.named.data.xpos['left_foot'][2])
        
        # Record torso height
        torso_height.append(physics.named.data.xpos['torso'][2])
        
        # Record joint angles/positions
        joint_angles.append(physics.data.qpos.copy())
        
        # Check foot contact with ground
        # In MuJoCo, we check if there are any active contacts involving the foot geoms
        right_foot_in_contact = False
        left_foot_in_contact = False
        
        for i in range(physics.data.ncon):
            contact = physics.data.contact[i]
            geom1_name = physics.model.id2name(contact.geom1, 'geom')
            geom2_name = physics.model.id2name(contact.geom2, 'geom')
            
            # Check if right_foot is in contact with floor
            if (geom1_name == 'right_foot' and geom2_name == 'floor') or \
               (geom2_name == 'right_foot' and geom1_name == 'floor'):
                right_foot_in_contact = True
            
            # Check if left_foot is in contact with floor
            if (geom1_name == 'left_foot' and geom2_name == 'floor') or \
               (geom2_name == 'left_foot' and geom1_name == 'floor'):
                left_foot_in_contact = True
        
        foot_contacts['right_foot'].append(right_foot_in_contact)
        foot_contacts['left_foot'].append(left_foot_in_contact)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0].copy())  # Store the action
        
        # Step environment
        obs, reward, done, info = vec_env.step(action)
        rewards.append(reward[0])
        
        step_count += 1
        done = done[0]
    
    # Convert lists to numpy arrays
    trajectory_data = {
        'body_positions': {name: np.array(positions) for name, positions in body_positions.items()},
        'body_orientations': {name: np.array(orientations) for name, orientations in body_orientations.items()},
        'timesteps': step_count,
        'rewards': np.array(rewards),
        'done': done,
        'seed': seed,
        'observations': np.array(observations),
        'actions': np.array(actions),
        'joint_angles': np.array(joint_angles),
        'joint_names': joint_names,
        'foot_clearances': {name: np.array(clearances) for name, clearances in foot_clearances.items()},
        'foot_contacts': {name: np.array(contacts, dtype=bool) for name, contacts in foot_contacts.items()},
        'torso_height': np.array(torso_height)
    }
    
    return trajectory_data


def save_trajectory_data(trajectory_data: dict, output_dir: Path, checkpoint_step: int):
    """
    Save trajectory data to disk.
    
    Args:
        trajectory_data: Dictionary containing trajectory information
        output_dir: Directory to save the data
        checkpoint_step: Model checkpoint step number
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed numpy file
    output_file = output_dir / f"trajectories_step_{checkpoint_step}.npz"
    
    # Flatten the nested dictionaries for saving
    save_dict = {
        'timesteps': trajectory_data['timesteps'],
        'rewards': trajectory_data['rewards'],
        'done': trajectory_data['done'],
        'seed': trajectory_data['seed'],
        'observations': trajectory_data['observations'],
        'actions': trajectory_data['actions'],
        'joint_angles': trajectory_data['joint_angles'],
        'joint_names': trajectory_data['joint_names']
    }
    
    # Add body positions
    for body_name, positions in trajectory_data['body_positions'].items():
        save_dict[f'pos_{body_name}'] = positions
    
    # Add body orientations
    for body_name, orientations in trajectory_data['body_orientations'].items():
        save_dict[f'quat_{body_name}'] = orientations
    
    # Add foot clearances
    for foot_name, clearances in trajectory_data['foot_clearances'].items():
        save_dict[f'clearance_{foot_name}'] = clearances
    
    # Add foot contacts (binary arrays)
    for foot_name, contacts in trajectory_data['foot_contacts'].items():
        save_dict[f'contact_{foot_name}'] = contacts
        # Calculate and save percentage of time in contact
        contact_percentage = (contacts.sum() / len(contacts)) * 100
        save_dict[f'contact_pct_{foot_name}'] = contact_percentage
    
    # Add torso height data
    save_dict['torso_height'] = trajectory_data['torso_height']
    save_dict['torso_height_mean'] = trajectory_data['torso_height'].mean()
    save_dict['torso_height_std'] = trajectory_data['torso_height'].std()
    
    np.savez_compressed(output_file, **save_dict)
    print(f"  Saved trajectory data to: {output_file}")


def main():
    """
    Main function to iterate through checkpoints and record body part trajectories.
    
    EDIT THESE PARAMETERS:
    """
    # ============================================================================
    # USER CONFIGURATION - Edit these values as needed
    # ============================================================================
    
    # Path to the run directory containing checkpoints
    #RUN_DIR = Path("/home/roy/MPC-RL/logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-112012-percentage-0pct")
    #RUN_DIR = Path("/home/roy/MPC-RL/logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-112659-percentage-25pct")
    RUN_DIR = Path("/home/roy/MPC-RL/logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-113507-percentage-50pct")
    #RUN_DIR = Path("/home/roy/MPC-RL/logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-114318-percentage-75pct")
    #RUN_DIR = Path("/home/roy/MPC-RL/logs/SAC-MPC-walker-velocity_only_reward/3rd_run/walker-walk-SAC-MPC-20260107-115404-percentage-100pct")
    
    # Range of checkpoints to process (inclusive, step by 25000)
    START_CHECKPOINT = 25_000
    END_CHECKPOINT = 500_000
    CHECKPOINT_STEP = 25_000
    
    # Episode parameters
    MAX_STEPS_PER_EPISODE = 1000
    RANDOM_SEED = 500
    
    # Body parts to track (for walker environment)
    BODY_NAMES = ['torso', 'right_thigh', 'right_leg', 'right_foot', 
                  'left_thigh', 'left_leg', 'left_foot']
    
    # Algorithm type ('SAC' or 'TD3')
    ALGORITHM = 'SAC-MPC'
    
    # Output directory (relative to this script)
    OUTPUT_DIR = Path(__file__).parent / "model_traj_data"
    
    # ============================================================================
    # END USER CONFIGURATION
    # ============================================================================
    
    # Load configuration
    print("="*80)
    print("Recording Body Trajectories Across Checkpoints")
    print("="*80)
    print(f"\nRun directory: {RUN_DIR}")
    
    config = load_config(RUN_DIR)
    print(f"Environment: {config['domain']}-{config['task']}")
    print(f"Algorithm: {config['algorithm']}")
    
    # Generate checkpoint range
    checkpoints = list(range(START_CHECKPOINT, END_CHECKPOINT + 1, CHECKPOINT_STEP))
    print(f"\nProcessing {len(checkpoints)} checkpoints: {START_CHECKPOINT} to {END_CHECKPOINT} (step: {CHECKPOINT_STEP})")
    print(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Body parts tracked: {', '.join(BODY_NAMES)}")
    
    # Create output subdirectory with run name
    run_name = RUN_DIR.name
    output_subdir = OUTPUT_DIR / run_name
    print(f"\nOutput directory: {output_subdir}")
    
    # Process each checkpoint
    print("\n" + "="*80)
    for i, checkpoint_step in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Processing checkpoint: {checkpoint_step}")
        print("-"*80)
        
        try:
            # Load model
            model, vec_env = load_model_and_vecnormalize(
                RUN_DIR, config, checkpoint_step, algorithm=ALGORITHM
            )
            
            # Run episode and collect trajectories
            print(f"  Running episode (seed={RANDOM_SEED})...")
            trajectory_data = run_episode_and_record_trajectories(
                model, vec_env, BODY_NAMES, max_steps=MAX_STEPS_PER_EPISODE, seed=RANDOM_SEED
            )
            
            # Print episode summary
            print(f"  Episode completed: {trajectory_data['timesteps']} steps")
            print(f"  Total reward: {trajectory_data['rewards'].sum():.2f}")
            print(f"  Early termination: {trajectory_data['done']}")
            
            # Print foot contact statistics
            right_contact_pct = (trajectory_data['foot_contacts']['right_foot'].sum() / trajectory_data['timesteps']) * 100
            left_contact_pct = (trajectory_data['foot_contacts']['left_foot'].sum() / trajectory_data['timesteps']) * 100
            print(f"  Right foot contact: {right_contact_pct:.1f}%")
            print(f"  Left foot contact: {left_contact_pct:.1f}%")
            
            # Print torso height statistics
            torso_mean_height = trajectory_data['torso_height'].mean()
            torso_std_height = trajectory_data['torso_height'].std()
            print(f"  Torso height: {torso_mean_height:.3f} ± {torso_std_height:.3f} m")
            
            # Save data
            save_trajectory_data(trajectory_data, output_subdir, checkpoint_step)
            
            # Clean up
            vec_env.close()
            
        except FileNotFoundError as e:
            print(f"  WARNING: Skipping checkpoint {checkpoint_step} - {e}")
            continue
        except Exception as e:
            print(f"  ERROR: Failed to process checkpoint {checkpoint_step}")
            print(f"  Error details: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"All trajectory data saved to: {output_subdir}")
    print("="*80)


if __name__ == "__main__":
    main()