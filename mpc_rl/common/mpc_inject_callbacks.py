import numpy as np
import zipfile
from stable_baselines3.common.callbacks import BaseCallback
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import mujoco
import shadow_hand_gym


class FixedMPCInjectCallback(BaseCallback):
    """
    Injects MPC trajectories into replay buffer during training at regular intervals.

    This only injects a fixed number of MPC trajectories every N timesteps. So naturally
    the percentage % of the replay buffer made up of MPC data will decrease over time.
    
    The MPC planner generates optimal trajectories which are added to the replay buffer
    to provide high-quality demonstration data that can accelerate learning.
    
    Note: For off-policy algorithms like SAC, injection is triggered based on timesteps
    rather than episodes, since callbacks are called during training updates.
    """
    def __init__(
        self,
        domain: str,                          # Environment domain (e.g., 'cartpole', 'walker')
        task: str,                            # Environment task (e.g., 'swingup', 'walk')
        mpc_planner=None,                     # MPCPlanner instance (not used if loading from file)
        inject_every_n_timesteps: int=10000,  # Inject after every N timesteps
        num_mpc_trajectories: int=1,          # How many MPC rollouts to inject
        data_dir: str=None,                   # Path to directory with saved trajectories
        random_select: bool=True,             # If True, randomly select trajectories from data_dir
        trajectory_files: list=None,          # List of specific filenames to load (used when random_select=False)
        seed: int=None,                       # Random seed for trajectory selection (for reproducibility)
        verbose: int=1                        # 0: no output, 1: info msgs, 2: debug msgs
        ):
        super().__init__(verbose)
        self.domain = domain
        self.task = task
        self.mpc_planner = mpc_planner
        self.inject_freq = inject_every_n_timesteps
        self.num_mpc_trajectories = num_mpc_trajectories
        self.total_injections = 0
        self.last_injection_timestep = 0
        self.total_mpc_trajectories_injected = 0  # Track total MPC trajectories
        
        # Trajectory loading configuration
        self.data_dir = data_dir
        self.random_select = random_select
        self.trajectory_files = trajectory_files if trajectory_files is not None else []
        self.trajectory_file_idx = 0  # For cycling through specified files
        
        # Store seed for reproducibility
        self.seed = seed
        
        # Seed the random number generator for reproducible trajectory selection
        if seed is not None:
            np.random.seed(seed)
        
        # If data_dir is provided, we'll load trajectories from files
        if data_dir is not None:
            from pathlib import Path
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
            # Get all available trajectory files
            self.available_files = list(self.data_dir.glob("*.npz"))
            if len(self.available_files) == 0:
                raise FileNotFoundError(f"No trajectory files found in {data_dir}")
            
            if verbose > 0:
                print(f"  Found {len(self.available_files)} trajectory files in {data_dir}")
        
        # Print initialization info
        print(f"\nMPC Injection Callback initialized:")
        print(f"  Environment: {domain}/{task}")
        print(f"  Inject every: {inject_every_n_timesteps} timesteps")
        print(f"  Trajectories per injection: {num_mpc_trajectories}")
        if data_dir:
            print(f"  Loading from: {data_dir}")
            print(f"  Random selection: {random_select}")
        else:
            print(f"  Generating trajectories with MPC planner")
        if seed is not None:
            print(f"  Seed: {seed}")
        print(f"  Verbose level: {verbose}\n")
    
    def _select_trajectory_file(self):
        """
        Select a trajectory file to load.
        
        Returns:
            Path to selected trajectory file
        """
        
        if self.random_select:
            # Randomly select from available files
            selected_file = np.random.choice(self.available_files)
            if self.verbose > 1:
                print(f"    Randomly selected: {selected_file.name}")
        else:
            # Cycle through specified trajectory files
            if len(self.trajectory_files) == 0:
                raise ValueError("trajectory_files list is empty. Provide filenames or set random_select=True")
            
            filename = self.trajectory_files[self.trajectory_file_idx]
            selected_file = self.data_dir / filename
            
            if not selected_file.exists():
                raise FileNotFoundError(f"Trajectory file not found: {selected_file}")
            
            if self.verbose > 1:
                print(f"    Selected: {selected_file.name}")
            
            # Cycle to next file for next time
            self.trajectory_file_idx = (self.trajectory_file_idx + 1) % len(self.trajectory_files)
        
        return selected_file
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Checks if it's time to inject MPC trajectories based on timesteps.
        """
        # For off-policy algorithms, trigger based on timesteps
        # Similar to how EvalCallback works
        if self.inject_freq > 0 and self.n_calls % self.inject_freq == 0:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"[Timestep {self.num_timesteps}] Injecting MPC trajectories...")
                print(f"{'='*60}")
            
            self._inject_mpc_trajectories()
            self.total_injections += 1
            self.last_injection_timestep = self.num_timesteps
        
        return True  # Continue training
    
    def _inject_mpc_trajectories(self):
        """
        Load or generate and inject MPC trajectories into replay buffer.
        
        For each trajectory:
        1. Load from file OR run MPC planner to generate optimal control sequence
        2. Extract downsampled controls (to match RL action timestep)
        3. Step through environment using MPC actions to get real rewards
        4. Add transitions to replay buffer
        
        Note: We create a temporary environment for MPC trajectory generation
        to avoid modifying the training environment's state.
        """
        # Determine downsample factor based on environment
        # This should match the ratio of MPC timestep to RL control timestep
        # Cartpole: MPC at 0.01s, RL at 0.01s -> downsample by 1
        # Walker: MPC at 0.0025s, RL at 0.025s -> downsample by 10
        # Shadow Hand: MPC at 0.002s, RL at 0.002s -> downsample by 1
        if self.domain == "cartpole":
            downsample_factor = 1  # MPC and RL both at 0.01s
        elif self.domain == "walker":
            downsample_factor = 10  # MPC at 0.0025s, RL at 0.025s
        elif self.domain == "shadow_hand":
            downsample_factor = 1  # MPC and RL both at 0.002s
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")
        
        # Create a temporary environment for MPC trajectory generation
        # This avoids corrupting the training environment's state
        # Create a standalone environment (not vectorized)
        if self.domain == "shadow_hand":
            # For shadow_hand, task is the full gym env name
            temp_env = gym.make(self.task, render_mode=None)
            temp_env = FlattenObservation(temp_env)
        else:
            # For dm_control environments
            dm_env = suite.load(domain_name=self.domain, task_name=self.task)
            temp_env = DmControlCompatibilityV0(dm_env, render_mode=None)
            temp_env = FlattenObservation(temp_env)
        
        # Seed the temporary environment for reproducibility
        if self.seed is not None:
            temp_env.reset(seed=self.seed)
        
        # Track actual transitions added in this injection
        total_transitions_added = 0
        
        for traj_idx in range(self.num_mpc_trajectories):
            # Load or generate trajectory
            if self.data_dir is not None:
                # Load from file with error handling for corrupted files
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        if self.verbose > 1:
                            print(f"  Loading MPC trajectory {traj_idx + 1}/{self.num_mpc_trajectories}...")
                        
                        traj_file = self._select_trajectory_file()
                        data = np.load(traj_file)
                        qpos = data["qpos"]
                        qvel = data["qvel"]
                        ctrl = data["ctrl"]
                        time = data["time"]
                        
                        if self.verbose > 1:
                            init_qpos = data["init_qpos"]
                            init_qvel = data["init_qvel"]
                            print(f"    Loaded: init_qpos={init_qpos}, init_qvel={init_qvel}")
                        
                        # Successfully loaded, break out of retry loop
                        break
                        
                    except (zipfile.BadZipFile, EOFError, IOError) as e:
                        if retry < max_retries - 1:
                            if self.verbose > 0:
                                print(f"    Warning: Failed to load {traj_file.name}: {e}")
                                print(f"    Retrying with different file ({retry + 1}/{max_retries})...")
                            continue
                        else:
                            # All retries exhausted
                            raise RuntimeError(f"Failed to load valid trajectory file after {max_retries} attempts. "
                                             f"Last error: {e}. Check your data directory for corrupted files.")
            else:
                # Generate with MPC planner
                if self.verbose > 1:
                    print(f"  Generating MPC trajectory {traj_idx + 1}/{self.num_mpc_trajectories}...")
                
                # Run MPC planner to generate trajectory
                # Enable noise for diversity across trajectories
                self.mpc_planner.set_init_state_noise_flag(True)
                self.mpc_planner.plan(keyframe="home")
                
                # Get trajectories
                qpos, qvel, ctrl, time = self.mpc_planner.get_trajectories()
            
            # Get downsampled control actions to match RL timestep
            if self.data_dir is not None:
                # Downsample loaded control directly
                ctrl_downsampled = ctrl[:, ::downsample_factor]
            else:
                # Use planner's downsampling method
                ctrl_downsampled = self.mpc_planner.get_ctrl_downsampled(downsample_factor)
            
            # Reset environment to match MPC initial state
            # Gymnasium API returns (observation, info)
            obs, _ = temp_env.reset()
            
            # Set the environment to the MPC initial state by setting physics directly
            if self.domain == "shadow_hand":
                # For shadow_hand (gymnasium environment)
                temp_env.unwrapped.data.qpos[:] = qpos[:, 0]
                temp_env.unwrapped.data.qvel[:] = qvel[:, 0]
                # Forward the physics to update the observation
                mujoco.mj_forward(temp_env.unwrapped.model, temp_env.unwrapped.data)
                # Get observation from environment
                obs = temp_env.unwrapped._get_obs()
            else:
                # For dm_control environments
                temp_env.unwrapped._env.physics.data.qpos[:] = qpos[:, 0]
                temp_env.unwrapped._env.physics.data.qvel[:] = qvel[:, 0]
                # Forward the physics to update the observation
                temp_env.unwrapped._env.physics.forward()
                # Get initial observation from environment (let the environment compute it)
                obs = temp_env.unwrapped._env.task.get_observation(temp_env.unwrapped._env.physics)
            
            # Flatten the observation if it's a dict
            if isinstance(obs, dict):
                obs = np.concatenate([v.flatten() for v in obs.values()])
            obs = obs.astype(np.float32)
            
            # Step through trajectory using MPC actions
            num_steps = ctrl_downsampled.shape[1]
            steps_added_this_traj = 0
            
            for step in range(num_steps):
                # Get MPC action
                action = ctrl_downsampled[:, step]
                
                # Step environment to get real reward
                # Gymnasium API returns 5 values: (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, info = temp_env.step(action)
                done = terminated or truncated
                
                # Add to replay buffer
                # The replay buffer expects vectorized data (shape for n_envs)
                # Replicate the MPC transition n_envs times to match expected shape
                n_envs = self.training_env.num_envs
                
                # Tile/repeat the same MPC data across all n_envs slots
                obs_vec = np.tile(obs, (n_envs, 1))  # (n_envs, obs_dim)
                next_obs_vec = np.tile(next_obs, (n_envs, 1))  # (n_envs, obs_dim)
                action_vec = np.tile(action, (n_envs, 1))  # (n_envs, action_dim)
                reward_vec = np.full(n_envs, reward)  # (n_envs,)
                done_vec = np.full(n_envs, done)  # (n_envs,)
                info_vec = [info] * n_envs  # list of n_envs infos
                
                # Note: VecNormalize will normalize obs/rewards when sampling
                self.model.replay_buffer.add(
                    obs_vec,
                    next_obs_vec,
                    action_vec,
                    reward_vec,
                    done_vec,
                    info_vec
                )
                
                # Track transitions added (each MPC step = 1 unique transition in buffer)
                # NOTE: Even though we tile/replicate data n_envs times above, the replay
                # buffer stores each transition only once. The vectorization is just for API
                # compatibility with the expected input shape.
                steps_added_this_traj += 1
                
                # Update observation for next step
                obs = next_obs
                
                # Stop if episode ended early (shouldn't happen with MPC)
                if done:
                    break
            
            # Accumulate total transitions for this injection
            total_transitions_added += steps_added_this_traj
        
        # Update total MPC trajectory count
        self.total_mpc_trajectories_injected += self.num_mpc_trajectories
        
        # Close temporary environment
        temp_env.close()
        
        if self.verbose > 0:
            buffer_size = self.model.replay_buffer.size()
            buffer_capacity = self.model.replay_buffer.buffer_size
            
            # Calculate what percentage of buffer is from this injection
            if buffer_size > 0:
                if total_transitions_added >= buffer_size:
                    # We added more than the buffer contains - buffer is entirely (or mostly) MPC data
                    # This can happen on first injection or if buffer was very small
                    mpc_percentage = 100.0
                    print(f"Total MPC trajectories injected: {self.total_mpc_trajectories_injected}")
                    print(f"  Transitions added this injection: {total_transitions_added}")
                    print(f"  MPC data in buffer: ~100% (buffer was smaller than injection)")
                else:
                    # Normal case: MPC is a portion of the buffer
                    mpc_percentage = (total_transitions_added / buffer_size) * 100
                    print(f"Total MPC trajectories injected: {self.total_mpc_trajectories_injected}")
                    print(f"  Transitions added this injection: {total_transitions_added}")
                    print(f"  MPC data in buffer: ~{mpc_percentage:.1f}% of current buffer")
            else:
                print(f"Total MPC trajectories injected: {self.total_mpc_trajectories_injected}")
                print(f"  Transitions added this injection: {total_transitions_added}")
                print(f"  MPC data in buffer: buffer is empty")
            
            print(f"  Replay buffer size: {buffer_size}/{buffer_capacity}")
            print(f"{'='*60}\n")


class PercentMPCInjectCallback(BaseCallback):
    """
    Injects MPC trajectories into replay buffer to maintain a target percentage of MPC data.

    This callback is designed to work with SAC_MPC's train() method, which checks the
    MPC percentage before sampling and calls _inject_mpc_trajectories() if needed.
    
    The callback itself doesn't trigger on timesteps - instead, SAC_MPC calls it
    directly when the MPC percentage falls below the target.
    
    The MPC planner generates optimal trajectories which are added to the replay buffer
    to provide high-quality demonstration data that can accelerate learning.
    """
    def __init__(
        self,
        domain: str,                          # Environment domain (e.g., 'cartpole', 'walker')
        task: str,                            # Environment task (e.g., 'swingup', 'walk')
        target_percentage: int=25,            # Target percentage of replay buffer to be MPC data (0-100)
        data_dir: str=None,                   # Path to directory with saved trajectories
        random_select: bool=True,             # If True, randomly select trajectories from data_dir
        trajectory_files: list=None,          # List of specific filenames to load (used when random_select=False)
        seed: int=None,                       # Random seed for trajectory selection (for reproducibility)
        verbose: int=1                        # 0: no output, 1: info msgs, 2: debug msgs
        ):
        super().__init__(verbose)
        self.domain = domain
        self.task = task
        self.target_percentage = target_percentage
        self.total_mpc_trajectories_injected = 0  # Track total MPC trajectories
        
        # Trajectory loading configuration
        self.data_dir = data_dir
        self.random_select = random_select
        self.trajectory_files = trajectory_files if trajectory_files is not None else []
        self.trajectory_file_idx = 0  # For cycling through specified files
        
        # Store seed for reproducibility
        self.seed = seed
        
        # Seed the random number generator for reproducible trajectory selection
        if seed is not None:
            np.random.seed(seed)
        
        # If data_dir is provided, we'll load trajectories from files
        if data_dir is not None:
            from pathlib import Path
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
            # Get all available trajectory files
            self.available_files = list(self.data_dir.glob("*.npz"))
            if len(self.available_files) == 0:
                raise FileNotFoundError(f"No trajectory files found in {data_dir}")
            
            if verbose > 0:
                print(f"  Found {len(self.available_files)} trajectory files in {data_dir}")
        
        # Print initialization info
        print(f"\nPercent MPC Injection Callback initialized:")
        print(f"  Environment: {domain}/{task}")
        print(f"  Target percentage: {target_percentage}%")
        if data_dir:
            print(f"  Loading from: {data_dir}")
            print(f"  Random selection: {random_select}")
        if seed is not None:
            print(f"  Seed: {seed}")
        print(f"  Verbose level: {verbose}")
        print(f"  Note: Injection triggered by SAC_MPC when MPC% falls below target\n")
    
    def _select_trajectory_file(self):
        """
        Select a trajectory file to load.
        
        Returns:
            Path to selected trajectory file
        """
        
        if self.random_select:
            # Randomly select from available files
            selected_file = np.random.choice(self.available_files)
            if self.verbose > 1:
                print(f"    Randomly selected: {selected_file.name}")
        else:
            # Cycle through specified trajectory files
            if len(self.trajectory_files) == 0:
                raise ValueError("trajectory_files list is empty. Provide filenames or set random_select=True")
            
            filename = self.trajectory_files[self.trajectory_file_idx]
            selected_file = self.data_dir / filename
            
            if not selected_file.exists():
                raise FileNotFoundError(f"Trajectory file not found: {selected_file}")
            
            if self.verbose > 1:
                print(f"    Selected: {selected_file.name}")
            
            # Cycle to next file for next time
            self.trajectory_file_idx = (self.trajectory_file_idx + 1) % len(self.trajectory_files)
        
        return selected_file
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        This callback doesn't inject on a schedule - instead, it's called directly
        by SAC_MPC.train() when the MPC percentage falls below target.
        
        We keep this method to satisfy the BaseCallback interface, but it just
        passes through.
        """
        return True  # Continue training
    
    def _inject_mpc_trajectories(self):
        """
        Load or generate and inject MPC trajectories into replay buffer to maintain target percentage.
        
        This method uses TaggedReplayBuffer to accurately track MPC vs RL transitions and injects
        trajectories one at a time until the target percentage is reached. To avoid overshooting,
        it employs a predictive stopping mechanism:
        
        - Checks actual MPC% from TaggedReplayBuffer before each injection
        - Estimates what the MPC% would be after adding one more trajectory (~1000 steps)
        - Stops if already at target OR if next injection would overshoot by >3% (Hard-coded)
        
        This ensures the MPC percentage stays close to the target (typically within 1-3%) while
        preventing significant overshooting that would occur from blindly adding full trajectories.
        
        For each trajectory:
        1. Load from file OR run MPC planner to generate optimal control sequence
        2. Extract downsampled controls (to match RL action timestep)
        3. Step through environment using MPC actions to get real rewards
        4. Add transitions to replay buffer with source=1 (MPC tag)
        5. Check if target percentage is reached or if next injection would overshoot
        
        Note: We create a temporary environment for MPC trajectory generation
        to avoid modifying the training environment's state.
        """
        # Determine downsample factor based on environment
        # This should match the ratio of MPC timestep to RL control timestep
        # Cartpole: MPC at 0.001s, RL at 0.01s -> downsample by 10
        #           TODO: Try Cartpole data collection at 0.01s to match RL timestep?
        # Walker: MPC at 0.0025s, RL at 0.025s -> downsample by 10
        # Shadow Hand: MPC at 0.002s, RL at 0.002s -> downsample by 1
        # NOTE: This can be calculated/seen from the env_modified.xml and the related
        #       task.xml files for MPC vs the env.py and env.py files for RL in dm_control.
        if self.domain == "cartpole":
            downsample_factor = 10  # MPC at 0.001s, RL at 0.01s
        elif self.domain == "walker":
            downsample_factor = 10  # MPC at 0.0025s, RL at 0.025s
        elif self.domain == "shadow_hand":
            downsample_factor = 1  # MPC and RL both at 0.002s
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")
        
        # Create a temporary environment for MPC trajectory generation
        # This avoids corrupting the training environment's state
        # Create a standalone environment (not vectorized)
        if self.domain == "shadow_hand":
            # For shadow_hand, task is the full gym env name
            temp_env = gym.make(self.task, render_mode=None)
            temp_env = FlattenObservation(temp_env)
        else:
            # For dm_control environments
            dm_env = suite.load(domain_name=self.domain, task_name=self.task)
            temp_env = DmControlCompatibilityV0(dm_env, render_mode=None)
            temp_env = FlattenObservation(temp_env)
        
        # Seed the temporary environment for reproducibility
        if self.seed is not None:
            temp_env.reset(seed=self.seed)
        
        # Track transitions added in this injection session
        total_transitions_added = 0
        num_trajectories_added = 0
        
        # Get initial buffer state
        initial_buffer_size = self.model.replay_buffer.size()
        
        # Keep injecting trajectories until target percentage is reached
        # Use TaggedReplayBuffer's accurate get_mpc_percentage() instead of estimation
        while True:
            # Handle edge cases first
            if self.target_percentage == 0:
                # 0% target means don't inject anything
                if self.verbose > 0:
                    print(f"  Target percentage is 0%, skipping injection")
                    print(f"{'='*60}\n")
                break
            
            # Check actual MPC percentage from TaggedReplayBuffer BEFORE injection
            if hasattr(self.model.replay_buffer, 'get_mpc_percentage'):
                actual_mpc_pct = self.model.replay_buffer.get_mpc_percentage()
            else:
                # Fallback: can't determine percentage without TaggedReplayBuffer
                print("Warning: Cannot determine MPC percentage without TaggedReplayBuffer")
                break
            
            buffer_size = self.model.replay_buffer.size()
            
            if self.target_percentage >= 100:
                # 100% target means fill the entire buffer with MPC
                buffer_capacity = self.model.replay_buffer.buffer_size
                
                # If buffer is full and we're close to 100% (>99%), accept it
                # We can't maintain exactly 100% because RL transitions keep coming and evict MPC
                if buffer_size >= buffer_capacity and actual_mpc_pct >= 99.0:
                    if self.verbose > 0:
                        print(f"  Target reached (buffer full): {actual_mpc_pct:.2f}% MPC")
                        print(f"  Cannot maintain exactly 100% with full buffer and ongoing RL collection")
                        print(f"  Injected {num_trajectories_added} trajectories this session")
                        print(f"  Transitions added: {total_transitions_added}")
                        print(f"  Buffer size: {buffer_size}/{buffer_capacity}")
                        print(f"{'='*60}\n")
                    break
                
                # If buffer not full yet, keep injecting until full or 100%
                if buffer_size < buffer_capacity and actual_mpc_pct >= 100:
                    if self.verbose > 0:
                        print(f"  Target reached: {actual_mpc_pct:.2f}% MPC")
                        print(f"  Injected {num_trajectories_added} trajectories this session")
                        print(f"  Transitions added: {total_transitions_added}")
                        print(f"  Buffer size: {buffer_size}/{buffer_capacity}")
                        print(f"{'='*60}\n")
                    break
            else:
                # Normal case: 0% < target < 100%
                # Check if we've already reached or exceeded target
                # Use a small tolerance to avoid overshooting with large trajectories
                # Since each trajectory is ~1000 steps, we might overshoot by adding one more
                # So we stop if we're within 1% of target OR if we would overshoot significantly
                
                # Calculate what percentage we'd have after adding ~1000 more transitions
                # (rough estimate to decide if we should inject another trajectory)
                if buffer_size > 0:
                    # Get current composition from TaggedReplayBuffer
                    stats = self.model.replay_buffer.get_composition_stats()
                    current_mpc_count = stats["mpc_transitions"]
                    current_total = stats["total_transitions"]
                    
                    # Estimate after adding 1 more trajectory (~1000 transitions)
                    # Note: 1 trajectory = 1000 MPC steps, each added once to buffer
                    estimated_new_mpc = current_mpc_count + 1000
                    estimated_new_total = current_total + 1000
                    estimated_new_pct = (estimated_new_mpc / estimated_new_total) * 100.0
                    
                    # Stop if we're already at target OR if adding one more would overshoot significantly
                    if actual_mpc_pct >= self.target_percentage:
                        if self.verbose > 0:
                            print(f"  Target reached: {actual_mpc_pct:.2f}% >= {self.target_percentage}%")
                            print(f"  Injected {num_trajectories_added} trajectories this session")
                            print(f"  Transitions added: {total_transitions_added}")
                            print(f"  Buffer size: {buffer_size}")
                            print(f"{'='*60}\n")
                        break
                    
                    # Stop if adding one more trajectory would overshoot by too much (>3%)
                    if estimated_new_pct > self.target_percentage + 3.0:
                        if self.verbose > 0:
                            print(f"  Stopping to avoid overshoot:")
                            print(f"    Current: {actual_mpc_pct:.2f}%")
                            print(f"    Target: {self.target_percentage}%")
                            print(f"    Estimated after +1 traj: {estimated_new_pct:.2f}%")
                            print(f"  Injected {num_trajectories_added} trajectories this session")
                            print(f"  Transitions added: {total_transitions_added}")
                            print(f"  Buffer size: {buffer_size}")
                            print(f"{'='*60}\n")
                        break
                else:
                    # Empty buffer, inject at least one trajectory
                    if actual_mpc_pct >= self.target_percentage:
                        if self.verbose > 0:
                            print(f"  Target reached: {actual_mpc_pct:.2f}% >= {self.target_percentage}%")
                            print(f"  Injected {num_trajectories_added} trajectories this session")
                            print(f"  Transitions added: {total_transitions_added}")
                            print(f"  Buffer size: {buffer_size}")
                            print(f"{'='*60}\n")
                        break
                
                if self.verbose > 1:
                    print(f"  Current MPC%: {actual_mpc_pct:.2f}% < Target: {self.target_percentage}%")
                    print(f"  Need to inject more trajectories...")
            
            # Inject one trajectory
            if self.verbose > 1:
                print(f"  Injecting trajectory {num_trajectories_added + 1}...")
            
            # Track transitions for this single trajectory
            steps_added_this_traj = 0
            
            # Load or generate trajectory
            if self.data_dir is not None:
                # Load from file with error handling for corrupted files
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        selected_file = self._select_trajectory_file()
                        
                        # Load the MPC trajectory data
                        traj_data = np.load(selected_file)
                        qpos = traj_data['qpos']  # Shape: (state_dim, num_steps)
                        qvel = traj_data['qvel']
                        ctrl = traj_data['ctrl']  # Shape: (ctrl_dim, num_steps)
                        
                        # Downsample controls to match RL action timestep
                        ctrl_downsampled = ctrl[:, ::downsample_factor]
                        
                        if self.verbose > 2:
                            print(f"    Loaded trajectory: MPC steps={ctrl.shape[1]}, Downsampled steps={ctrl_downsampled.shape[1]}")
                        
                        # Successfully loaded, break out of retry loop
                        break
                        
                    except (zipfile.BadZipFile, EOFError, IOError) as e:
                        if retry < max_retries - 1:
                            if self.verbose > 0:
                                print(f"    Warning: Failed to load {selected_file.name}: {e}")
                                print(f"    Retrying with different file ({retry + 1}/{max_retries})...")
                            continue
                        else:
                            # All retries exhausted
                            raise RuntimeError(f"Failed to load valid trajectory file after {max_retries} attempts. "
                                             f"Last error: {e}. Check your data directory for corrupted files.")
            
            else:
                # Generate trajectory using MPC planner
                raise NotImplementedError("On-the-fly MPC generation not yet implemented. Please provide data_dir.")
            
            # Set the environment to the MPC initial state by setting physics directly
            if self.domain == "shadow_hand":
                # For shadow_hand (gymnasium environment)
                temp_env.unwrapped.data.qpos[:] = qpos[:, 0]
                temp_env.unwrapped.data.qvel[:] = qvel[:, 0]
                # Forward the physics to update the observation
                mujoco.mj_forward(temp_env.unwrapped.model, temp_env.unwrapped.data)
                # Get observation from environment
                obs = temp_env.unwrapped._get_obs()
            else:
                # For dm_control environments
                temp_env.unwrapped._env.physics.data.qpos[:] = qpos[:, 0]
                temp_env.unwrapped._env.physics.data.qvel[:] = qvel[:, 0]
                # Forward the physics to update the observation
                temp_env.unwrapped._env.physics.forward()
                # Get initial observation from environment (let the environment compute it)
                obs = temp_env.unwrapped._env.task.get_observation(temp_env.unwrapped._env.physics)
            
            # Flatten the observation if it's a dict
            if isinstance(obs, dict):
                obs = np.concatenate([v.flatten() for v in obs.values()])
            obs = obs.astype(np.float32)
            
            # Step through trajectory using MPC actions
            num_steps = ctrl_downsampled.shape[1]
            steps_added_this_traj = 0
            
            for step in range(num_steps):
                # Get MPC action
                action = ctrl_downsampled[:, step]
                
                # Step environment to get real reward
                # Gymnasium API returns 5 values: (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, info = temp_env.step(action)
                done = terminated or truncated
                
                # Add to replay buffer
                # The replay buffer expects vectorized data (shape for n_envs)
                # Replicate the MPC transition n_envs times to match expected shape
                n_envs = self.training_env.num_envs
                
                # Tile/repeat the same MPC data across all n_envs slots
                obs_vec = np.tile(obs, (n_envs, 1))  # (n_envs, obs_dim)
                next_obs_vec = np.tile(next_obs, (n_envs, 1))  # (n_envs, obs_dim)
                action_vec = np.tile(action, (n_envs, 1))  # (n_envs, action_dim)
                reward_vec = np.full(n_envs, reward)  # (n_envs,)
                done_vec = np.full(n_envs, done)  # (n_envs,)
                info_vec = [info] * n_envs  # list of n_envs infos
                
                # Note: VecNormalize will normalize obs/rewards when sampling
                self.model.replay_buffer.add(
                    obs=obs_vec,
                    next_obs=next_obs_vec,
                    action=action_vec,
                    reward=reward_vec,
                    done=done_vec,
                    infos=info_vec,
                    source=1  # Mark as MPC source
                )
                
                # Track transitions added (each MPC step = 1 unique transition in buffer)
                # NOTE: Even though we tile/replicate data n_envs times above, the replay
                # buffer stores each transition only once. The vectorization is just for API
                # compatibility with the expected input shape.
                steps_added_this_traj += 1
                
                # Update observation for next step
                obs = next_obs
                
                # Stop if episode ended early (shouldn't happen with MPC)
                if done:
                    break
            
            # Update tracking counters for this trajectory
            self.total_mpc_trajectories_injected += 1
            total_transitions_added += steps_added_this_traj
            num_trajectories_added += 1
            
            if self.verbose > 2:
                print(f"    Added {steps_added_this_traj} transitions from this trajectory")
                print(f"    Total added this session: {total_transitions_added}")
            
            # Check if we've reached target AFTER adding this trajectory
            # This prevents continuing the loop and adding another trajectory when we've already hit target
            if hasattr(self.model.replay_buffer, 'get_mpc_percentage'):
                actual_mpc_pct_after = self.model.replay_buffer.get_mpc_percentage()
                
                # Stop if we've now reached or exceeded the target
                if actual_mpc_pct_after >= self.target_percentage:
                    if self.verbose > 0:
                        print(f"  Target reached after injection: {actual_mpc_pct_after:.2f}% >= {self.target_percentage}%")
                        print(f"  Injected {num_trajectories_added} trajectories this session")
                        print(f"  Transitions added: {total_transitions_added}")
                        print(f"  Buffer size: {self.model.replay_buffer.size()}")
                        print(f"{'='*60}\n")
                    break
        
        # Close temporary environment
        temp_env.close()
        
        # Log actual MPC percentage to TensorBoard if using TaggedReplayBuffer
        if hasattr(self.model.replay_buffer, 'get_mpc_percentage'):
            actual_mpc_pct = self.model.replay_buffer.get_mpc_percentage()
            stats = self.model.replay_buffer.get_composition_stats()
            
            # Log to TensorBoard via the model's logger
            self.logger.record("replay_buffer/mpc_percentage_actual", actual_mpc_pct)
            self.logger.record("replay_buffer/mpc_percentage_target", self.target_percentage)
            self.logger.record("replay_buffer/mpc_transitions", stats["mpc_transitions"])
            self.logger.record("replay_buffer/rl_transitions", stats["rl_transitions"])
            self.logger.record("replay_buffer/total_transitions", stats["total_transitions"])
            
            if self.verbose > 0:
                print(f"\nReplay Buffer Composition:")
                print(f"  Target MPC percentage: {self.target_percentage}%")
                print(f"  Actual MPC percentage: {actual_mpc_pct:.2f}%")
                print(f"  MPC transitions: {stats['mpc_transitions']:,}")
                print(f"  RL transitions: {stats['rl_transitions']:,}")
                print(f"  Total transitions: {stats['total_transitions']:,}")
                print(f"{'='*60}\n")


class AdaptiveMPCInjectCallback(BaseCallback):
    """
    Inject MPC when performance stagnates.
    
    TODO: Template up to change!
    """
    
    def __init__(
        self,
        mpc_planner,
        performance_threshold: float = 0.8,  # Inject if avg reward < this
        check_every_n_episodes: int = 10,
        num_mpc_trajectories: int = 10,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.mpc_planner = mpc_planner
        self.performance_threshold = performance_threshold
        self.check_every_n_episodes = check_every_n_episodes
        self.num_mpc_trajectories = num_mpc_trajectories
        self.episode_count = 0
        self.recent_rewards = []
    
    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            # Track episode reward
            episode_reward = self.locals.get('episode_reward', 0)
            self.recent_rewards.append(episode_reward)
            self.episode_count += 1
            
            # Check performance periodically
            if self.episode_count % self.check_every_n_episodes == 0:
                avg_reward = np.mean(self.recent_rewards[-10:])  # Last 10 episodes
                
                if avg_reward < self.performance_threshold:
                    if self.verbose > 0:
                        print(f"\n[Episode {self.episode_count}] "
                              f"Performance low (avg={avg_reward:.2f}), injecting MPC...")
                    
                    self._inject_mpc_trajectories()
                
                self.recent_rewards = []  # Reset
        
        return True
    
    def _inject_mpc_trajectories(self):
        # Same as above
        pass