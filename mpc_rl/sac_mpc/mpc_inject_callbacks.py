import random
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation


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
        mpc_planner=None,                     # MPCPlanner instance (not used if loading from file)
        inject_every_n_timesteps: int=10000,  # Inject after every N timesteps
        num_mpc_trajectories: int=1,          # How many MPC rollouts to inject
        data_dir: str=None,                   # Path to directory with saved trajectories
        random_select: bool=True,             # If True, randomly select trajectories from data_dir
        trajectory_files: list=None,          # List of specific filenames to load (used when random_select=False)
        verbose: int=1                        # 0: no output, 1: info msgs, 2: debug msgs
        ):
        super().__init__(verbose)
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
        print(f"  Inject every: {inject_every_n_timesteps} timesteps")
        print(f"  Trajectories per injection: {num_mpc_trajectories}")
        if data_dir:
            print(f"  Loading from: {data_dir}")
            print(f"  Random selection: {random_select}")
        else:
            print(f"  Generating trajectories with MPC planner")
        print(f"  Verbose level: {verbose}\n")
    
    def _select_trajectory_file(self):
        """
        Select a trajectory file to load.
        
        Returns:
            Path to selected trajectory file
        """
        
        if self.random_select:
            # Randomly select from available files
            selected_file = random.choice(self.available_files)
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
        # Get the MPC downsampled control for RL timestep alignment
        # For cartpole: MPC runs at 0.001s, RL acts at 0.01s, so downsample by 10
        downsample_factor = 10
        
        # Create a temporary environment for MPC trajectory generation
        # This avoids corrupting the training environment's state
        # Create a standalone environment (not vectorized)
        dm_env = suite.load(domain_name="cartpole", task_name="swingup")
        temp_env = DmControlCompatibilityV0(dm_env, render_mode=None)
        temp_env = FlattenObservation(temp_env)
        
        # Track actual transitions added in this injection
        total_transitions_added = 0
        
        for traj_idx in range(self.num_mpc_trajectories):
            # Load or generate trajectory
            if self.data_dir is not None:
                # Load from file
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
            
            # Set the environment to the MPC initial state
            # For cartpole swingup, convert qpos/qvel to observation format
            # DM Control obs: [cart_pos, cos(pole_angle), sin(pole_angle), cart_vel, pole_vel]
            cart_pos_init = qpos[0, 0]
            pole_angle_init = qpos[1, 0]
            cart_vel_init = qvel[0, 0]
            pole_vel_init = qvel[1, 0]
            
            # Set the physics state directly
            temp_env.unwrapped._env.physics.data.qpos[:] = qpos[:, 0]
            temp_env.unwrapped._env.physics.data.qvel[:] = qvel[:, 0]
            
            # Get initial observation from environment
            obs = np.array([
                cart_pos_init,
                np.cos(pole_angle_init),
                np.sin(pole_angle_init),
                cart_vel_init,
                pole_vel_init
            ], dtype=np.float32)
            
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

    Unlike FixedMPCInjectCallback which injects a fixed number of trajectories, this callback
    dynamically calculates how many trajectories to inject based on the current replay buffer
    size to maintain a target percentage of MPC data.
    
    The MPC planner generates optimal trajectories which are added to the replay buffer
    to provide high-quality demonstration data that can accelerate learning.
    
    Note: For off-policy algorithms like SAC, injection is triggered based on timesteps
    rather than episodes, since callbacks are called during training updates.
    """
    def __init__(
        self,
        mpc_planner=None,                     # MPCPlanner instance (not used if loading from file)
        inject_every_n_timesteps: int=10000,  # Inject after every N timesteps
        target_percentage: int=25,            # Target percentage of replay buffer to be MPC data (0-100)
        data_dir: str=None,                   # Path to directory with saved trajectories
        random_select: bool=True,             # If True, randomly select trajectories from data_dir
        trajectory_files: list=None,          # List of specific filenames to load (used when random_select=False)
        verbose: int=1                        # 0: no output, 1: info msgs, 2: debug msgs
        ):
        super().__init__(verbose)
        self.mpc_planner = mpc_planner
        self.inject_freq = inject_every_n_timesteps
        self.target_percentage = target_percentage
        self.total_injections = 0
        self.last_injection_timestep = 0
        self.total_mpc_trajectories_injected = 0  # Track total MPC trajectories
        
        # NOTE: We cannot accurately track MPC transitions across injections because
        # the replay buffer evicts old transitions when full. Instead, we inject
        # trajectories until the percentage is met, checking after each trajectory.
        
        # Trajectory loading configuration
        self.data_dir = data_dir
        self.random_select = random_select
        self.trajectory_files = trajectory_files if trajectory_files is not None else []
        self.trajectory_file_idx = 0  # For cycling through specified files
        
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
        print(f"  Inject every: {inject_every_n_timesteps} timesteps")
        print(f"  Target percentage: {target_percentage}%")
        if data_dir:
            print(f"  Loading from: {data_dir}")
            print(f"  Random selection: {random_select}")
        else:
            print(f"  Generating trajectories with MPC planner")
        print(f"  Verbose level: {verbose}\n")
    
    def _select_trajectory_file(self):
        """
        Select a trajectory file to load.
        
        Returns:
            Path to selected trajectory file
        """
        
        if self.random_select:
            # Randomly select from available files
            selected_file = random.choice(self.available_files)
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
        Load or generate and inject MPC trajectories into replay buffer to maintain target percentage.
        
        NOTE: LIMITATION:
        This method injects trajectories until approximately the target percentage is reached.
        However, it cannot maintain an exact percentage over time because:
        1. The replay buffer evicts old transitions when full (FIFO)
        2. We cannot track which evicted transitions were MPC vs RL
        3. Therefore, the actual MPC percentage will fluctuate over time
        
        This callback is best used when the buffer is not yet full, or when you want to
        periodically "top up" the MPC data to approximately the target percentage.
        
        For each trajectory:
        1. Load from file OR run MPC planner to generate optimal control sequence
        2. Extract downsampled controls (to match RL action timestep)
        3. Step through environment using MPC actions to get real rewards
        4. Add transitions to replay buffer
        5. Check if approximate target percentage is reached
        
        Note: We create a temporary environment for MPC trajectory generation
        to avoid modifying the training environment's state.
        """
        # Get the MPC downsampled control for RL timestep alignment
        # For cartpole: MPC runs at 0.001s, RL acts at 0.01s, so downsample by 10
        downsample_factor = 10
        
        # Create a temporary environment for MPC trajectory generation
        # This avoids corrupting the training environment's state
        # Create a standalone environment (not vectorized)
        dm_env = suite.load(domain_name="cartpole", task_name="swingup")
        temp_env = DmControlCompatibilityV0(dm_env, render_mode=None)
        temp_env = FlattenObservation(temp_env)
        
        # Track transitions added in this injection session
        total_transitions_added = 0
        num_trajectories_added = 0
        
        # Get initial buffer state
        initial_buffer_size = self.model.replay_buffer.size()
        
        # Keep injecting trajectories until we estimate the target percentage is reached
        # We use a simple heuristic: transitions_just_added / (buffer_size + transitions_just_added)
        while True:
            buffer_size = self.model.replay_buffer.size()
            
            # Calculate approximate current MPC percentage based on this injection session
            # This is an approximation because we don't know how much MPC data was already in buffer
            if buffer_size == 0:
                current_percentage = 0
                if self.verbose > 1:
                    print(f"  Buffer is empty, injecting first trajectory")
            else:
                # Estimate: assume we're starting from 0% MPC and injecting to target
                # This gives us: target% = added / (initial + added)
                # Solve for how much to add: added = initial * target / (100 - target)
                target_transitions = int((initial_buffer_size * self.target_percentage) / (100 - self.target_percentage))
                
                if self.verbose > 1:
                    print(f"  Initial buffer size: {initial_buffer_size}")
                    print(f"  Current buffer size: {buffer_size}")
                    print(f"  Transitions added so far: {total_transitions_added}")
                    print(f"  Target transitions to add: {target_transitions}")
                    print(f"  Target MPC percentage: {self.target_percentage}%")
                
                # Check if we've added enough
                if total_transitions_added >= target_transitions:
                    estimated_percentage = int((total_transitions_added * 100) / buffer_size)
                    if self.verbose > 0:
                        print(f"  Target reached: added {total_transitions_added} transitions")
                        print(f"  Estimated MPC percentage: ~{estimated_percentage}%")
                        print(f"  Injected {num_trajectories_added} trajectories this session")
                        print(f"  Buffer size: {buffer_size}")
                        print(f"{'='*60}\n")
                    break
            
            # Inject one trajectory
            if self.verbose > 1:
                print(f"  Injecting trajectory {num_trajectories_added + 1}...")
            
            # Track transitions for this single trajectory
            steps_added_this_traj = 0
            
            # Load or generate trajectory
            if self.data_dir is not None:
                # Load from file
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
            
            else:
                # Generate trajectory using MPC planner
                raise NotImplementedError("On-the-fly MPC generation not yet implemented. Please provide data_dir.")
            
            # Set the environment to the MPC initial state
            # For cartpole swingup, convert qpos/qvel to observation format
            # DM Control obs: [cart_pos, cos(pole_angle), sin(pole_angle), cart_vel, pole_vel]
            cart_pos_init = qpos[0, 0]
            pole_angle_init = qpos[1, 0]
            cart_vel_init = qvel[0, 0]
            pole_vel_init = qvel[1, 0]
            
            # Set the physics state directly
            temp_env.unwrapped._env.physics.data.qpos[:] = qpos[:, 0]
            temp_env.unwrapped._env.physics.data.qvel[:] = qvel[:, 0]
            
            # Get initial observation from environment
            obs = np.array([
                cart_pos_init,
                np.cos(pole_angle_init),
                np.sin(pole_angle_init),
                cart_vel_init,
                pole_vel_init
            ], dtype=np.float32)
            
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
            
            # Update tracking counters for this trajectory
            self.total_mpc_trajectories_injected += 1
            total_transitions_added += steps_added_this_traj
            num_trajectories_added += 1
            
            if self.verbose > 2:
                print(f"    Added {steps_added_this_traj} transitions from this trajectory")
                print(f"    Total added this session: {total_transitions_added}")
        
        # Close temporary environment
        temp_env.close()


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