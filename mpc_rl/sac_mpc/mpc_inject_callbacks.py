import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation


class EpisodeMPCInjectCallback(BaseCallback):
    """
    Injects MPC trajectories into replay buffer during training at regular intervals.
    
    The MPC planner generates optimal trajectories which are added to the replay buffer
    to provide high-quality demonstration data that can accelerate learning.
    
    Note: For off-policy algorithms like SAC, injection is triggered based on timesteps
    rather than episodes, since callbacks are called during training updates.
    """
    def __init__(
        self,
        mpc_planner,                          # MPCPlanner instance for trajectory optimization
        inject_every_n_timesteps: int=10000,  # Inject after every N timesteps
        num_mpc_trajectories: int=1,          # How many MPC rollouts to inject
        verbose: int=1                        # 0: no output, 1: info msgs, 2: debug msgs
    ):
        super().__init__(verbose)
        self.mpc_planner = mpc_planner
        self.inject_freq = inject_every_n_timesteps
        self.num_mpc_trajectories = num_mpc_trajectories
        self.total_injections = 0
        self.last_injection_timestep = 0
        
        # Print initialization info
        print(f"\nMPC Injection Callback initialized:")
        print(f"  Inject every: {inject_every_n_timesteps} timesteps")
        print(f"  Trajectories per injection: {num_mpc_trajectories}")
        print(f"  Verbose level: {verbose}\n")
    
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
        Generate and inject MPC trajectories into replay buffer.
        
        For each trajectory:
        1. Run MPC planner to generate optimal control sequence
        2. Extract downsampled controls (to match RL action timestep)
        3. Step through environment using MPC actions to get real rewards
        4. Add transitions to replay buffer
        
        Note: We create a temporary environment for MPC trajectory generation
        to avoid modifying the training environment's state.
        """
        # Get the MPC downsampled control for RL timestep alignment
        # For cartpole: MPC runs at 0.001s, RL acts at 0.01s, so downsample by 10
        downsample_factor = 10 # TODO: Make this configurable if needed
        
        # Create a temporary environment for MPC trajectory generation
        # This avoids corrupting the training environment's state
        # Create a standalone environment (not vectorized)
        dm_env = suite.load(domain_name="cartpole", task_name="swingup")
        temp_env = DmControlCompatibilityV0(dm_env, render_mode=None)
        temp_env = FlattenObservation(temp_env)
        
        for traj_idx in range(self.num_mpc_trajectories):
            if self.verbose > 1:
                print(f"  Generating MPC trajectory {traj_idx + 1}/{self.num_mpc_trajectories}...")
            
            # Run MPC planner to generate trajectory
            # Enable noise for diversity across trajectories
            self.mpc_planner.set_init_state_noise_flag(True)
            self.mpc_planner.plan(keyframe="home")
            
            # Get trajectories
            qpos, qvel, ctrl, time = self.mpc_planner.get_trajectories()
            
            # Get downsampled control actions to match RL timestep
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
                
                # Update observation for next step
                obs = next_obs
                
                # Stop if episode ended early (shouldn't happen with MPC)
                if done:
                    break
        
        # Close temporary environment
        temp_env.close()
        
        if self.verbose > 0:
            buffer_size = self.model.replay_buffer.size()
            # Calculate actual transitions added (accounting for n_envs replication)
            transitions_added_per_traj = num_steps * self.training_env.num_envs
            total_transitions_added = self.num_mpc_trajectories * transitions_added_per_traj
            print(f"Injected {total_transitions_added} transitions from {self.num_mpc_trajectories} MPC trajectories")
            print(f"  Replay buffer size: {buffer_size}")
            print(f"  Total injections so far: {self.total_injections}")
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