import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeMPCInjectCallback(BaseCallback):
    """
    Injects MPC trajectories into replay buffer during training after certain episode counts.
    
    The MPC planner generates optimal trajectories which are added to the replay buffer
    to provide high-quality demonstration data that can accelerate learning.
    """
    def __init__(
        self,
        mpc_planner,                        # MPCPlanner instance for trajectory optimization
        inject_every_n_episodes: int = 1000,  # Inject after every N episodes
        num_mpc_trajectories: int = 100,      # How many MPC rollouts to inject
        verbose: int = 0                      # 0: no output, 1: info msgs, 2: debug msgs
    ):
        super().__init__(verbose)
        self.mpc_planner = mpc_planner
        self.inject_every_n_episodes = inject_every_n_episodes
        self.num_mpc_trajectories = num_mpc_trajectories
        self.episode_count = 0
        self.total_injections = 0
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Checks if episode ended and if it's time to inject MPC trajectories.
        """
        # Check if any episode just finished (works with vectorized envs)
        dones = self.locals.get('dones', [False])
        
        if any(dones):  # At least one episode ended
            self.episode_count += 1
            
            # Check if it's time to inject MPC trajectories
            if self.episode_count % self.inject_every_n_episodes == 0:
                if self.verbose > 0:
                    print(f"\n{'='*50}")
                    print(f"[Episode {self.episode_count}] Injecting MPC trajectories...")
                    print(f"{'='*50}")
                
                self._inject_mpc_trajectories()
                self.total_injections += 1
        
        return True  # Continue training
    
    def _inject_mpc_trajectories(self):
        """
        Generate and inject MPC trajectories into replay buffer.
        
        For each trajectory:
        1. Run MPC planner to generate optimal control sequence
        2. Extract downsampled controls (to match RL action timestep)
        3. Extract state trajectory
        4. Compute rewards (using environment reward function)
        5. Add transitions to replay buffer
        """
        # Get the MPC downsampled control for RL timestep alignment
        # For cartpole: MPC runs at 0.001s, RL acts at 0.01s, so downsample by 10
        downsample_factor = 10
        
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
            
            # For cartpole swingup, we need to convert qpos/qvel to observation format
            # DM Control obs: [cart_pos, cos(pole_angle), sin(pole_angle), cart_vel, pole_vel]
            num_steps = ctrl_downsampled.shape[1]
            
            for step in range(num_steps):
                # Get state at current and next timestep (downsampled)
                idx_current = step * downsample_factor
                idx_next = min((step + 1) * downsample_factor, qpos.shape[1] - 1)
                
                # Current observation
                cart_pos = qpos[0, idx_current]
                pole_angle = qpos[1, idx_current]
                cart_vel = qvel[0, idx_current]
                pole_vel = qvel[1, idx_current]
                obs = np.array([
                    cart_pos,
                    np.cos(pole_angle),
                    np.sin(pole_angle),
                    cart_vel,
                    pole_vel
                ], dtype=np.float32)
                
                # Next observation
                cart_pos_next = qpos[0, idx_next]
                pole_angle_next = qpos[1, idx_next]
                cart_vel_next = qvel[0, idx_next]
                pole_vel_next = qvel[1, idx_next]
                next_obs = np.array([
                    cart_pos_next,
                    np.cos(pole_angle_next),
                    np.sin(pole_angle_next),
                    cart_vel_next,
                    pole_vel_next
                ], dtype=np.float32)
                
                # Action
                action = ctrl_downsampled[:, step]
                
                # Compute reward (cartpole swingup reward)
                # Reward for verticality (pole upright) and centeredness (cart at center)
                upright = (np.cos(pole_angle_next) + 1) / 2  # 0 to 1, 1 when upright
                centered = np.exp(-cart_pos_next**2 / 2)     # ~1 when centered
                small_control = np.exp(-0.1 * action[0]**2)  # Penalize large controls
                reward = upright * centered * small_control
                
                # Episode done when reaching end of trajectory
                done = (step == num_steps - 1)
                
                # Add to replay buffer
                # Note: VecNormalize will normalize obs/rewards when sampling
                self.model.replay_buffer.add(
                    obs,
                    next_obs,
                    action,
                    reward,
                    done,
                    [{}]  # Empty info dict
                )
        
        if self.verbose > 0:
            buffer_size = self.model.replay_buffer.size()
            transitions_added = self.num_mpc_trajectories * (num_steps)
            print(f"Injected {transitions_added} transitions from {self.num_mpc_trajectories} MPC trajectories")
            print(f"  Replay buffer size: {buffer_size}")
            print(f"  Total injections so far: {self.total_injections}")
            print(f"{'='*70}\n")


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