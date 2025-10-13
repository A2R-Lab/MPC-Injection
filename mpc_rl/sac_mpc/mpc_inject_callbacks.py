import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EpisodeMPCInjectCallback(BaseCallback):
    """
    Injects MPC trajectories into replay buffer during training after certain episode counts.'

    TODO: Template up to change!
    """
    def __init__(
        self, 
        mpc_planner,           # TODO: MPC planner trajopt collector instance
        inject_every_n_episodes: int = 10,  # Inject after every N episodes
        num_mpc_trajectories: int = 5,      # How many MPC rollouts to inject
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.mpc_planner = mpc_planner
        self.inject_every_n_episodes = inject_every_n_episodes
        self.num_mpc_trajectories = num_mpc_trajectories
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        """
        # Check if episode just finished
        if self.locals['dones'][0]:  # Episode ended
            self.episode_count += 1
            
            # Check if it's time to inject MPC trajectories
            if self.episode_count % self.inject_every_n_episodes == 0:
                if self.verbose > 0:
                    print(f"\n[Episode {self.episode_count}] Injecting MPC trajectories...")
                
                self._inject_mpc_trajectories()
        
        return True  # Continue training
    
    def _inject_mpc_trajectories(self):
        """
        Generate and inject MPC trajectories into replay buffer.
        """
        env = self.training_env.envs[0]  # Get first environment
        
        for traj_idx in range(self.num_mpc_trajectories):
            # Reset environment to random state
            obs = env.reset()
            
            # Generate MPC trajectory
            for step in range(100):  # TODO: Use episode horizon maybe
                # Get MPC action
                action = self.mpc_planner.plan(obs)
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                
                # Inject into replay buffer
                self.model.replay_buffer.add(
                    obs, 
                    next_obs, 
                    action, 
                    reward, 
                    done, 
                    [info]
                )
                
                if done:
                    break
                
                obs = next_obs
            
            if self.verbose > 0:
                print(f"  Injected MPC trajectory {traj_idx + 1}/{self.num_mpc_trajectories}")
        
        if self.verbose > 0:
            print(f"  Replay buffer size: {self.model.replay_buffer.size()}")


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