"""Adapter for SBX (Stable-Baselines3 + JAX) agents to work with reward surfaces."""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import jax
import jax.numpy as jnp
import sys

from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sbx import SAC, TD3, PPO

# Add parent directory to path to import mpc_rl modules
script_dir = Path(__file__).parent
reward_surfaces_root = script_dir.parent
mpc_rl_root = reward_surfaces_root.parent
sys.path.insert(0, str(mpc_rl_root))

from mpc_rl.sac_mpc.sac_mpc import SAC_MPC
from mpc_rl.td3_mpc.td3_mpc import TD3_MPC

# Import custom replay buffer - needed for unpickling saved models
# Models may reference it from either location
from mpc_rl.common.tagged_replay_buffer import TaggedReplayBuffer
# Create alias for backward compatibility with old model saves
import mpc_rl.sac_mpc as sac_mpc_module
import mpc_rl.td3_mpc as td3_mpc_module
sac_mpc_module.tagged_replay_buffer = sys.modules['mpc_rl.common.tagged_replay_buffer']
td3_mpc_module.tagged_replay_buffer = sys.modules['mpc_rl.common.tagged_replay_buffer']
# Also register under the old module path for cloudpickle
sys.modules['mpc_rl.sac_mpc.tagged_replay_buffer'] = sys.modules['mpc_rl.common.tagged_replay_buffer']
sys.modules['mpc_rl.td3_mpc.tagged_replay_buffer'] = sys.modules['mpc_rl.common.tagged_replay_buffer']

# Add reward_surfaces to path for core imports
sys.path.insert(0, str(reward_surfaces_root))
from reward_surfaces.core.evaluator import RewardSurfaceEvaluator


def make_dm_control_env(domain: str, task: str):
    """Create DM Control environment."""
    dm_env = suite.load(domain_name=domain, task_name=task)
    gym_env = DmControlCompatibilityV0(dm_env, render_mode=None)
    gym_env = FlattenObservation(gym_env)
    return gym_env


class SBXRewardSurfaceEvaluator(RewardSurfaceEvaluator):
    """
    Adapter for SBX agents to work with reward surface evaluation.
    
    Handles JAX parameter conversion and DM Control environment setup.
    """
    
    def __init__(
        self,
        model_path: str,
        domain: str,
        task: str,
        vecnormalize_path: Optional[str] = None,
        algorithm: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize SBX evaluator.
        
        Args:
            model_path: Path to saved model (.zip file)
            domain: DM Control domain name (e.g., 'walker')
            task: DM Control task name (e.g., 'walk')
            vecnormalize_path: Optional path to vec_normalize.pkl
            algorithm: One of 'SAC', 'TD3', 'PPO', 'SAC-MPC', 'TD3-MPC', or 'auto'
            seed: Random seed for evaluation
        """
        self.model_path = Path(model_path)
        self.domain = domain
        self.task = task
        self.vecnormalize_path = Path(vecnormalize_path) if vecnormalize_path else None
        self.seed = seed
        
        # Create environment
        self.env = DummyVecEnv([lambda: make_dm_control_env(domain, task)])
        
        # Apply normalization if available
        if self.vecnormalize_path and self.vecnormalize_path.exists():
            self.env = VecNormalize.load(self.vecnormalize_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
            print(f"Loaded VecNormalize from {self.vecnormalize_path}")
        
        # Detect algorithm from model path if auto
        if algorithm == "auto":
            algorithm = self._detect_algorithm()
        
        # Load model
        algo_map = {
            "SAC": SAC,
            "TD3": TD3,
            "PPO": PPO,
            "SAC-MPC": SAC_MPC,
            "TD3-MPC": TD3_MPC,
        }
        
        if algorithm not in algo_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.algo_class = algo_map[algorithm]
        self.model = self.algo_class.load(str(self.model_path), env=self.env)
        print(f"Loaded {algorithm} model from {self.model_path}")
    
    def _detect_algorithm(self) -> str:
        """Detect algorithm from model path or parent directory name."""
        path_str = str(self.model_path.parent.parent)
        
        if "SAC-MPC" in path_str:
            return "SAC-MPC"
        elif "TD3-MPC" in path_str:
            return "TD3-MPC"
        elif "SAC" in path_str:
            return "SAC"
        elif "TD3" in path_str:
            return "TD3"
        elif "PPO" in path_str:
            return "PPO"
        else:
            # Default to SAC for SBX
            print("Warning: Could not detect algorithm, defaulting to SAC")
            return "SAC"
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Extract model parameters as numpy arrays.
        
        For JAX-based models (SBX), converts from JAX arrays to numpy.
        """
        # SBX stores parameters in TrainState objects:
        # self.model.policy.actor_state.params (for actor)
        # self.model.policy.qf_state.params (for critics)
        params_dict = {}
        
        # Get actor parameters
        if hasattr(self.model.policy, 'actor_state'):
            actor_params = self.model.policy.actor_state.params
            params_dict['actor'] = actor_params
        
        # Get critic parameters (for SAC/TD3)
        if hasattr(self.model.policy, 'qf_state'):
            params_dict['qf'] = self.model.policy.qf_state.params
        
        # Flatten all parameters into a list of numpy arrays
        param_list = []
        for network_name, network_params in params_dict.items():
            flat_params = jax.tree_util.tree_leaves(network_params)
            # Convert JAX arrays to numpy
            for param in flat_params:
                if isinstance(param, jnp.ndarray):
                    param_list.append(np.array(param))
                else:
                    param_list.append(param)
        
        return param_list
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays.
        
        Converts numpy arrays back to JAX format.
        """
        # This is tricky for JAX models - need to reconstruct the tree structure
        # We'll iterate through the same structure we used in get_weights
        
        weight_idx = 0
        
        def set_network_params(network_params):
            nonlocal weight_idx
            flat_params = jax.tree_util.tree_leaves(network_params)
            new_flat_params = []
            
            for param in flat_params:
                # Convert numpy back to JAX array
                new_param = jnp.array(weights[weight_idx])
                new_flat_params.append(new_param)
                weight_idx += 1
            
            # Reconstruct tree structure
            return jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(network_params),
                new_flat_params
            )
        
        # Set actor parameters via TrainState.replace()
        if hasattr(self.model.policy, 'actor_state'):
            new_actor_params = set_network_params(self.model.policy.actor_state.params)
            self.model.policy.actor_state = self.model.policy.actor_state.replace(params=new_actor_params)
        
        # Set critic parameters via TrainState.replace()
        if hasattr(self.model.policy, 'qf_state'):
            new_qf_params = set_network_params(self.model.policy.qf_state.params)
            self.model.policy.qf_state = self.model.policy.qf_state.replace(params=new_qf_params)
    
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate the agent for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
        
        Returns:
            Dictionary with evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        episode_rewards = np.array(episode_rewards)
        episode_lengths = np.array(episode_lengths)
        
        return {
            "episode_rewards": float(np.mean(episode_rewards)),
            "episode_std_rewards": float(np.std(episode_rewards, ddof=1)) if len(episode_rewards) > 1 else 0.0,
            "episode_stderr_rewards": float(np.std(episode_rewards, ddof=1) / np.sqrt(len(episode_rewards))) if len(episode_rewards) > 1 else 0.0,
            "episode_avg_len": float(np.mean(episode_lengths)),
            "episode_std_avg_len": float(np.std(episode_lengths, ddof=1)) if len(episode_lengths) > 1 else 0.0,
        }
    
    def __del__(self):
        """Clean up environment."""
        if hasattr(self, 'env'):
            self.env.close()
