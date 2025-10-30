import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, List, Optional, Union
from gymnasium import spaces

class TaggedReplayBuffer(ReplayBuffer):
    """
    Custom replay buffer for off-policy algorithms that tags each transition
    with its source (MPC or RL policy). This enables accurate tracking
    of the buffer composition (percentage) over training.

    Each transition is tagged with:
        0 for RL policy trajectory
        1 for MPC policy trajectory
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tag array for the source of the transition: 0 for RL & 1 for MPC policy
        self.transition_sources = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.uint8
        )

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list[dict[str, Any]],
            source: int = 0, # Default to the RL policy unless the MPC Callback injection used
        ) -> None:
        """
        Add a transition to the buffer with a tag indicating its source.
        
        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            reward: Reward received
            done: Episode done flag
            infos: Additional information
            source: Source of the transition (0 for RL policy, 1 for MPC)
        """
        # Call parent class add method
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Tag the source of this transition at the current position
        # NOTE: pos has already been incremented by the parent class,
        # so we need to tag at (pos - 1) % buffer_size
        tag_pos = (self.pos - 1) % self.buffer_size
        self.transition_sources[tag_pos] = source

    def get_mpc_percentage(self) -> float:
        """
        Calculate the actual percentage of MPC transitions currently in the buffer.
        
        Returns:
            Percentage of MPC transitions (0.0 to 100.0)
        """
        # Only consider filled portion of buffer
        filled_size = self.size()
        if filled_size == 0:
            return 0.0
        
        # Count MPC transitions (source == 1) in the filled portion
        if self.full:
            # Buffer is full, check all positions
            mpc_count = np.sum(self.transition_sources == 1)
        else:
            # Buffer not full yet, only check up to current position
            mpc_count = np.sum(self.transition_sources[:self.pos] == 1)
        
        # Calculate percentage
        total_transitions = filled_size * self.n_envs
        mpc_transitions = mpc_count
        
        return (mpc_transitions / total_transitions) * 100.0
    
    def get_composition_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get detailed statistics about buffer composition.
        
        Returns:
            Dictionary with composition statistics
        """
        filled_size = self.size()
        total_transitions = filled_size * self.n_envs
        
        if filled_size == 0:
            return {
                "total_transitions": 0,
                "mpc_transitions": 0,
                "rl_transitions": 0,
                "mpc_percentage": 0.0,
                "rl_percentage": 0.0,
            }
        
        if self.full:
            mpc_count = np.sum(self.transition_sources == 1)
        else:
            mpc_count = np.sum(self.transition_sources[:self.pos] == 1)
        
        rl_count = total_transitions - mpc_count
        
        return {
            "total_transitions": total_transitions,
            "mpc_transitions": int(mpc_count),
            "rl_transitions": int(rl_count),
            "mpc_percentage": (mpc_count / total_transitions) * 100.0,
            "rl_percentage": (rl_count / total_transitions) * 100.0,
        }

    def reset(self) -> None:
        """
        Reset the buffer, including the transition source tags.
        """
        super().reset()
        self.transition_sources = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.uint8
        )
