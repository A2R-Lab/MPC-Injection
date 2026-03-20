import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer
from typing import Any, Dict, Union


class TaggedDictReplayBuffer(DictReplayBuffer):
    """
    Dict-observation replay buffer that tags each transition with its source
    (MPC or RL policy). Extends SB3's DictReplayBuffer for environments with
    Dict observation spaces (e.g., quadruped asymmetric actor-critic).

    Each transition is tagged with:
        0 for RL policy trajectory
        1 for MPC policy trajectory
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tag array: 0 = RL policy, 1 = MPC policy
        self.transition_sources = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.uint8
        )

    def add(self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list[dict[str, Any]],
            source: int = 0,
        ) -> None:
        """
        Add a transition to the buffer with a source tag.

        Args:
            obs: Current observation dict
            next_obs: Next observation dict
            action: Action taken
            reward: Reward received
            done: Episode done flag
            infos: Additional information
            source: 0 for RL policy, 1 for MPC
        """
        super().add(obs, next_obs, action, reward, done, infos)

        # Tag the transition (pos already incremented by parent)
        tag_pos = (self.pos - 1) % self.buffer_size
        self.transition_sources[tag_pos] = source

    def get_mpc_percentage(self) -> float:
        """Calculate the percentage of MPC transitions in the buffer."""
        filled_size = self.size()
        if filled_size == 0:
            return 0.0

        if self.full:
            mpc_count = np.sum(self.transition_sources == 1)
        else:
            mpc_count = np.sum(self.transition_sources[:self.pos] == 1)

        total_transitions = filled_size * self.n_envs
        return (mpc_count / total_transitions) * 100.0

    def get_composition_stats(self) -> Dict[str, Union[int, float]]:
        """Get detailed buffer composition statistics."""
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
        """Reset the buffer including source tags."""
        super().reset()
        self.transition_sources = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.uint8
        )
