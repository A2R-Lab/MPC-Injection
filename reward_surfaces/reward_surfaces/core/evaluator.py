"""Abstract interface for reward surface evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class RewardSurfaceEvaluator(ABC):
    """
    Abstract base class for agents that can be evaluated on reward surfaces.
    
    Your custom RL algorithm should implement this interface to work with
    the reward surface visualization tools.
    """
    
    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """
        Get model parameters as a list of numpy arrays.
        
        Returns:
            List of numpy arrays representing model parameters
        """
        pass
    
    @abstractmethod
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of numpy arrays.
        
        Args:
            weights: List of numpy arrays representing model parameters
        """
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
        
        Returns:
            Dictionary containing evaluation metrics, must include:
            - 'episode_rewards': Mean episode reward
            - 'episode_std_rewards': Std of episode rewards
            - 'episode_stderr_rewards': Standard error of episode rewards
        """
        pass
