"""Statistical calculations for evaluation results."""

import numpy as np
from typing import List, Tuple, Dict


def calculate_stats(
    episode_data: List[Tuple[float, bool, float]], 
    gamma: float = 0.99
) -> Dict[str, float]:
    """
    Calculate statistics from episode data.
    
    Args:
        episode_data: List of (reward, done, value) tuples
        gamma: Discount factor
    
    Returns:
        Dictionary with statistics including:
        - episode_rewards: Mean episode reward
        - episode_std_rewards: Std of episode rewards
        - episode_stderr_rewards: Standard error of episode rewards
        - episode_avg_len: Mean episode length
    """
    episode_rewards = []
    episode_lens = []
    ep_rews = []
    
    for rew, done, value in episode_data:
        ep_rews.append(rew)
        if done:
            episode_rewards.append(sum(ep_rews))
            episode_lens.append(len(ep_rews))
            ep_rews = []
    
    episode_rewards = np.array(episode_rewards, dtype=np.float64)
    episode_lens = np.array(episode_lens, dtype=np.float64)
    
    if len(episode_rewards) == 0:
        return {
            "episode_rewards": 0.0,
            "episode_std_rewards": 0.0,
            "episode_stderr_rewards": 0.0,
            "episode_avg_len": 0.0,
        }
    
    return {
        "episode_rewards": float(np.mean(episode_rewards)),
        "episode_std_rewards": float(np.std(episode_rewards, ddof=1)) if len(episode_rewards) > 1 else 0.0,
        "episode_stderr_rewards": float(np.std(episode_rewards, ddof=1) / np.sqrt(len(episode_rewards))) if len(episode_rewards) > 1 else 0.0,
        "episode_avg_len": float(np.mean(episode_lens)),
        "episode_std_avg_len": float(np.std(episode_lens, ddof=1)) if len(episode_lens) > 1 else 0.0,
    }
