"""Observation-history wrappers for quadruped asymmetric actor-critic training."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DictObservationHistoryWrapper(gym.Wrapper):
    """Stack a fixed-length history for 1D Dict observations."""

    def __init__(self, env: gym.Env, history_length: int):
        if history_length < 1:
            raise ValueError(
                f"history_length must be >= 1, got {history_length}"
            )
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                "DictObservationHistoryWrapper requires a Dict observation space, "
                f"got {type(env.observation_space)}"
            )

        super().__init__(env)
        self.history_length = int(history_length)
        self._obs_history: dict[str, deque[np.ndarray]] = {}

        stacked_spaces: dict[str, spaces.Box] = {}
        for key, space in env.observation_space.spaces.items():
            if not isinstance(space, spaces.Box):
                raise TypeError(
                    "DictObservationHistoryWrapper only supports Box subspaces, "
                    f"got {type(space)} for key '{key}'"
                )
            if len(space.shape) != 1:
                raise ValueError(
                    "DictObservationHistoryWrapper only supports 1D Box subspaces, "
                    f"got shape {space.shape} for key '{key}'"
                )
            stacked_spaces[key] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(space.shape[0] * self.history_length,),
                dtype=space.dtype,
            )
            self._obs_history[key] = deque(maxlen=self.history_length)

        self.observation_space = spaces.Dict(stacked_spaces)

    def _copy_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.array(value, copy=True) for key, value in obs.items()}

    def _raw_env_obs(self) -> dict[str, np.ndarray]:
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        return base_env._get_obs()

    def _stack_history(self) -> dict[str, np.ndarray]:
        stacked = {}
        for key, history in self._obs_history.items():
            if len(history) != self.history_length:
                raise RuntimeError("Observation history is not initialized")
            stacked[key] = np.concatenate(list(history), axis=-1).astype(
                self.observation_space[key].dtype,
                copy=False,
            )
        return stacked

    def _set_history(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        copied_obs = self._copy_obs(obs)
        for key, value in copied_obs.items():
            history = self._obs_history[key]
            history.clear()
            for _ in range(self.history_length):
                history.append(np.array(value, copy=True))
        return self._stack_history()

    def _append_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        copied_obs = self._copy_obs(obs)
        for key, value in copied_obs.items():
            self._obs_history[key].append(value)
        return self._stack_history()

    def _replace_latest_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        copied_obs = self._copy_obs(obs)
        for key, value in copied_obs.items():
            history = self._obs_history[key]
            if not history:
                return self._set_history(copied_obs)
            history[-1] = value
        return self._stack_history()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._set_history(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._append_obs(obs), reward, terminated, truncated, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Return the current stacked observation without shifting history."""
        return self._replace_latest_obs(self._raw_env_obs())

    def reset_observation_history(self) -> dict[str, np.ndarray]:
        """Reinitialize the history buffer from the current wrapped env state."""
        return self._set_history(self._raw_env_obs())

    def append_current_observation(self) -> dict[str, np.ndarray]:
        """Append the current wrapped env state as a new history frame."""
        return self._append_obs(self._raw_env_obs())


def maybe_wrap_dict_observation_history(
    env: gym.Env,
    history_length: int,
) -> gym.Env:
    """Wrap Dict observations with history stacking when requested."""
    if history_length <= 1:
        return env
    return DictObservationHistoryWrapper(env, history_length=history_length)
