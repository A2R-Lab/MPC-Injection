from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class QuadrupedTensorboardCallback(BaseCallback):
    """
    Logs selected rollout metrics from quadruped env info dicts to TensorBoard.

    Intended for quadruped velocity-tracking environments that expose:
      - info["commands"]
      - info["base_lin_vel_body"]
      - info["applied_torques"]
      - info["reward_components"]
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        vel_xy_errors = []
        track_lin_vel_terms = []
        track_ang_vel_terms = []
        track_base_height_terms = []
        torque_abs_means = []

        for info in infos:
            if not isinstance(info, dict):
                continue

            commands = info.get("commands", None)
            base_lin_vel_body = info.get("base_lin_vel_body", None)
            base_ang_vel_body = info.get("base_ang_vel_body", None)
            base_height = info.get("base_height", None)
            applied_torques = info.get("applied_torques", None)
            reward_components = info.get("reward_components", None)

            if commands is not None and base_lin_vel_body is not None:
                xy_error = np.sum((commands[:2] - base_lin_vel_body[:2]) ** 2)
                vel_xy_errors.append(float(xy_error))

            if reward_components is not None and "track_lin_vel" in reward_components:
                track_lin_vel_terms.append(float(reward_components["track_lin_vel"]))

            if reward_components is not None and "track_ang_vel" in reward_components:
                track_ang_vel_terms.append(float(reward_components["track_ang_vel"]))

            if reward_components is not None and "track_base_height" in reward_components:
                track_base_height_terms.append(float(reward_components["track_base_height"]))

            if applied_torques is not None:
                torque_abs_means.append(float(np.mean(np.abs(applied_torques))))

        if self.n_calls % self.log_freq == 0:
            if vel_xy_errors:
                self.logger.record("rollout/vel_xy_error_mean", float(np.mean(vel_xy_errors)))
            if track_lin_vel_terms:
                self.logger.record("reward/track_lin_vel", float(np.mean(track_lin_vel_terms)))
            if track_ang_vel_terms:
                self.logger.record("reward/track_ang_vel", float(np.mean(track_ang_vel_terms)))
            if track_base_height_terms:
                self.logger.record("reward/track_base_height", float(np.mean(track_base_height_terms)))
            if torque_abs_means:
                self.logger.record("rollout/torque_abs_mean", float(np.mean(torque_abs_means)))

        return True