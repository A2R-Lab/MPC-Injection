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
        cmd_vxs = []
        base_vxs = []
        base_heights = []
        vx_abs_errors = []
        active_command_flags = []
        track_lin_vel_terms = []
        track_ang_vel_terms = []
        lin_vel_forward_terms = []
        pose_terms = []
        track_base_height_terms = []
        feet_air_time_terms = []
        feet_clearance_terms = []
        feet_slip_terms = []
        action_rate_terms = []
        torque_abs_means = []

        for info in infos:
            if not isinstance(info, dict):
                continue

            commands = info.get("commands", None)
            base_lin_vel_body = info.get("base_lin_vel_body", None)
            base_height = info.get("base_height", None)
            applied_torques = info.get("applied_torques", None)
            reward_components = info.get("reward_components", None)

            if commands is not None and base_lin_vel_body is not None:
                xy_error = np.sum((commands[:2] - base_lin_vel_body[:2]) ** 2)
                vel_xy_errors.append(float(xy_error))
                cmd_vxs.append(float(commands[0]))
                base_vxs.append(float(base_lin_vel_body[0]))
                vx_abs_errors.append(float(abs(commands[0] - base_lin_vel_body[0])))
                command_magnitude = np.linalg.norm(commands[:2]) + abs(commands[2])
                active_command_flags.append(float(command_magnitude > 0.1))

            if base_height is not None:
                base_heights.append(float(base_height))

            if reward_components is not None and "track_lin_vel" in reward_components:
                track_lin_vel_terms.append(float(reward_components["track_lin_vel"]))

            if reward_components is not None and "track_ang_vel" in reward_components:
                track_ang_vel_terms.append(float(reward_components["track_ang_vel"]))

            if reward_components is not None and "lin_vel_forward" in reward_components:
                lin_vel_forward_terms.append(float(reward_components["lin_vel_forward"]))

            if reward_components is not None and "pose" in reward_components:
                pose_terms.append(float(reward_components["pose"]))

            if reward_components is not None and "track_base_height" in reward_components:
                track_base_height_terms.append(float(reward_components["track_base_height"]))

            if reward_components is not None and "feet_air_time" in reward_components:
                feet_air_time_terms.append(float(reward_components["feet_air_time"]))

            if reward_components is not None and "feet_clearance" in reward_components:
                feet_clearance_terms.append(float(reward_components["feet_clearance"]))

            if reward_components is not None and "feet_slip" in reward_components:
                feet_slip_terms.append(float(reward_components["feet_slip"]))

            if reward_components is not None and "action_rate" in reward_components:
                action_rate_terms.append(float(reward_components["action_rate"]))

            if applied_torques is not None:
                torque_abs_means.append(float(np.mean(np.abs(applied_torques))))

        if self.n_calls % self.log_freq == 0:
            if vel_xy_errors:
                self.logger.record("rollout/vel_xy_error_mean", float(np.mean(vel_xy_errors)))
            if cmd_vxs:
                self.logger.record("rollout/cmd_vx_mean", float(np.mean(cmd_vxs)))
            if base_vxs:
                self.logger.record("rollout/base_vx_mean", float(np.mean(base_vxs)))
            if base_heights:
                self.logger.record("rollout/base_height_mean", float(np.mean(base_heights)))
            if vx_abs_errors:
                self.logger.record("rollout/vx_abs_error_mean", float(np.mean(vx_abs_errors)))
            if active_command_flags:
                self.logger.record("rollout/active_cmd_fraction", float(np.mean(active_command_flags)))
            if track_lin_vel_terms:
                self.logger.record("reward/track_lin_vel", float(np.mean(track_lin_vel_terms)))
            if track_ang_vel_terms:
                self.logger.record("reward/track_ang_vel", float(np.mean(track_ang_vel_terms)))
            if lin_vel_forward_terms:
                self.logger.record("reward/lin_vel_forward", float(np.mean(lin_vel_forward_terms)))
            if pose_terms:
                self.logger.record("reward/pose", float(np.mean(pose_terms)))
            if track_base_height_terms:
                self.logger.record("reward/track_base_height", float(np.mean(track_base_height_terms)))
            if feet_air_time_terms:
                self.logger.record("reward/feet_air_time", float(np.mean(feet_air_time_terms)))
            if feet_clearance_terms:
                self.logger.record("reward/feet_clearance", float(np.mean(feet_clearance_terms)))
            if feet_slip_terms:
                self.logger.record("reward/feet_slip", float(np.mean(feet_slip_terms)))
            if action_rate_terms:
                self.logger.record("reward/action_rate", float(np.mean(action_rate_terms)))
            if torque_abs_means:
                self.logger.record("rollout/torque_abs_mean", float(np.mean(torque_abs_means)))

        return True
