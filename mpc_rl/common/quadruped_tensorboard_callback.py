from __future__ import annotations

from collections import Counter

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

    def __init__(self, log_freq: int = 100, task: str = "velocity_tracking", verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.task = task

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
        flat_orientation_terms = []
        pose_terms = []
        track_base_height_terms = []
        body_ang_vel_terms = []
        angular_momentum_terms = []
        feet_air_time_terms = []
        feet_clearance_terms = []
        feet_slip_terms = []
        action_rate_terms = []
        torque_abs_means = []
        phases = []
        roll_progresses = []
        roll_errors = []
        contact_fractions = []
        stability_counts = []
        success_flags = []
        failure_reasons = Counter()
        roll_tracking_terms = []
        signed_progress_terms = []
        terminal_outcome_terms = []
        action_clip_fractions = []
        torque_saturation_fractions = []

        for info in infos:
            if not isinstance(info, dict):
                continue

            commands = info.get("commands", None)
            base_lin_vel_body = info.get("base_lin_vel_body", None)
            base_height = info.get("base_height", None)
            applied_torques = info.get("applied_torques", None)
            reward_components = info.get("reward_components", None)

            if self.task == "barrel_roll":
                if info.get("phase") is not None:
                    phases.append(float(info["phase"]))
                if info.get("roll_progress") is not None:
                    roll_progresses.append(float(info["roll_progress"]))
                if info.get("roll_error") is not None:
                    roll_errors.append(float(info["roll_error"]))
                if info.get("contact_state") is not None:
                    contact_fractions.append(float(np.mean(info["contact_state"])))
                if info.get("stability_count") is not None:
                    stability_counts.append(float(info["stability_count"]))
                if info.get("is_success") is not None:
                    success_flags.append(float(bool(info["is_success"])))
                if info.get("failure_reason"):
                    failure_reasons[str(info["failure_reason"])] += 1
                if info.get("action_clip_fraction") is not None:
                    action_clip_fractions.append(float(info["action_clip_fraction"]))
                if info.get("torque_saturation_fraction") is not None:
                    torque_saturation_fractions.append(
                        float(info["torque_saturation_fraction"])
                    )

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

            if reward_components is not None and "flat_orientation" in reward_components:
                flat_orientation_terms.append(float(reward_components["flat_orientation"]))

            if reward_components is not None and "pose" in reward_components:
                pose_terms.append(float(reward_components["pose"]))

            if reward_components is not None and "track_base_height" in reward_components:
                track_base_height_terms.append(float(reward_components["track_base_height"]))

            if reward_components is not None and "body_ang_vel" in reward_components:
                body_ang_vel_terms.append(float(reward_components["body_ang_vel"]))

            if reward_components is not None and "angular_momentum" in reward_components:
                angular_momentum_terms.append(float(reward_components["angular_momentum"]))

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
                if self.task == "barrel_roll" and info.get("torque_saturation_fraction") is None:
                    limits = np.tile(np.array([23.7, 23.7, 45.43]), 4)
                    torque_saturation_fractions.append(
                        float(np.mean(np.isclose(np.abs(applied_torques), limits, atol=1.0e-6)))
                    )

            if reward_components is not None:
                if "roll_tracking" in reward_components:
                    roll_tracking_terms.append(float(reward_components["roll_tracking"]))
                if "signed_progress" in reward_components:
                    signed_progress_terms.append(float(reward_components["signed_progress"]))
                if "terminal_outcome" in reward_components:
                    terminal_outcome_terms.append(float(reward_components["terminal_outcome"]))

        if self.task == "barrel_roll" and not action_clip_fractions:
            actions = self.locals.get("actions")
            if actions is not None:
                action_clip_fractions.append(
                    float(np.mean(np.isclose(np.abs(actions), 1.0, atol=1.0e-6)))
                )

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
            if flat_orientation_terms:
                self.logger.record("reward/flat_orientation", float(np.mean(flat_orientation_terms)))
            if pose_terms:
                self.logger.record("reward/pose", float(np.mean(pose_terms)))
            if track_base_height_terms:
                self.logger.record("reward/track_base_height", float(np.mean(track_base_height_terms)))
            if body_ang_vel_terms:
                self.logger.record("reward/body_ang_vel", float(np.mean(body_ang_vel_terms)))
            if angular_momentum_terms:
                self.logger.record("reward/angular_momentum", float(np.mean(angular_momentum_terms)))
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
            if phases:
                self.logger.record("barrel_roll/phase_mean", float(np.mean(phases)))
            if roll_progresses:
                self.logger.record("barrel_roll/progress_mean", float(np.mean(roll_progresses)))
            if roll_errors:
                self.logger.record("barrel_roll/error_mean", float(np.mean(roll_errors)))
            if contact_fractions:
                self.logger.record("barrel_roll/contact_fraction", float(np.mean(contact_fractions)))
            if stability_counts:
                self.logger.record("barrel_roll/stability_count_mean", float(np.mean(stability_counts)))
            if success_flags:
                self.logger.record("barrel_roll/success_fraction", float(np.mean(success_flags)))
            for reason, count in failure_reasons.items():
                safe_reason = reason.replace(":", "_").replace("/", "_")
                self.logger.record(f"barrel_roll/failure_reason/{safe_reason}", float(count))
            if roll_tracking_terms:
                self.logger.record("reward/roll_tracking", float(np.mean(roll_tracking_terms)))
            if signed_progress_terms:
                self.logger.record("reward/signed_progress", float(np.mean(signed_progress_terms)))
            if terminal_outcome_terms:
                self.logger.record("reward/terminal_outcome", float(np.mean(terminal_outcome_terms)))
            if action_clip_fractions:
                self.logger.record(
                    "barrel_roll/action_clip_fraction",
                    float(np.mean(action_clip_fractions)),
                )
            if torque_saturation_fractions:
                self.logger.record(
                    "barrel_roll/torque_saturation_fraction",
                    float(np.mean(torque_saturation_fractions)),
                )
            replay_buffer = getattr(self.model, "replay_buffer", None)
            if replay_buffer is not None and hasattr(replay_buffer, "get_mpc_percentage"):
                self.logger.record(
                    "replay_buffer/mpc_percentage_actual",
                    float(replay_buffer.get_mpc_percentage()),
                )

        return True
