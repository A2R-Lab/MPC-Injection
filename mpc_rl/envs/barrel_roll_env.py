"""Fixed-direction, one-shot Go2 barrel-roll Gymnasium environment."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from mpc_rl.envs.barrel_roll_common import (
    ACTION_SCALE,
    CONTROL_DT,
    CONTROL_STEPS,
    FINAL_HOLD_CONTROL_STEPS,
    FINAL_HOLD_START_TIME,
    HOLD_CONDITION_PRECEDENCE,
    REWARD_CONFIG,
    ROLL_DIRECTION_SIGN,
    ROLL_END_TIME,
    ROLL_START_TIME,
    SUCCESS_CONFIG,
    RollProgressTracker,
    assert_valid_initial_state,
    body_up_tilt_from_quaternion,
    desired_roll_at_time,
    final_hold_conditions,
    find_nonfoot_ground_contact,
    maneuver_phase_at_time,
    sample_symmetric_hip_spread,
    standing_subscores,
    terminal_failure_reason,
)
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv


class QuadrupedBarrelRollEnv(QuadrupedVelocityTrackingEnv):
    """Nominal Go2 task with a 45D actor and 4D privileged critic input."""

    def __init__(
        self,
        *,
        robot: str = "go2",
        domain_rand_cfg=None,
        use_go2_sysid: bool = True,
        action_scale: float = ACTION_SCALE,
        **kwargs,
    ):
        if robot.lower() != "go2":
            raise ValueError("QuadrupedBarrelRollEnv supports robot='go2' only")
        if not use_go2_sysid:
            raise ValueError("barrel-roll requires use_go2_sysid=True")
        if domain_rand_cfg is not None and domain_rand_cfg.enable:
            raise ValueError("barrel-roll does not support domain randomization")
        if not np.isclose(action_scale, ACTION_SCALE, atol=0.0, rtol=0.0):
            raise ValueError(
                f"barrel-roll schema-v4 requires action_scale={ACTION_SCALE}, "
                f"got {action_scale}"
            )
        kwargs.pop("simple_reward", None)
        super().__init__(
            robot="go2",
            scene="flat",
            use_go2_sysid=True,
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
            apply_startup_domain_rand_on_init=False,
            sim_dt=0.005,
            decimation=4,
            action_scale=ACTION_SCALE,
            **kwargs,
        )
        if not np.isclose(self.control_dt, CONTROL_DT):
            raise ValueError(f"barrel-roll requires control_dt={CONTROL_DT}, got {self.control_dt}")
        # The MPX Go2 execution model uses pyramidal friction cones.  Keep this
        # task-specific plant on the same contact convention; the shared Gym
        # scene otherwise defaults to elliptic cones and diverges from the first
        # stance command under an identical torque trace.
        self.mjModel.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.policy_obs_dim = 45
        self.privileged_obs_dim = 4
        self.observation_space = spaces.Dict({
            "policy": spaces.Box(-np.inf, np.inf, (45,), dtype=np.float64),
            "privileged": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float64),
        })
        self._spread = 0.0
        self._roll_tracker = RollProgressTracker.from_reset_quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
        self._roll_progress = 0.0
        self._previous_roll_progress = 0.0
        self._physics_failure_reason: str | None = None
        self._failure_reason: str | None = None
        self._stability_count = 0
        self._final_hold_streak = 0
        self._final_hold_window: deque[dict[str, bool]] = deque(
            maxlen=FINAL_HOLD_CONTROL_STEPS
        )
        self._hold_condition_failure_counts = Counter()
        self._hold_conditions = {
            name: False for name in HOLD_CONDITION_PRECEDENCE
        }
        self._hold_metrics = {
            "base_height": float("nan"),
            "body_up_tilt": float("nan"),
            "base_linear_speed": float("nan"),
            "base_angular_speed": float("nan"),
            "joint_velocity_norm": float("nan"),
        }
        self._raw_barrel_contacts = np.zeros(4, dtype=bool)
        self._previous_barrel_raw_contacts = np.zeros(4, dtype=bool)
        self._filtered_barrel_contacts = np.zeros(4, dtype=bool)
        self._last_nonfoot_contact: str | None = None
        self._nonfoot_contact_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetDataKeyframe(self.mjModel, self.mjData, 0)
        if self.robot_cfg.qpos0_js is not None:
            self.mjData.qpos[7:] = np.asarray(self.robot_cfg.qpos0_js, dtype=np.float64)
        requested_spread = None if options is None else options.get("spread")
        self.mjData.qpos[:], self._spread = sample_symmetric_hip_spread(
            self.mjModel,
            self.mjData.qpos,
            self.np_random,
            spread=requested_spread,
        )
        self.mjData.qvel[:] = 0.0
        self.mjData.qacc[:] = 0.0
        self.mjData.ctrl[:] = 0.0
        mujoco.mj_forward(self.mjModel, self.mjData)
        assert_valid_initial_state(self)
        self._last_action[:] = 0.0
        self._prev_last_action[:] = 0.0
        self._raw_q_target = self.default_joint_pos.copy()
        self._filtered_q_target = self.default_joint_pos.copy()
        self._applied_torques[:] = 0.0
        self._step_count = 0
        self._steps_since_command_resample = 0
        self._roll_tracker = RollProgressTracker.from_reset_quaternion(self.mjData.qpos[3:7])
        self._roll_progress = 0.0
        self._previous_roll_progress = 0.0
        self._physics_failure_reason = None
        self._failure_reason = None
        self._stability_count = 0
        self._final_hold_streak = 0
        self._final_hold_window.clear()
        self._hold_condition_failure_counts.clear()
        self._last_foot_contacts[:] = False
        self._current_contacts[:] = False
        initial_contacts = self._get_foot_contacts()
        self._raw_barrel_contacts = initial_contacts.copy()
        self._previous_barrel_raw_contacts = initial_contacts.copy()
        self._filtered_barrel_contacts = initial_contacts.copy()
        self._last_nonfoot_contact = None
        self._nonfoot_contact_count = 0
        self._reward_components = {}
        self._refresh_hold_classification(advance_streak=False)
        obs = self._get_obs()
        return obs, self._get_info()

    def _after_physics_substep(self) -> None:
        if (
            not np.isfinite(self.mjData.qpos).all()
            or not np.isfinite(self.mjData.qvel).all()
            or not np.isfinite(self.mjData.ctrl).all()
        ):
            self._physics_failure_reason = "non_finite_state"
            return
        try:
            self._roll_progress = self._roll_tracker.update(self.mjData.qpos[3:7])
        except ValueError:
            self._physics_failure_reason = "non_finite_state"
            return
        nonfoot = find_nonfoot_ground_contact(self)
        if nonfoot is not None:
            self._last_nonfoot_contact = nonfoot
            self._nonfoot_contact_count += 1
            self._physics_failure_reason = "non_foot_ground_contact"

    def _physics_substeps_should_stop(self) -> bool:
        return self._physics_failure_reason is not None

    def _before_control_step(self) -> None:
        self._previous_roll_progress = self._roll_progress

    def _get_obs(self) -> dict[str, np.ndarray]:
        base_ang_vel = self.mjData.qvel[3:6].copy()
        phase_time = self._step_count * self.control_dt
        desired_roll = desired_roll_at_time(phase_time)
        policy = np.concatenate((
            base_ang_vel,
            self._projected_gravity(),
            np.array([maneuver_phase_at_time(phase_time), np.sin(desired_roll), np.cos(desired_roll)]),
            self.mjData.qpos[7:] - self.default_joint_pos,
            self.mjData.qvel[6:],
            self._last_action,
        )).astype(np.float64)
        privileged = np.concatenate((self._base_lin_vel_body(), np.array([self._roll_progress]))).astype(np.float64)
        return {"policy": policy, "privileged": privileged}

    def _upright_errors(self) -> tuple[float, float]:
        rotation = Rotation.from_quat(np.roll(self.mjData.qpos[3:7], -1))
        roll, pitch, _ = rotation.as_euler("xyz")
        return float(roll), float(pitch)

    def _refresh_hold_classification(self, *, advance_streak: bool) -> None:
        quaternion = self.mjData.qpos[3:7]
        tilt = (
            body_up_tilt_from_quaternion(quaternion)
            if np.isfinite(quaternion).all() and np.linalg.norm(quaternion) > 0.0
            else float("nan")
        )
        self._hold_metrics = {
            "base_height": float(self.mjData.qpos[2]),
            "body_up_tilt": tilt,
            "base_linear_speed": float(np.linalg.norm(self.mjData.qvel[:3])),
            "base_angular_speed": float(np.linalg.norm(self.mjData.qvel[3:6])),
            "joint_velocity_norm": float(np.linalg.norm(self.mjData.qvel[6:])),
        }
        self._hold_conditions = final_hold_conditions(
            filtered_foot_contacts=self._filtered_barrel_contacts,
            roll_progress=self._roll_progress,
            **self._hold_metrics,
        )
        if not advance_streak:
            return
        self._final_hold_window.append(self._hold_conditions.copy())
        hold_valid = all(self._hold_conditions.values())
        self._final_hold_streak = self._final_hold_streak + 1 if hold_valid else 0
        self._hold_condition_failure_counts.update(
            name for name, valid in self._hold_conditions.items() if not valid
        )

    def _final_window_failure_counts(self) -> dict[str, int]:
        return {
            name: sum(not endpoint[name] for endpoint in self._final_hold_window)
            for name in HOLD_CONDITION_PRECEDENCE
        }

    def _terminal_success(self) -> bool:
        return bool(
            self._step_count >= CONTROL_STEPS
            and self._physics_failure_reason is None
            and self._failure_reason is None
            and self._final_hold_streak >= SUCCESS_CONFIG.final_hold_control_steps
        )

    def _check_termination(self) -> bool:
        contacts = self._get_foot_contacts()
        self._raw_barrel_contacts = contacts.copy()
        self._stability_count = self._stability_count + 1 if np.all(contacts) else 0
        self._filtered_barrel_contacts = np.logical_or(
            contacts, self._previous_barrel_raw_contacts
        )
        self._previous_barrel_raw_contacts = contacts.copy()
        self._refresh_hold_classification(advance_streak=True)
        if self._physics_failure_reason is not None:
            self._failure_reason = self._physics_failure_reason
            return True
        if self._step_count < CONTROL_STEPS:
            return False
        if self._final_hold_streak >= SUCCESS_CONFIG.final_hold_control_steps:
            self._failure_reason = None
        else:
            self._failure_reason = terminal_failure_reason(
                self._final_window_failure_counts()
            ) or "final_hold_streak_too_short"
        return True

    def _compute_reward(self, action: np.ndarray, terminated: bool) -> float:
        del action
        time_s = self._step_count * self.control_dt
        desired_roll = desired_roll_at_time(time_s)
        finite_dense_state = bool(
            self._physics_failure_reason != "non_finite_state"
            and
            np.isfinite(self._roll_progress)
            and np.isfinite(self._previous_roll_progress)
            and all(np.isfinite(value) for value in self._hold_metrics.values())
        )
        if finite_dense_state:
            error = desired_roll - self._roll_progress
            tracking = REWARD_CONFIG.tracking_weight * np.exp(
                -(error / REWARD_CONFIG.tracking_sigma) ** 2
            )
            expected_step = 2.0 * np.pi * self.control_dt / (
                ROLL_END_TIME - ROLL_START_TIME
            )
            signed_progress = REWARD_CONFIG.progress_weight * np.clip(
                ROLL_DIRECTION_SIGN
                * (self._roll_progress - self._previous_roll_progress)
                / expected_step,
                -REWARD_CONFIG.progress_clip,
                REWARD_CONFIG.progress_clip,
            )
            if time_s > ROLL_END_TIME:
                signed_progress = 0.0
            sub_scores = standing_subscores(
                filtered_foot_contacts=self._filtered_barrel_contacts,
                **self._hold_metrics,
            )
            standing_score = (
                float(np.mean(tuple(sub_scores.values())))
                if time_s >= REWARD_CONFIG.standing_start_time
                else 0.0
            )
        else:
            tracking = 0.0
            signed_progress = 0.0
            sub_scores = {
                "foot_score": 0.0,
                "height_score": 0.0,
                "tilt_score": 0.0,
                "linear_speed_score": 0.0,
                "angular_speed_score": 0.0,
                "joint_speed_score": 0.0,
            }
            standing_score = 0.0
        outcome = 0.0
        if terminated:
            outcome = (
                REWARD_CONFIG.success_bonus
                if self._failure_reason is None
                else REWARD_CONFIG.failure_penalty
            )
        self._reward_components = {
            "roll_tracking": float(tracking),
            "signed_progress": float(signed_progress),
            "standing_score": float(standing_score),
            "terminal_outcome": float(outcome),
            **sub_scores,
        }
        return float(tracking + signed_progress + standing_score + outcome)

    def _get_info(self) -> dict:
        info = super()._get_info()
        desired = desired_roll_at_time(self._step_count * self.control_dt)
        info.update({
            "is_success": self._terminal_success(),
            "failure_reason": self._failure_reason,
            "phase": maneuver_phase_at_time(self._step_count * self.control_dt),
            "desired_roll": desired,
            "roll_progress": self._roll_progress,
            "roll_error": desired - self._roll_progress,
            "contact_state": self._filtered_barrel_contacts.copy(),
            "raw_contact_state": self._raw_barrel_contacts.copy(),
            "stability_count": self._stability_count,
            "final_hold_streak": self._final_hold_streak,
            "final_hold_active": bool(
                self._step_count * self.control_dt > FINAL_HOLD_START_TIME
            ),
            "hold_valid": bool(all(self._hold_conditions.values())),
            "hold_conditions": self._hold_conditions.copy(),
            "hold_condition_failure_counts": dict(
                self._hold_condition_failure_counts
            ),
            "final_hold_window_failure_counts": self._final_window_failure_counts(),
            "sampled_spread": self._spread,
            "body_up_tilt": self._hold_metrics["body_up_tilt"],
            "nonfoot_ground_contact": self._last_nonfoot_contact,
            "nonfoot_ground_contact_count": self._nonfoot_contact_count,
            "base_angular_speed": self._hold_metrics["base_angular_speed"],
            "base_linear_speed": self._hold_metrics["base_linear_speed"],
            "joint_velocity_norm": self._hold_metrics["joint_velocity_norm"],
            "action_change_norm": float(np.linalg.norm(self._last_action - self._prev_last_action)),
        })
        info.update({
            f"hold_{name}_valid": valid
            for name, valid in self._hold_conditions.items()
        })
        return info
