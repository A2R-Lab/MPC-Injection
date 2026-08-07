"""Fixed-direction, one-shot Go2 barrel-roll Gymnasium environment."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from mpc_rl.envs.barrel_roll_common import (
    ACTION_SCALE,
    CONTROL_DT,
    CONTROL_STEPS,
    REWARD_CONFIG,
    ROLL_DIRECTION_SIGN,
    ROLL_END_TIME,
    ROLL_START_TIME,
    ROLL_TARGET,
    SUCCESS_CONFIG,
    RollProgressTracker,
    assert_valid_initial_state,
    desired_roll_at_time,
    find_nonfoot_ground_contact,
    maneuver_phase_at_time,
    sample_symmetric_hip_spread,
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
                f"barrel-roll schema-v2 requires action_scale={ACTION_SCALE}, "
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
        self._filtered_barrel_contacts = np.zeros(4, dtype=bool)

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
        self._last_foot_contacts[:] = False
        self._current_contacts[:] = False
        self._filtered_barrel_contacts[:] = False
        self._reward_components = {}
        obs = self._get_obs()
        return obs, self._get_info()

    def _after_physics_substep(self) -> None:
        self._roll_progress = self._roll_tracker.update(self.mjData.qpos[3:7])
        if not np.isfinite(self.mjData.qpos).all() or not np.isfinite(self.mjData.qvel).all() or not np.isfinite(self.mjData.ctrl).all():
            self._physics_failure_reason = "non_finite_state"
        elif self._physics_failure_reason is None:
            nonfoot = find_nonfoot_ground_contact(self)
            if nonfoot is not None:
                self._physics_failure_reason = f"non_foot_ground_contact:{nonfoot}"

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

    def _terminal_success(self) -> bool:
        final_roll, pitch = self._upright_errors()
        required_progress = ROLL_DIRECTION_SIGN * self._roll_progress >= 2.0 * np.pi - SUCCESS_CONFIG.progress_tolerance
        upright = abs(final_roll) <= SUCCESS_CONFIG.upright_tolerance and abs(pitch) <= SUCCESS_CONFIG.upright_tolerance
        stable = self._stability_count >= SUCCESS_CONFIG.stable_control_steps
        return bool(required_progress and upright and self.mjData.qpos[2] >= SUCCESS_CONFIG.min_base_height and stable)

    def _check_termination(self) -> bool:
        if self._physics_failure_reason is not None:
            self._failure_reason = self._physics_failure_reason
            return True
        contacts = self._get_foot_contacts()
        self._filtered_barrel_contacts = np.logical_or(contacts, self._last_foot_contacts)
        if np.all(self._filtered_barrel_contacts):
            self._stability_count += 1
        else:
            self._stability_count = 0
        if self._step_count < CONTROL_STEPS:
            return False
        if self._terminal_success():
            self._failure_reason = None
        else:
            self._failure_reason = "incomplete_roll"
        return True

    def _compute_reward(self, action: np.ndarray, terminated: bool) -> float:
        desired_roll = desired_roll_at_time(self._step_count * self.control_dt)
        error = desired_roll - self._roll_progress
        tracking = REWARD_CONFIG.tracking_weight * np.exp(-(error / REWARD_CONFIG.tracking_sigma) ** 2)
        expected_step = (
            2.0 * np.pi * self.control_dt / (ROLL_END_TIME - ROLL_START_TIME)
        )
        signed_delta = ROLL_DIRECTION_SIGN * (self._roll_progress - self._previous_roll_progress)
        progress = REWARD_CONFIG.progress_weight * np.clip(signed_delta / expected_step, -REWARD_CONFIG.progress_clip, REWARD_CONFIG.progress_clip)
        outcome = 0.0
        if terminated:
            outcome = REWARD_CONFIG.success_bonus if self._failure_reason is None else REWARD_CONFIG.failure_penalty
        self._reward_components = {"roll_tracking": float(tracking), "signed_progress": float(progress), "terminal_outcome": float(outcome)}
        return float(tracking + progress + outcome)

    def _get_info(self) -> dict:
        info = super()._get_info()
        desired = desired_roll_at_time(self._step_count * self.control_dt)
        info.update({
            "is_success": self._failure_reason is None and self._step_count >= CONTROL_STEPS,
            "failure_reason": self._failure_reason,
            "phase": maneuver_phase_at_time(self._step_count * self.control_dt),
            "desired_roll": desired,
            "roll_progress": self._roll_progress,
            "roll_error": desired - self._roll_progress,
            "contact_state": self._filtered_barrel_contacts.copy(),
            "stability_count": self._stability_count,
            "sampled_spread": self._spread,
        })
        return info
