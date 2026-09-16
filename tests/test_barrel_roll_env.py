"""Focused contract tests for the Go2 barrel-roll environment."""

import numpy as np
import mujoco
import pytest
from scipy.spatial.transform import Rotation

from mpc_rl.envs.barrel_roll_common import (
    ACTION_SCALE,
    CONTROL_DT,
    CONTROL_STEPS,
    EPISODE_HORIZON,
    FINAL_HOLD_CONTROL_STEPS,
    FINAL_HOLD_START_TIME,
    HOLD_FAILURE_REASONS,
    MANEUVER_HORIZON,
    REWARD_CONFIG,
    ROLL_END_TIME,
    ROLL_START_TIME,
    ROLL_TARGET,
    RollProgressTracker,
    body_up_tilt_from_quaternion,
    desired_roll_at_time,
    desired_roll_rate_at_time,
    final_hold_conditions,
    maneuver_phase_at_time,
    standing_subscores,
    terminal_failure_reason,
)
from mpc_rl.envs.barrel_roll_env import QuadrupedBarrelRollEnv
from mpc_rl.envs.go2_sysid import assert_go2_sysid_joint_dynamics


def _x_quaternion(angle: float) -> np.ndarray:
    return np.roll(Rotation.from_euler("x", angle).as_quat(), 1)


def test_schedule_and_unwrapped_roll_handle_wrap_and_quaternion_signs():
    assert desired_roll_at_time(0.0) == 0.0
    assert desired_roll_at_time(ROLL_START_TIME) == 0.0
    assert np.isclose(
        desired_roll_at_time(0.5 * (ROLL_START_TIME + ROLL_END_TIME)),
        0.5 * ROLL_TARGET,
    )
    assert np.isclose(desired_roll_at_time(ROLL_END_TIME), ROLL_TARGET)
    assert desired_roll_rate_at_time(ROLL_START_TIME) == 0.0
    assert np.isclose(
        desired_roll_rate_at_time(0.5 * (ROLL_START_TIME + ROLL_END_TIME)),
        5.0 * np.pi,
    )
    assert desired_roll_rate_at_time(ROLL_END_TIME) == 0.0
    assert maneuver_phase_at_time(ROLL_END_TIME) == 1.0
    assert maneuver_phase_at_time(MANEUVER_HORIZON) == 1.0
    assert np.isclose(maneuver_phase_at_time(EPISODE_HORIZON), 1.0)
    assert CONTROL_STEPS == 125
    assert np.isclose(CONTROL_STEPS * CONTROL_DT, EPISODE_HORIZON)
    assert FINAL_HOLD_CONTROL_STEPS == 25
    assert np.isclose(FINAL_HOLD_START_TIME, 2.0)
    tracker = RollProgressTracker.from_reset_quaternion(_x_quaternion(0.0))
    for angle in (2.8, 3.2, 6.1, 6.4):
        tracker.update(_x_quaternion(angle))
    assert np.isclose(tracker.progress, 6.4, atol=1e-6)
    before = tracker.progress
    tracker.update(-_x_quaternion(6.4))
    assert np.isclose(tracker.progress, before, atol=1e-6)


def test_go2_reset_is_seeded_spread_and_has_exact_observation_shapes():
    env = QuadrupedBarrelRollEnv()
    assert env.action_scale == ACTION_SCALE == 2.0
    assert env.mjModel.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    assert_go2_sysid_joint_dynamics(env.mjModel)
    obs_a, info_a = env.reset(seed=13)
    qpos_a = env.mjData.qpos.copy()
    obs_b, info_b = env.reset(seed=13)
    qpos_b = env.mjData.qpos.copy()
    assert np.array_equal(qpos_a, qpos_b)
    assert info_a["sampled_spread"] == info_b["sampled_spread"]
    assert 0.0 <= info_a["sampled_spread"] <= 0.10
    assert obs_a["policy"].shape == (45,)
    assert obs_a["privileged"].shape == (4,)
    assert np.isfinite(obs_a["policy"]).all() and np.isfinite(obs_a["privileged"]).all()
    addresses = [
        int(env.mjModel.jnt_qposadr[mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in ("FL_hip_joint", "RL_hip_joint", "FR_hip_joint", "RR_hip_joint")
    ]
    spread = info_a["sampled_spread"]
    assert np.isclose(qpos_a[addresses[0]] - env.default_qpos[addresses[0]], spread)
    assert np.isclose(qpos_a[addresses[1]] - env.default_qpos[addresses[1]], spread)
    assert np.isclose(qpos_a[addresses[2]] - env.default_qpos[addresses[2]], -spread)
    assert np.isclose(qpos_a[addresses[3]] - env.default_qpos[addresses[3]], -spread)
    env.close()


def test_barrel_roll_requires_sysid_and_supports_nominal_commissioning_spread():
    with pytest.raises(ValueError, match="requires use_go2_sysid=True"):
        QuadrupedBarrelRollEnv(use_go2_sysid=False)
    with pytest.raises(ValueError, match="schema-v4 requires action_scale=2.0"):
        QuadrupedBarrelRollEnv(action_scale=0.5)

    env = QuadrupedBarrelRollEnv()
    try:
        _, info = env.reset(seed=5, options={"spread": 0.0})
        assert info["sampled_spread"] == 0.0
    finally:
        env.close()


def test_hand_authored_action_rollout_is_finite_without_premature_termination():
    env = QuadrupedBarrelRollEnv()
    env.reset(seed=3)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        assert np.isfinite(reward)
        assert np.isfinite(obs["policy"]).all()
        assert not terminated and not truncated
        assert info["failure_reason"] is None
    env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.ones(12),
        -np.ones(12),
        np.tile(np.array([1.0, -1.0, 1.0]), 4),
    ],
)
def test_schema_v4_extreme_action_pulse_is_finite_and_torque_limited(action):
    env = QuadrupedBarrelRollEnv()
    try:
        env.reset(seed=146)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        assert np.isfinite(reward)
        assert np.isfinite(obs["policy"]).all()
        assert np.isfinite(env.mjData.qpos).all()
        assert np.isfinite(env.mjData.qvel).all()
        assert np.all(env._applied_torques >= env.torque_limits[:, 0])
        assert np.all(env._applied_torques <= env.torque_limits[:, 1])
        np.testing.assert_allclose(
            env._raw_q_target,
            env.default_joint_pos + ACTION_SCALE * action,
        )
    finally:
        env.close()


def _set_ideal_hold_state(env):
    env._roll_progress = 2.0 * np.pi
    env.mjData.qpos[2] = 0.27
    env.mjData.qpos[3:7] = _x_quaternion(0.0)
    env.mjData.qvel[:] = 0.0
    env._filtered_barrel_contacts[:] = True
    env._refresh_hold_classification(advance_streak=False)


def test_schema_v4_reward_gates_and_terminal_outcomes():
    env = QuadrupedBarrelRollEnv()
    try:
        env.reset(seed=7, options={"spread": 0.0})
        env._step_count = int(round(0.5 / CONTROL_DT))
        env._roll_progress = desired_roll_at_time(0.5)
        expected_step = 2.0 * np.pi * CONTROL_DT / (ROLL_END_TIME - ROLL_START_TIME)
        env._previous_roll_progress = env._roll_progress - expected_step
        env._filtered_barrel_contacts[:] = True
        env._refresh_hold_classification(advance_streak=False)
        reward = env._compute_reward(np.zeros(12), terminated=False)
        assert reward == pytest.approx(1.25)
        assert env._reward_components["roll_tracking"] == pytest.approx(1.0)
        assert env._reward_components["signed_progress"] == pytest.approx(0.25)
        assert env._reward_components["standing_score"] == 0.0
        assert env._reward_components["terminal_outcome"] == 0.0

        env._step_count = int(round(0.82 / CONTROL_DT))
        env._previous_roll_progress = env._roll_progress - expected_step
        env._compute_reward(np.ones(12), terminated=False)
        assert env._reward_components["signed_progress"] == 0.0

        _set_ideal_hold_state(env)
        env._previous_roll_progress = env._roll_progress
        env._step_count = int(round((MANEUVER_HORIZON - CONTROL_DT) / CONTROL_DT))
        env._compute_reward(np.zeros(12), terminated=False)
        assert env._reward_components["standing_score"] == 0.0
        env._step_count += 1
        standing_reward = env._compute_reward(np.zeros(12), terminated=False)
        assert env._reward_components["standing_score"] == pytest.approx(1.0)
        assert standing_reward == pytest.approx(2.0)

        env._step_count = CONTROL_STEPS
        env._failure_reason = None
        terminal_reward = env._compute_reward(np.zeros(12), terminated=True)
        assert env._reward_components["terminal_outcome"] == REWARD_CONFIG.success_bonus
        assert terminal_reward == pytest.approx(27.0)
        env._failure_reason = "body_tilt_above_maximum"
        failed_reward = env._compute_reward(np.zeros(12), terminated=True)
        assert env._reward_components["terminal_outcome"] == REWARD_CONFIG.failure_penalty
        assert failed_reward == pytest.approx(-23.0)
    finally:
        env.close()


def _hold_conditions(**overrides):
    values = {
        "filtered_foot_contacts": np.ones(4, dtype=bool),
        "roll_progress": 2.0 * np.pi,
        "base_height": 0.27,
        "body_up_tilt": 0.0,
        "base_linear_speed": 0.0,
        "base_angular_speed": 0.0,
        "joint_velocity_norm": 0.0,
    }
    values.update(overrides)
    return final_hold_conditions(**values)


@pytest.mark.parametrize(
    ("name", "on_boundary", "inside", "outside"),
    [
        ("rotation", 2.0 * np.pi - 0.35, 2.0 * np.pi - 0.35 + 1e-9, 2.0 * np.pi - 0.35 - 1e-9),
        ("rotation", 2.0 * np.pi + 0.35, 2.0 * np.pi + 0.35 - 1e-9, 2.0 * np.pi + 0.35 + 1e-9),
        ("height", 0.20, 0.20 + 1e-9, 0.20 - 1e-9),
        ("tilt", 0.35, 0.35 - 1e-9, 0.35 + 1e-9),
        ("base_linear_speed", 0.10, 0.10 - 1e-9, 0.10 + 1e-9),
        ("base_angular_speed", 0.50, 0.50 - 1e-9, 0.50 + 1e-9),
        ("joint_speed", 1.0, 1.0 - 1e-9, 1.0 + 1e-9),
    ],
)
def test_schema_v4_hold_thresholds_on_just_inside_and_just_outside(
    name, on_boundary, inside, outside
):
    argument = {
        "rotation": "roll_progress",
        "height": "base_height",
        "tilt": "body_up_tilt",
        "base_linear_speed": "base_linear_speed",
        "base_angular_speed": "base_angular_speed",
        "joint_speed": "joint_velocity_norm",
    }[name]
    inclusive = name in {"rotation", "height", "tilt"}
    assert _hold_conditions(**{argument: on_boundary})[name] is inclusive
    assert _hold_conditions(**{argument: inside})[name]
    assert not _hold_conditions(**{argument: outside})[name]


def test_schema_v4_foot_support_and_failure_precedence_are_explicit():
    assert _hold_conditions()["foot_support"]
    assert not _hold_conditions(
        filtered_foot_contacts=np.array([True, True, False, True])
    )["foot_support"]
    failures = {name: 1 for name in HOLD_FAILURE_REASONS}
    assert terminal_failure_reason(failures) == "rotation_out_of_band"
    failures["rotation"] = 0
    assert terminal_failure_reason(failures) == "insufficient_foot_support"
    assert terminal_failure_reason({name: 0 for name in failures}) is None


def test_schema_v4_standing_subscores_follow_the_locked_smooth_functions():
    scores = standing_subscores(
        filtered_foot_contacts=np.array([True, False, True, False]),
        base_height=0.34,
        body_up_tilt=0.35,
        base_linear_speed=0.10,
        base_angular_speed=0.50,
        joint_velocity_norm=1.0,
    )
    assert scores["foot_score"] == 0.5
    for name in (
        "height_score",
        "tilt_score",
        "linear_speed_score",
        "angular_speed_score",
        "joint_speed_score",
    ):
        assert scores[name] == pytest.approx(np.exp(-1.0))


def test_schema_v4_contact_filter_uses_genuine_previous_control_endpoint(monkeypatch):
    env = QuadrupedBarrelRollEnv()
    try:
        env.reset(seed=11, options={"spread": 0.0})
        _set_ideal_hold_state(env)
        env._previous_barrel_raw_contacts[:] = True
        dropout = np.array([True, True, False, True])
        monkeypatch.setattr(env, "_get_foot_contacts", lambda: dropout.copy())
        env._step_count = 1
        assert not env._check_termination()
        assert np.all(env._filtered_barrel_contacts)
        assert env._final_hold_streak == 1
        env._step_count = 2
        assert not env._check_termination()
        assert np.array_equal(env._filtered_barrel_contacts, dropout)
        assert env._final_hold_streak == 0
    finally:
        env.close()


def test_schema_v4_terminal_success_requires_all_last_25_endpoints(monkeypatch):
    env = QuadrupedBarrelRollEnv()
    try:
        env.reset(seed=12, options={"spread": 0.0})
        _set_ideal_hold_state(env)
        monkeypatch.setattr(
            env, "_get_foot_contacts", lambda: np.ones(4, dtype=bool)
        )
        env._previous_barrel_raw_contacts[:] = True
        for step in range(CONTROL_STEPS - FINAL_HOLD_CONTROL_STEPS + 1, CONTROL_STEPS + 1):
            env._step_count = step
            env.mjData.qvel[:] = 0.0
            if step == CONTROL_STEPS - FINAL_HOLD_CONTROL_STEPS + 1:
                env.mjData.qvel[0] = 0.10
            terminated = env._check_termination()
        assert terminated
        assert env._final_hold_streak == FINAL_HOLD_CONTROL_STEPS - 1
        assert env._failure_reason == "base_linear_speed_above_maximum"
        assert not env._terminal_success()

        env.reset(seed=12, options={"spread": 0.0})
        _set_ideal_hold_state(env)
        env._previous_barrel_raw_contacts[:] = True
        for step in range(CONTROL_STEPS - FINAL_HOLD_CONTROL_STEPS + 1, CONTROL_STEPS + 1):
            env._step_count = step
            terminated = env._check_termination()
        assert terminated
        assert env._final_hold_streak == FINAL_HOLD_CONTROL_STEPS
        assert env._failure_reason is None
        assert env._terminal_success()
    finally:
        env.close()


def test_nonfoot_contact_and_nonfinite_state_fail_immediately(monkeypatch):
    env = QuadrupedBarrelRollEnv()
    try:
        env.reset(seed=17, options={"spread": 0.0})
        monkeypatch.setattr(
            "mpc_rl.envs.barrel_roll_env.find_nonfoot_ground_contact",
            lambda unused_env: "torso",
        )
        start_time = float(env.mjData.time)
        _, reward, terminated, truncated, info = env.step(np.zeros(12))
        assert terminated and not truncated
        assert info["failure_reason"] == "non_foot_ground_contact"
        assert info["nonfoot_ground_contact"] == "torso"
        assert env.mjData.time - start_time == pytest.approx(env.sim_dt)
        assert np.isfinite(reward)

        monkeypatch.undo()
        env.reset(seed=17, options={"spread": 0.0})
        env.mjData.qpos[0] = np.nan
        env._after_physics_substep()
        assert env._physics_failure_reason == "non_finite_state"
        assert env._check_termination()
        assert env._failure_reason == "non_finite_state"
        reward = env._compute_reward(np.zeros(12), terminated=True)
        assert reward == REWARD_CONFIG.failure_penalty
    finally:
        env.close()


def test_schema_v4_contract_has_no_flight_duration_hold_condition():
    assert "flight" not in _hold_conditions()


def test_body_up_tilt_uses_body_z_against_world_z():
    assert body_up_tilt_from_quaternion(_x_quaternion(0.0)) == pytest.approx(0.0)
    assert body_up_tilt_from_quaternion(_x_quaternion(np.pi / 3.0)) == pytest.approx(
        np.pi / 3.0
    )
