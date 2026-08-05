"""Focused contract tests for the Go2 barrel-roll environment."""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from mpc_rl.envs.barrel_roll_common import (
    ROLL_TARGET,
    RollProgressTracker,
    desired_roll_at_time,
    maneuver_phase_at_time,
)
from mpc_rl.envs.barrel_roll_env import QuadrupedBarrelRollEnv


def _x_quaternion(angle: float) -> np.ndarray:
    return np.roll(Rotation.from_euler("x", angle).as_quat(), 1)


def test_schedule_and_unwrapped_roll_handle_wrap_and_quaternion_signs():
    assert desired_roll_at_time(0.0) == 0.0
    assert desired_roll_at_time(0.20) == 0.0
    assert np.isclose(desired_roll_at_time(0.80), ROLL_TARGET)
    assert np.isclose(maneuver_phase_at_time(1.0), 1.0)
    tracker = RollProgressTracker.from_reset_quaternion(_x_quaternion(0.0))
    for angle in (2.8, 3.2, 6.1, 6.4):
        tracker.update(_x_quaternion(angle))
    assert np.isclose(tracker.progress, 6.4, atol=1e-6)
    before = tracker.progress
    tracker.update(-_x_quaternion(6.4))
    assert np.isclose(tracker.progress, before, atol=1e-6)


def test_go2_reset_is_seeded_spread_and_has_exact_observation_shapes():
    env = QuadrupedBarrelRollEnv()
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
