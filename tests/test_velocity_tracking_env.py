"""Tests for the QuadrupedVelocityTrackingEnv and play_quad.py utilities.

Covers:
    1. Timing / control frequency (sim_dt, decimation, control_dt)
    2. set_commands / fixed_commands (no auto-resampling during eval/play)
    3. Basic env sanity (obs shapes, action space, step/reset)
    4. play_quad.py helpers (VelocityCommander, detect_algorithm)

Run:
    conda activate mpc-rl
    python -m pytest tests/test_velocity_tracking_env.py -v
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure mpc_rl is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import mpc_rl.envs
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def env():
    """Create a default QuadrupedVelocityTrackingEnv and close it after the test."""
    e = QuadrupedVelocityTrackingEnv(robot="go2", scene="flat")
    yield e
    e.close()


@pytest.fixture
def env_custom():
    """Create an env with custom sim_dt/decimation for parameterised tests."""
    e = QuadrupedVelocityTrackingEnv(
        robot="go2", scene="flat", sim_dt=0.002, decimation=10
    )
    yield e
    e.close()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Timing / Control Frequency
# ═══════════════════════════════════════════════════════════════════════════


class TestTimingAndFrequency:
    """Verify sim_dt, decimation, and control_dt match the real Go2 robot."""

    def test_default_sim_dt(self, env):
        """Default physics timestep should be 0.005s (200 Hz)."""
        assert env.sim_dt == 0.005

    def test_default_decimation(self, env):
        """Default decimation should be 4."""
        assert env.decimation == 4

    def test_default_control_dt(self, env):
        """control_dt = sim_dt * decimation = 0.005 * 4 = 0.02s (50 Hz)."""
        assert env.control_dt == pytest.approx(0.02)

    def test_control_frequency_50hz(self, env):
        """Control frequency should be 50 Hz."""
        freq = 1.0 / env.control_dt
        assert freq == pytest.approx(50.0)

    def test_mujoco_timestep_matches_sim_dt(self, env):
        """MuJoCo model.opt.timestep must equal sim_dt after init."""
        assert env.mjModel.opt.timestep == pytest.approx(env.sim_dt)

    def test_simulation_time_advances_correctly(self, env):
        """N env.step() calls should advance sim time by N * control_dt."""
        env.reset(seed=42)
        t0 = env.mjData.time
        n_steps = 10
        action = np.zeros(env.num_joints)
        for _ in range(n_steps):
            env.step(action)
        t1 = env.mjData.time
        expected_advance = n_steps * env.control_dt
        assert t1 - t0 == pytest.approx(expected_advance, abs=1e-9)

    def test_custom_sim_dt_decimation(self, env_custom):
        """Custom sim_dt=0.002, decimation=10 → control_dt=0.02 (still 50 Hz)."""
        assert env_custom.sim_dt == 0.002
        assert env_custom.decimation == 10
        assert env_custom.control_dt == pytest.approx(0.02)
        assert env_custom.mjModel.opt.timestep == pytest.approx(0.002)

    def test_custom_timing_sim_advance(self, env_custom):
        """Verify sim time advance with custom dt/decimation."""
        env_custom.reset(seed=42)
        t0 = env_custom.mjData.time
        action = np.zeros(env_custom.num_joints)
        for _ in range(5):
            env_custom.step(action)
        t1 = env_custom.mjData.time
        assert t1 - t0 == pytest.approx(5 * 0.02, abs=1e-9)

    def test_render_fps_metadata(self, env):
        """Metadata render_fps should match control frequency."""
        assert env.metadata["render_fps"] == 50


# ═══════════════════════════════════════════════════════════════════════════
# 2. set_commands / Fixed Commands
# ═══════════════════════════════════════════════════════════════════════════


class TestFixedCommands:
    """Verify that set_commands disables auto-resampling."""

    def test_set_commands_sets_flag(self, env):
        """set_commands() should set _fixed_commands to True."""
        assert env._fixed_commands is False
        env.set_commands(vx=0.5, vy=0.1, wz=-0.2)
        assert env._fixed_commands is True

    def test_set_commands_values(self, env):
        """set_commands() should set the internal commands array."""
        env.set_commands(vx=0.5, vy=-0.3, wz=1.0)
        np.testing.assert_allclose(env._commands, [0.5, -0.3, 1.0])

    def test_release_commands(self, env):
        """release_commands() should re-enable auto-resampling."""
        env.set_commands(vx=1.0)
        assert env._fixed_commands is True
        env.release_commands()
        assert env._fixed_commands is False

    def test_step_does_not_resample_when_fixed(self, env):
        """When _fixed_commands=True, step() must not change commands."""
        env.reset(seed=42)
        env.set_commands(vx=0.77, vy=-0.33, wz=0.11)
        action = np.zeros(env.num_joints)

        # Step many times past the resample interval
        for _ in range(env.command_resample_interval + 100):
            env.step(action)

        np.testing.assert_allclose(env._commands, [0.77, -0.33, 0.11])

    def test_reset_does_not_resample_when_fixed(self, env):
        """When _fixed_commands=True, reset() must not change commands."""
        env.set_commands(vx=0.5, vy=0.2, wz=-0.1)
        env.reset(seed=99)
        np.testing.assert_allclose(env._commands, [0.5, 0.2, -0.1])

    def test_step_resamples_when_not_fixed(self, env):
        """When _fixed_commands=False, step() should resample after interval."""
        env.reset(seed=42)
        initial_cmds = env._commands.copy()
        action = np.zeros(env.num_joints)

        # Step past the resample interval
        for _ in range(env.command_resample_interval + 1):
            env.step(action)

        # Commands should have changed (extremely unlikely to be identical)
        assert not np.allclose(env._commands, initial_cmds), (
            "Commands did not change after exceeding command_resample_interval"
        )

    def test_commands_in_observation(self, env):
        """Fixed commands should appear in the policy observation [6:9]."""
        env.reset(seed=42)
        env.set_commands(vx=0.8, vy=-0.4, wz=0.6)
        obs = env._get_obs()
        np.testing.assert_allclose(obs["policy"][6:9], [0.8, -0.4, 0.6])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Basic Environment Sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvSanity:
    """Basic gym.Env contract tests."""

    def test_observation_space_dict(self, env):
        """Observation space should be a Dict with 'policy' and 'privileged'."""
        assert "policy" in env.observation_space.spaces
        assert "privileged" in env.observation_space.spaces

    def test_policy_obs_dim(self, env):
        """Policy obs should be 45-dim (3+3+3+12+12+12)."""
        assert env.observation_space["policy"].shape == (45,)

    def test_privileged_obs_dim(self, env):
        """Privileged obs should be 3-dim (base linear velocity)."""
        assert env.observation_space["privileged"].shape == (3,)

    def test_action_space(self, env):
        """Action space should be 12-dim in [-1, 1]."""
        assert env.action_space.shape == (12,)
        np.testing.assert_allclose(env.action_space.low, -1.0)
        np.testing.assert_allclose(env.action_space.high, 1.0)

    def test_reset_returns_correct_shapes(self, env):
        """reset() should return (obs_dict, info_dict)."""
        obs, info = env.reset(seed=42)
        assert obs["policy"].shape == (45,)
        assert obs["privileged"].shape == (3,)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self, env):
        """step() should return (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["policy"].shape == (45,)
        assert obs["privileged"].shape == (3,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_multiple_steps_no_crash(self, env):
        """Run 200 steps with zero action without crashing."""
        env.reset(seed=42)
        action = np.zeros(env.num_joints)
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                env.reset(seed=42)

    def test_step_count_increments(self, env):
        """_step_count should increment by 1 per step() call."""
        env.reset(seed=42)
        assert env._step_count == 0
        env.step(np.zeros(env.num_joints))
        assert env._step_count == 1
        env.step(np.zeros(env.num_joints))
        assert env._step_count == 2

    def test_reset_clears_step_count(self, env):
        """reset() should set _step_count to 0."""
        env.reset(seed=42)
        env.step(np.zeros(env.num_joints))
        env.step(np.zeros(env.num_joints))
        env.reset(seed=42)
        assert env._step_count == 0

    def test_info_keys(self, env):
        """Info dict should contain expected debugging keys."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(env.num_joints))
        expected_keys = {
            "step_count", "commands", "base_lin_vel_body",
            "base_ang_vel_body", "base_height", "applied_torques",
        }
        assert expected_keys.issubset(info.keys())

    def test_robot_stands_with_zero_action(self, env):
        """With zero action the robot should remain roughly upright for 100 steps."""
        env.reset(seed=42)
        action = np.zeros(env.num_joints)
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                pytest.fail("Robot fell with zero action within 100 steps")
        # Base height should still be reasonable (above 0.15m)
        assert info["base_height"] > 0.15, (
            f"Base height {info['base_height']:.3f}m is too low"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. play_quad.py Utilities
# ═══════════════════════════════════════════════════════════════════════════


class TestVelocityCommander:
    """Test the VelocityCommander keyboard state machine."""

    def test_initial_state(self):
        from mpc_rl.play_quad import VelocityCommander
        cmd = VelocityCommander(step=0.1)
        assert cmd.get() == (0.0, 0.0, 0.0)
        assert cmd.is_stopped() is False

    def test_arrow_keys(self):
        from mpc_rl.play_quad import (
            VelocityCommander, _KEY_UP, _KEY_DOWN, _KEY_LEFT, _KEY_RIGHT,
        )
        cmd = VelocityCommander(step=0.1)
        cmd.on_key(_KEY_UP)
        assert cmd.get()[0] == pytest.approx(0.1)
        cmd.on_key(_KEY_DOWN)
        assert cmd.get()[0] == pytest.approx(0.0)
        cmd.on_key(_KEY_LEFT)
        assert cmd.get()[1] == pytest.approx(0.1)
        cmd.on_key(_KEY_RIGHT)
        assert cmd.get()[1] == pytest.approx(0.0)

    def test_bracket_keys(self):
        from mpc_rl.play_quad import (
            VelocityCommander, _KEY_LEFT_BRACKET, _KEY_RIGHT_BRACKET,
        )
        cmd = VelocityCommander(step=0.2)
        cmd.on_key(_KEY_LEFT_BRACKET)
        assert cmd.get()[2] == pytest.approx(0.2)
        cmd.on_key(_KEY_RIGHT_BRACKET)
        assert cmd.get()[2] == pytest.approx(0.0)

    def test_reset_key(self):
        from mpc_rl.play_quad import VelocityCommander, _KEY_UP, _KEY_R
        cmd = VelocityCommander(step=0.5)
        cmd.on_key(_KEY_UP)
        cmd.on_key(_KEY_UP)
        assert cmd.get()[0] == pytest.approx(1.0)
        cmd.on_key(_KEY_R)
        assert cmd.get() == (0.0, 0.0, 0.0)

    def test_escape_stops(self):
        from mpc_rl.play_quad import VelocityCommander, _KEY_ESCAPE
        cmd = VelocityCommander(step=0.1)
        assert cmd.is_stopped() is False
        cmd.on_key(_KEY_ESCAPE)
        assert cmd.is_stopped() is True

    def test_custom_step_size(self):
        from mpc_rl.play_quad import VelocityCommander, _KEY_UP
        cmd = VelocityCommander(step=0.5)
        cmd.on_key(_KEY_UP)
        assert cmd.get()[0] == pytest.approx(0.5)


class TestDetectAlgorithm:
    """Test algorithm detection from directory names."""

    def test_sac_in_dirname(self):
        from mpc_rl.play_quad import detect_algorithm
        assert detect_algorithm("/logs/quadruped-velocity_tracking-SAC-20250101-120000") == "SAC"

    def test_td3_in_dirname(self):
        from mpc_rl.play_quad import detect_algorithm
        assert detect_algorithm("/logs/quadruped-velocity_tracking-TD3-20250101-120000") == "TD3"

    def test_sac_with_suffix(self):
        from mpc_rl.play_quad import detect_algorithm
        assert detect_algorithm("/logs/quadruped-velocity_tracking-SAC-20250101-120000-mysuffix") == "SAC"

    def test_fallback_to_config_json(self):
        from mpc_rl.play_quad import detect_algorithm
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"algorithm": "TD3"}
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
            assert detect_algorithm(tmpdir) == "TD3"

    def test_config_json_strips_mpc_suffix(self):
        from mpc_rl.play_quad import detect_algorithm
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"algorithm": "SAC-MPC"}
            with open(Path(tmpdir) / "config.json", "w") as f:
                json.dump(config, f)
            assert detect_algorithm(tmpdir) == "SAC"

    def test_unknown_algo_raises(self):
        from mpc_rl.play_quad import detect_algorithm
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Could not detect algorithm"):
                detect_algorithm(tmpdir)


class TestResolveModelPath:
    """Test glob-based model path resolution."""

    def test_exact_path(self):
        from mpc_rl.play_quad import resolve_model_path
        with tempfile.TemporaryDirectory() as tmpdir:
            p = resolve_model_path(tmpdir)
            assert p == Path(tmpdir)

    def test_glob_pattern(self):
        from mpc_rl.play_quad import resolve_model_path
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "quadruped-SAC-20250101"
            sub.mkdir()
            p = resolve_model_path(str(Path(tmpdir) / "quadruped-SAC-*"))
            assert p == sub

    def test_no_match_raises(self):
        from mpc_rl.play_quad import resolve_model_path
        with pytest.raises(FileNotFoundError):
            resolve_model_path("/nonexistent/path/that/does/not/exist-*")
