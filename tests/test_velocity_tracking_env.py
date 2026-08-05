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

import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import mujoco
from gymnasium import spaces

# Ensure mpc_rl is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import mpc_rl.envs
import mpx.config.config_go2 as go2_config
from gym_quadruped.quadruped_env import QuadrupedEnv
from mpc_rl.common.mpc_inject_callbacks import (
    PercentMPCInjectCallback,
    _apply_quadruped_trajectory_dr_patch,
    _assert_quadruped_generation_friction,
)
from mpc_rl.common.tagged_dict_replay_buffer import TaggedDictReplayBuffer
from mpc_rl.envs.domain_randomization import (
    DomainRandomizationConfig,
    apply_startup_domain_rand_patch,
    extract_startup_domain_rand_patch,
    resolve_startup_domain_rand_config,
    sample_startup_domain_rand_patch,
)
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.planner.gen_traj_data_mpx_dr import (
    gen_traj_quadruped_dr,
    generate_trajectory as generate_dr_mpx_trajectory,
)


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


class FakeQuadrupedMPC:
    """Small deterministic MPC stub for trajectory-generator tests."""

    def __init__(self):
        self.robot_height = go2_config.robot_height
        self.duty_factor = 0.5

    def reset(self, qpos, qvel):
        self.last_reset = (np.array(qpos, copy=True), np.array(qvel, copy=True))

    def run(self, qpos, qvel, mpx_input, contact):
        del qpos, qvel, mpx_input, contact
        return (
            np.zeros(go2_config.n_joints, dtype=np.float64),
            np.array(go2_config.q0, dtype=np.float64),
            np.zeros(go2_config.n_joints, dtype=np.float64),
        )


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


class TestRewardConfiguration:
    """Reward-shaping regression tests for the shared full reward."""

    def test_reward_tuning_values(self):
        """Shared full reward should use the locomotion tune."""
        cfg = QuadrupedVelocityTrackingEnv._default_reward_cfg()

        assert cfg["w_track_lin_vel"] == pytest.approx(3.0)
        assert cfg["w_lin_vel_forward"] == pytest.approx(5.0)
        assert cfg["w_track_ang_vel"] == pytest.approx(1.5)
        assert cfg["w_flat_orientation"] == pytest.approx(-0.5)
        assert cfg["w_pose"] == pytest.approx(0.35)
        assert cfg["w_track_base_height"] == pytest.approx(1.0)
        assert cfg["base_height_target"] == pytest.approx(0.27)
        assert cfg["base_height_sigma"] == pytest.approx(0.01)
        assert cfg["w_body_ang_vel"] == pytest.approx(-0.12)
        assert cfg["w_angular_momentum"] == pytest.approx(-0.012)
        assert cfg["w_action_rate"] == pytest.approx(-0.045)
        assert cfg["w_feet_air_time"] == pytest.approx(0.75)
        assert cfg["w_feet_clearance"] == pytest.approx(-0.8)
        assert cfg["w_feet_slip"] == pytest.approx(-0.1)

    def test_simple_reward_dispatch_still_bypasses_full_reward(self, env):
        """The MPC-injection simple reward guard must remain intact."""
        env.simple_reward = True
        sentinel_reward = 123.456

        def _sentinel_simple_reward(action, terminated):
            del action, terminated
            return sentinel_reward

        env._compute_simple_reward = _sentinel_simple_reward

        reward = env._compute_reward(np.zeros(env.num_joints), terminated=False)

        assert reward == pytest.approx(sentinel_reward)


class TestActionLowPassFilter:
    """Verify policy targets are filtered before PD control."""

    def test_default_action_lpf_cutoff(self, env):
        """Default action target LPF should be enabled at 5 Hz."""
        assert env.action_lpf_cutoff_hz == pytest.approx(5.0)
        expected_alpha = 1.0 - np.exp(-2.0 * np.pi * 5.0 * env.control_dt)
        assert env.action_lpf_alpha == pytest.approx(expected_alpha)

    def test_action_lpf_filters_joint_targets(self, env):
        """First nonzero action should move the PD target partway from default pose."""
        env.reset(seed=42)
        action = np.ones(env.num_joints)
        raw_target = env.default_joint_pos + env.action_scale * action
        expected_target = (
            env.default_joint_pos
            + env.action_lpf_alpha * (raw_target - env.default_joint_pos)
        )

        env.step(action)

        np.testing.assert_allclose(env._raw_q_target, raw_target)
        np.testing.assert_allclose(env._filtered_q_target, expected_target)

    def test_action_lpf_can_be_disabled(self):
        """Setting cutoff to None should pass raw targets straight to the PD loop."""
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            action_lpf_cutoff_hz=None,
        )
        try:
            env.reset(seed=42)
            action = np.ones(env.num_joints)
            expected_target = env.default_joint_pos + env.action_scale * action

            env.step(action)

            assert env.action_lpf_alpha == pytest.approx(1.0)
            np.testing.assert_allclose(env._filtered_q_target, expected_target)
        finally:
            env.close()


class TestSubstepDiagnostics:
    """Verify opt-in diagnostics cover the exact environment control loop."""

    def test_diagnostics_are_disabled_by_default(self, env):
        env.reset(seed=42)

        _, _, _, _, info = env.step(np.zeros(env.num_joints))

        assert "substep_diagnostics" not in info
        assert env._last_substep_diagnostics is None

    def test_diagnostics_capture_clipping_torque_contacts_and_state(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(
                enable=False, push_robots=False
            ),
            enable_substep_diagnostics=True,
        )
        try:
            env.reset(seed=42)
            env.torque_limits[:] = np.array([-0.01, 0.01])

            _, _, _, _, info = env.step(np.full(env.num_joints, 2.0))
            diagnostics = info["substep_diagnostics"]

            np.testing.assert_array_equal(
                diagnostics["clipped_action"], np.ones(env.num_joints)
            )
            assert np.all(diagnostics["action_clipping_mask"])
            np.testing.assert_allclose(
                diagnostics["action_clipping_magnitude"],
                np.ones(env.num_joints),
            )
            assert diagnostics["requested_torques"].shape == (
                env.decimation,
                env.num_joints,
            )
            assert diagnostics["applied_torques"].shape == (
                env.decimation,
                env.num_joints,
            )
            assert diagnostics["torque_saturation_mask"].shape == (
                env.decimation,
                env.num_joints,
            )
            assert np.any(diagnostics["torque_saturation_mask"])
            np.testing.assert_allclose(
                diagnostics["torque_saturation_magnitude"],
                np.abs(
                    diagnostics["requested_torques"]
                    - diagnostics["applied_torques"]
                ),
            )
            assert diagnostics["foot_contacts"].shape == (
                env.decimation,
                4,
            )
            assert diagnostics["non_foot_ground_contact"].shape == (
                env.decimation,
            )
            assert diagnostics["qpos"].shape == (
                env.decimation,
                env.mjModel.nq,
            )
            assert diagnostics["qvel"].shape == (
                env.decimation,
                env.mjModel.nv,
            )
            assert diagnostics["time"].shape == (env.decimation,)
            assert diagnostics["push_delta_qvel"].shape == (6,)
            np.testing.assert_array_equal(
                diagnostics["applied_torques"][-1], info["applied_torques"]
            )
        finally:
            env.close()

    def test_transient_non_foot_contact_is_retained(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            enable_substep_diagnostics=True,
        )
        try:
            env.reset(seed=42)
            contacts = iter(
                [None, ("base", "base touched ground"), None, None]
            )
            env._find_non_foot_ground_contact = lambda: next(contacts)

            _, _, _, _, info = env.step(np.zeros(env.num_joints))
            diagnostics = info["substep_diagnostics"]

            np.testing.assert_array_equal(
                diagnostics["non_foot_ground_contact"],
                [False, True, False, False],
            )
            assert diagnostics["non_foot_ground_contact_body"][1] == "base"
            assert (
                diagnostics["non_foot_ground_contact_detail"][1]
                == "base touched ground"
            )
        finally:
            env.close()

    def test_enabling_diagnostics_does_not_change_transition(self):
        common_kwargs = {
            "robot": "go2",
            "scene": "flat",
            "domain_rand_cfg": DomainRandomizationConfig(
                enable=False, push_robots=False
            ),
        }
        plain_env = QuadrupedVelocityTrackingEnv(**common_kwargs)
        diagnostic_env = QuadrupedVelocityTrackingEnv(
            **common_kwargs,
            enable_substep_diagnostics=True,
        )
        try:
            plain_obs, _ = plain_env.reset(seed=17)
            diagnostic_obs, _ = diagnostic_env.reset(seed=17)
            action = np.linspace(-0.8, 0.8, plain_env.num_joints)

            plain_transition = plain_env.step(action)
            diagnostic_transition = diagnostic_env.step(action)

            np.testing.assert_array_equal(
                plain_obs["policy"], diagnostic_obs["policy"]
            )
            np.testing.assert_array_equal(
                plain_transition[0]["policy"],
                diagnostic_transition[0]["policy"],
            )
            np.testing.assert_array_equal(
                plain_transition[0]["privileged"],
                diagnostic_transition[0]["privileged"],
            )
            assert plain_transition[1:4] == diagnostic_transition[1:4]
            np.testing.assert_array_equal(
                plain_env.mjData.qpos, diagnostic_env.mjData.qpos
            )
            np.testing.assert_array_equal(
                plain_env.mjData.qvel, diagnostic_env.mjData.qvel
            )
        finally:
            plain_env.close()
            diagnostic_env.close()

    def test_deterministic_push_schedule_uses_zero_based_control_steps(self):
        delta = np.array([0.2, -0.1, 0.05, 0.08, -0.06, 0.12])
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(
                enable=False, push_robots=False
            ),
            enable_substep_diagnostics=True,
            deterministic_push_schedule={1: delta},
        )
        try:
            env.reset(seed=42)

            first_info = env.step(np.zeros(env.num_joints))[4]
            second_info = env.step(np.zeros(env.num_joints))[4]

            np.testing.assert_array_equal(
                first_info["substep_diagnostics"]["push_delta_qvel"],
                np.zeros(6),
            )
            np.testing.assert_array_equal(
                second_info["substep_diagnostics"]["push_delta_qvel"],
                delta,
            )
        finally:
            env.close()

    @pytest.mark.parametrize(
        "schedule",
        [
            {-1: np.zeros(6)},
            {0: np.zeros(5)},
            {0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan])},
        ],
    )
    def test_invalid_deterministic_push_schedule_is_rejected(self, schedule):
        with pytest.raises(ValueError, match="deterministic push"):
            QuadrupedVelocityTrackingEnv(
                robot="go2",
                scene="flat",
                deterministic_push_schedule=schedule,
            )


class TestNominalPlantAlignment:
    """Ensure the no-DR RL env matches the quadruped MPC plant."""

    @staticmethod
    def _geom_friction(model, geom_name: str) -> np.ndarray:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert geom_id >= 0, f"Geom '{geom_name}' not found"
        return model.geom_friction[geom_id].copy()

    def test_no_dr_matches_generation_contact_plant(self):
        rl_env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
        )
        mpc_env = QuadrupedEnv(
            robot="go2",
            scene="flat",
            sim_dt=1 / 200,
            ref_base_lin_vel=0.0,
            ground_friction_coeff=0.7,
            base_vel_command_type="human",
            state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
        )

        try:
            rl_env.reset(seed=0)
            mpc_env.reset(random=False)

            rl_env.assert_generation_contact_friction_matches()

            for geom_name in ["floor", "FL", "FR", "RL", "RR"]:
                np.testing.assert_allclose(
                    self._geom_friction(rl_env.mjModel, geom_name),
                    self._geom_friction(mpc_env.mjModel, geom_name),
                )

            # The RL env now applies Go2 sysID joint dynamics at load time, so
            # only the contact plant should match the nominal QuadrupedEnv.
            assert not np.allclose(rl_env.mjModel.dof_damping, mpc_env.mjModel.dof_damping)
            assert not np.allclose(rl_env.mjModel.dof_armature, mpc_env.mjModel.dof_armature)
            assert not np.allclose(
                rl_env.mjModel.dof_frictionloss,
                mpc_env.mjModel.dof_frictionloss,
            )
            np.testing.assert_allclose(rl_env.mjModel.body_mass, mpc_env.mjModel.body_mass)
            np.testing.assert_allclose(rl_env.mjModel.geom_size, mpc_env.mjModel.geom_size)
            assert int(rl_env.mjModel.opt.cone) == int(mpc_env.mjModel.opt.cone)
            assert rl_env.mjModel.opt.iterations == mpc_env.mjModel.opt.iterations
            assert rl_env.mjModel.opt.ls_iterations == mpc_env.mjModel.opt.ls_iterations
        finally:
            rl_env.close()
            mpc_env.close()

    def test_no_dr_reset_restores_nominal_contact_friction(self):
        rl_env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
        )

        try:
            rl_env.reset(seed=0)
            rl_env.mjModel.geom_friction[:, :] = np.array([1.1, 0.1, 0.01], dtype=np.float64)
            rl_env.reset(seed=1)
            rl_env.assert_generation_contact_friction_matches()
        finally:
            rl_env.close()

    def test_callback_temp_env_needs_no_friction_patch(self):
        temp_env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            render_mode=None,
            domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
            simple_reward=True,
        )

        try:
            _assert_quadruped_generation_friction(temp_env)
            temp_env.reset(seed=0)
            _assert_quadruped_generation_friction(temp_env)
        finally:
            temp_env.close()


class TestStartupDomainRandomizationPatch:
    """Tests for portable startup DR patch sampling and replay."""

    def test_default_no_push_matches_expected_rl_dr(self):
        resolved_type, cfg = resolve_startup_domain_rand_config("default_no_push")
        assert resolved_type == "default_no_push"
        assert cfg.added_mass_range == (0.0, 0.0)
        assert cfg.push_robots is False
        assert cfg.obs_noise_level > 0.0
        assert not hasattr(cfg, "controller_delay_range")

    def test_patch_sampling_is_deterministic(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
        )
        try:
            resolved_type, dr_cfg = resolve_startup_domain_rand_config("half_no_push")
            base_body_id = env._base_body_id
            patch_a = sample_startup_domain_rand_patch(
                env.mjModel,
                dr_cfg,
                rng=np.random.RandomState(123),
                dr_config_type=resolved_type,
                dr_seed=123,
                base_body_id=base_body_id,
                nominal_kp=env._nominal_kp,
                nominal_kd=env._nominal_kd,
                num_joints=env.num_joints,
            )
            patch_b = sample_startup_domain_rand_patch(
                env.mjModel,
                dr_cfg,
                rng=np.random.RandomState(123),
                dr_config_type=resolved_type,
                dr_seed=123,
                base_body_id=base_body_id,
                nominal_kp=env._nominal_kp,
                nominal_kd=env._nominal_kd,
                num_joints=env.num_joints,
            )
            for key in (
                "dr_patch_geom_friction",
                "dr_patch_body_mass",
                "dr_patch_body_ipos",
                "dr_patch_dof_damping",
                "dr_patch_dof_armature",
                "dr_patch_dof_frictionloss",
                "dr_patch_actuator_ctrlrange",
                "dr_patch_actuator_forcerange",
                "dr_torque_limits",
                "dr_encoder_bias",
                "dr_realized_kp",
                "dr_realized_kd",
            ):
                np.testing.assert_allclose(patch_a[key], patch_b[key])
            assert patch_a["dr_config_type"] == patch_b["dr_config_type"] == resolved_type
            assert patch_a["dr_seed"] == patch_b["dr_seed"] == 123
            assert patch_a["dr_motor_strength_scale"] == pytest.approx(
                patch_b["dr_motor_strength_scale"]
            )
            assert patch_a["dr_added_mass_kg"] == pytest.approx(patch_b["dr_added_mass_kg"])
            np.testing.assert_array_equal(
                patch_a["dr_applied_fields"],
                patch_b["dr_applied_fields"],
            )
        finally:
            env.close()

    def test_apply_patch_changes_targeted_model_arrays(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
        )
        try:
            custom_cfg = DomainRandomizationConfig(
                enable=True,
                friction_range=(0.31, 0.32),
                added_mass_range=(0.1, 0.2),
                com_displacement_range=(-0.02, 0.02),
                joint_damping_scale_range=(0.5, 0.6),
                joint_armature_scale_range=(1.4, 1.5),
                joint_friction_range=(0.01, 0.02),
                motor_strength_range=(0.6, 0.7),
                encoder_bias_range=(0.0, 0.0),
                push_robots=False,
            )
            patch = sample_startup_domain_rand_patch(
                env.mjModel,
                custom_cfg,
                rng=np.random.RandomState(7),
                dr_config_type="custom-test",
                dr_seed=7,
                base_body_id=env._base_body_id,
                nominal_kp=env._nominal_kp,
                nominal_kd=env._nominal_kd,
                num_joints=env.num_joints,
            )

            nominal_body_mass = env.mjModel.body_mass.copy()
            nominal_body_ipos = env.mjModel.body_ipos.copy()
            nominal_damping = env.mjModel.dof_damping.copy()
            nominal_armature = env.mjModel.dof_armature.copy()
            nominal_frictionloss = env.mjModel.dof_frictionloss.copy()
            nominal_torque_limits = env.torque_limits.copy()

            env.torque_limits = apply_startup_domain_rand_patch(env.mjModel, env.mjData, patch)

            assert not np.allclose(env.mjModel.body_mass, nominal_body_mass)
            assert not np.allclose(env.mjModel.body_ipos, nominal_body_ipos)
            assert not np.allclose(env.mjModel.dof_damping, nominal_damping)
            assert not np.allclose(env.mjModel.dof_armature, nominal_armature)
            assert not np.allclose(env.mjModel.dof_frictionloss, nominal_frictionloss)
            assert not np.allclose(env.torque_limits, nominal_torque_limits)
            np.testing.assert_allclose(env.torque_limits, patch["dr_torque_limits"])
        finally:
            env.close()

    def test_disabled_dr_generator_uses_new_direct_transition_schema(self):
        dr_disabled = generate_dr_mpx_trajectory(
            seed=5,
            domain_rand_config_type="disabled",
            mpc=FakeQuadrupedMPC(),
            episode_length=3,
            verbose=0,
            render=False,
        )

        assert dr_disabled["dr_enabled"] is False
        assert dr_disabled["dr_config_type"] == "disabled"
        assert int(dr_disabled["controller_delay_steps"]) == 0
        assert float(dr_disabled["controller_delay_s"]) == pytest.approx(0.0)
        assert dr_disabled["dr_added_mass_kg"] == pytest.approx(0.0)
        assert "body_mass" not in dr_disabled["dr_applied_fields"].tolist()
        assert "policy_obs" in dr_disabled
        assert "next_policy_obs" in dr_disabled
        assert "actions" in dr_disabled
        assert "rewards" in dr_disabled
        assert dr_disabled["policy_obs"].shape[0] == 3
        assert dr_disabled["next_policy_obs"].shape[0] == 3
        assert dr_disabled["raw_actions"].shape == (3, 12)
        assert dr_disabled["action_clipping_mask"].shape == (3, 12)
        assert dr_disabled["next_qpos_ctrl"].shape == (3, 19)
        assert dr_disabled["next_qvel_ctrl"].shape == (3, 18)
        assert dr_disabled["requested_torques_ctrl"].shape == (3, 4, 12)
        assert dr_disabled["applied_torques_ctrl"].shape == (3, 4, 12)
        assert dr_disabled["torque_saturation_mask"].shape == (3, 4, 12)
        assert dr_disabled["foot_contacts_substeps"].shape == (3, 4, 4)
        assert dr_disabled["non_foot_ground_contact_substeps"].shape == (3, 4)
        assert dr_disabled["push_delta_qvel"].shape == (3, 6)
        assert extract_startup_domain_rand_patch(dr_disabled) is not None

    def test_dr_replay_matches_saved_rollout_for_short_horizon(self):
        traj = generate_dr_mpx_trajectory(
            seed=11,
            domain_rand_config_type="default_no_push",
            mpc=FakeQuadrupedMPC(),
            episode_length=2,
            verbose=0,
            render=False,
        )
        dr_patch = extract_startup_domain_rand_patch(traj)

        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
        )
        try:
            env.reset(seed=0)
            _apply_quadruped_trajectory_dr_patch(env, dr_patch)
            env.mjData.qpos[:] = traj["qpos"][:, 0]
            env.mjData.qvel[:] = traj["qvel"][:, 0]
            env.mjData.ctrl[:] = 0.0
            env.mjData.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(env.mjModel, env.mjData)

            max_qpos_err = 0.0
            max_qvel_err = 0.0
            for sim_idx in range(traj["tau_applied"].shape[1]):
                torques = traj["tau_applied"][:, sim_idx].copy()
                torques = np.clip(torques, env.torque_limits[:, 0], env.torque_limits[:, 1])
                env.mjData.ctrl[:] = torques
                mujoco.mj_step(env.mjModel, env.mjData)
                qpos_err = float(np.max(np.abs(env.mjData.qpos - traj["qpos"][:, sim_idx + 1])))
                qvel_err = float(np.max(np.abs(env.mjData.qvel - traj["qvel"][:, sim_idx + 1])))
                max_qpos_err = max(max_qpos_err, qpos_err)
                max_qvel_err = max(max_qvel_err, qvel_err)

            assert max_qpos_err < 2e-3, f"max qpos replay error too large: {max_qpos_err}"
            assert max_qvel_err < 0.12, f"max qvel replay error too large: {max_qvel_err}"
        finally:
            env.close()

    def test_nominal_replay_path_restores_nominal_model_after_dr(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig(enable=False, push_robots=False),
        )
        try:
            env.reset(seed=0)
            nominal_body_mass = env.mjModel.body_mass.copy()
            nominal_body_ipos = env.mjModel.body_ipos.copy()
            nominal_damping = env.mjModel.dof_damping.copy()

            _, dr_cfg = resolve_startup_domain_rand_config("default_no_push")
            patch = sample_startup_domain_rand_patch(
                env.mjModel,
                dr_cfg,
                rng=np.random.RandomState(9),
                dr_config_type="default_no_push",
                dr_seed=9,
                base_body_id=env._base_body_id,
                nominal_kp=env._nominal_kp,
                nominal_kd=env._nominal_kd,
                num_joints=env.num_joints,
            )
            _apply_quadruped_trajectory_dr_patch(env, patch)
            assert not np.allclose(env.mjModel.body_ipos, nominal_body_ipos)

            _apply_quadruped_trajectory_dr_patch(env, None)
            np.testing.assert_allclose(env.mjModel.body_mass, nominal_body_mass)
            np.testing.assert_allclose(env.mjModel.body_ipos, nominal_body_ipos)
            np.testing.assert_allclose(env.mjModel.dof_damping, nominal_damping)
            env.assert_generation_contact_friction_matches()
        finally:
            env.close()


# ═══════════════════════════════════════════════════════════════════════════
# 4. RL-Matched DR Generator / Injection Path
# ═══════════════════════════════════════════════════════════════════════════


class DummyLogger:
    def record(self, *args, **kwargs):
        del args, kwargs


class DummyCallbackModel:
    def __init__(self, replay_buffer, num_envs):
        self.replay_buffer = replay_buffer
        self._env = SimpleNamespace(num_envs=num_envs)
        self.logger = DummyLogger()

    def get_env(self):
        return self._env


class TestRLMatchedDRGenerator:
    def test_generation_smoke_writes_new_schema_and_manifest(self, tmp_path):
        gen_traj_quadruped_dr(
            num_trajectories=1,
            episode_length=2,
            start_seed=3,
            output_dir=tmp_path,
            max_attempts=5,
            verbose=0,
            render=False,
            domain_rand_config_type="default_no_push",
            mpc=FakeQuadrupedMPC(),
        )

        generated_files = sorted(tmp_path.glob("*.npz"))
        assert len(generated_files) == 1
        traj = np.load(generated_files[0], allow_pickle=True)

        assert int(traj["controller_delay_steps"]) == 0
        assert float(traj["controller_delay_s"]) == pytest.approx(0.0)
        assert float(traj["dr_added_mass_kg"]) == pytest.approx(0.0)
        assert "body_mass" not in traj["dr_applied_fields"].tolist()
        assert "dr_encoder_bias" in traj.files
        assert "policy_obs" in traj.files
        assert "next_policy_obs" in traj.files
        assert "actions" in traj.files
        assert "rewards" in traj.files
        assert "terminated_ctrl" in traj.files
        assert traj["policy_obs"].shape == (2, 45)
        assert traj["next_policy_obs"].shape == (2, 45)
        assert traj["privileged_obs"].shape == (2, 3)
        assert traj["next_privileged_obs"].shape == (2, 3)

        manifest_path = tmp_path / "generation_manifest.jsonl"
        assert manifest_path.exists()
        manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(manifest_lines) >= 1
        manifest_record = json.loads(manifest_lines[-1])
        assert manifest_record["success"] is True
        assert manifest_record["controller_delay_steps"] == 0
        assert manifest_record["dr_summary"]["dr_added_mass_kg"] == pytest.approx(0.0)

    def test_saved_transition_parity(self):
        traj = generate_dr_mpx_trajectory(
            seed=13,
            domain_rand_config_type="default_no_push",
            mpc=FakeQuadrupedMPC(),
            episode_length=3,
            verbose=0,
            render=False,
        )
        np.testing.assert_allclose(traj["policy_obs"][:, 6:9], traj["commands_ctrl"])
        np.testing.assert_allclose(traj["next_policy_obs"][:-1], traj["policy_obs"][1:])

    def test_old_vs_new_dataset_schema_contrast(self, tmp_path):
        old_dir = Path("data/quadruped_dr/default_no_push_old")
        old_files = sorted(old_dir.glob("*.npz")) if old_dir.exists() else []
        if not old_files:
            pytest.skip("Legacy default_no_push_old dataset not present")

        gen_traj_quadruped_dr(
            num_trajectories=1,
            episode_length=2,
            start_seed=21,
            output_dir=tmp_path,
            max_attempts=5,
            verbose=0,
            render=False,
            domain_rand_config_type="default_no_push",
            mpc=FakeQuadrupedMPC(),
        )

        old_traj = np.load(old_files[0], allow_pickle=True)
        new_traj = np.load(sorted(tmp_path.glob("*.npz"))[0], allow_pickle=True)

        assert "dr_encoder_bias" not in old_traj.files
        assert "policy_obs" not in old_traj.files
        assert "next_policy_obs" not in old_traj.files
        assert "dr_encoder_bias" in new_traj.files
        assert "policy_obs" in new_traj.files
        assert "next_policy_obs" in new_traj.files

    def test_direct_transition_injection_path_skips_legacy_replay(self, tmp_path, monkeypatch):
        gen_traj_quadruped_dr(
            num_trajectories=1,
            episode_length=2,
            start_seed=34,
            output_dir=tmp_path,
            max_attempts=5,
            verbose=0,
            render=False,
            domain_rand_config_type="default_no_push",
            mpc=FakeQuadrupedMPC(),
        )

        callback = PercentMPCInjectCallback(
            domain="quadruped",
            task="velocity_tracking",
            target_percentage=10,
            data_dir=str(tmp_path),
            random_select=False,
            trajectory_files=[sorted(tmp_path.glob("*.npz"))[0].name],
            verbose=0,
        )

        replay_buffer = TaggedDictReplayBuffer(
            buffer_size=32,
            observation_space=spaces.Dict(
                {
                    "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64),
                    "privileged": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                }
            ),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float64),
            device="cpu",
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
        )
        callback.init_callback(DummyCallbackModel(replay_buffer=replay_buffer, num_envs=1))

        def _legacy_replay_should_not_run(*args, **kwargs):
            raise AssertionError("legacy quadruped replay path should not run for new DR files")

        monkeypatch.setattr(callback, "_replay_quadruped_trajectory", _legacy_replay_should_not_run)
        callback._inject_mpc_trajectories()

        saved_traj = np.load(sorted(tmp_path.glob("*.npz"))[0], allow_pickle=True)
        assert replay_buffer.size() == 1
        assert replay_buffer.get_mpc_percentage() == pytest.approx(100.0)
        np.testing.assert_allclose(
            replay_buffer.observations["policy"][0, 0],
            saved_traj["policy_obs"][0],
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. play_quad.py Utilities
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
