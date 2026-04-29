"""Tests for Go2 sysID integration and sysID-aware DR presets."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import mpx.config.config_go2 as go2_config
from mpc_rl.envs.domain_randomization import (
    FLOOR_FRICTION_TARGET_GEOM_NAMES,
    DomainRandomizationConfig,
    resolve_startup_domain_rand_config,
    sample_startup_domain_rand_patch,
)
from mpc_rl.envs.go2_sysid import (
    GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS,
    apply_go2_sysid_joint_dynamics,
    assert_go2_sysid_joint_dynamics,
    looks_like_go2_model,
    parse_go2_sysid_report,
)
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.planner.gen_traj_data_mpx_dr import generate_trajectory as generate_dr_mpx_trajectory
from mpx.utils.mpc_wrapper import MPCControllerWrapper


class FakeQuadrupedMPC:
    """Small deterministic MPC stub for short trajectory smoke tests."""

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


def _joint_dynamics(model: mujoco.MjModel, joint_name: str) -> dict[str, float]:
    """Read armature, frictionloss, and damping for one MuJoCo joint."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    dof_index = model.jnt_dofadr[joint_id]
    return {
        "armature": float(model.dof_armature[dof_index]),
        "frictionloss": float(model.dof_frictionloss[dof_index]),
        "damping": float(model.dof_damping[dof_index]),
    }


def _target_friction_geom_ids(model: mujoco.MjModel, target_geom_names: tuple[str, ...]) -> set[int]:
    """Resolve the target floor/contact-surface geom IDs for a model."""
    normalized = {name.lower() for name in target_geom_names}
    target_ids = set()
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and geom_name.lower() in normalized:
            target_ids.add(geom_id)
    return target_ids


class TestGo2SysIdTable:
    def test_checked_in_table_matches_report_html(self):
        parsed = parse_go2_sysid_report()

        assert len(parsed) == 12
        assert sum(len(fields) for fields in parsed.values()) == 36
        assert parsed == GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS

    def test_runtime_patch_writes_expected_joint_dynamics(self):
        model = mujoco.MjModel.from_xml_path(str(go2_config.model_path))
        assert looks_like_go2_model(model) is True

        apply_go2_sysid_joint_dynamics(model)

        for joint_name, expected in GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS.items():
            assert _joint_dynamics(model, joint_name) == pytest.approx(expected)

    def test_assert_helper_rejects_unpatched_go2_model(self):
        model = mujoco.MjModel.from_xml_path(str(go2_config.model_path))

        with pytest.raises(AssertionError, match="Go2 sysID joint dynamics"):
            assert_go2_sysid_joint_dynamics(model)

    def test_assert_helper_accepts_patched_go2_model(self):
        model = mujoco.MjModel.from_xml_path(str(go2_config.model_path))

        apply_go2_sysid_joint_dynamics(model)
        assert_go2_sysid_joint_dynamics(model)


class TestGo2SysIdIntegration:
    def test_velocity_tracking_env_uses_identified_nominal_dynamics(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
        )
        try:
            for joint_name, expected in GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS.items():
                joint_id = mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                dof_index = env.mjModel.jnt_dofadr[joint_id]

                assert _joint_dynamics(env.mjModel, joint_name) == pytest.approx(expected)
                assert env._nominal_dof_armature[dof_index] == pytest.approx(expected["armature"])
                assert env._nominal_dof_frictionloss[dof_index] == pytest.approx(
                    expected["frictionloss"]
                )
                assert env._nominal_dof_damping[dof_index] == pytest.approx(expected["damping"])
        finally:
            env.close()

    def test_velocity_tracking_env_can_disable_identified_nominal_dynamics(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
            use_go2_sysid=False,
        )
        try:
            with pytest.raises(AssertionError, match="Go2 sysID joint dynamics"):
                assert_go2_sysid_joint_dynamics(env.mjModel)
        finally:
            env.close()

    def test_mpx_wrapper_uses_identified_joint_dynamics(self):
        controller = MPCControllerWrapper(go2_config)

        representative_joints = ("FL_hip_joint", "FR_thigh_joint", "RR_calf_joint")
        for joint_name in representative_joints:
            assert _joint_dynamics(controller.model, joint_name) == pytest.approx(
                GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS[joint_name]
            )

    def test_mpx_wrapper_can_disable_identified_joint_dynamics(self):
        raw_model = mujoco.MjModel.from_xml_path(str(go2_config.model_path))
        controller = MPCControllerWrapper(go2_config, use_go2_sysid=False)

        representative_joints = ("FL_hip_joint", "FR_thigh_joint", "RR_calf_joint")
        for joint_name in representative_joints:
            assert _joint_dynamics(controller.model, joint_name) == pytest.approx(
                _joint_dynamics(raw_model, joint_name)
            )


class TestSysIdAwareDomainRandomizationPresets:
    def test_resolve_sysid_floor_only_preset(self):
        resolved_type, cfg = resolve_startup_domain_rand_config("sysid_floor_only_no_push")

        assert resolved_type == "sysid_floor_only_no_push"
        assert cfg.enable is True
        assert cfg.friction_target_geom_names == FLOOR_FRICTION_TARGET_GEOM_NAMES
        assert cfg.com_displacement_range == (0.0, 0.0)
        assert cfg.encoder_bias_range == (0.0, 0.0)
        assert cfg.obs_noise_level == pytest.approx(0.0)
        assert cfg.push_robots is False

    def test_resolve_sysid_floor_sensing_preset(self):
        _, floor_only = resolve_startup_domain_rand_config("sysid_floor_only_no_push")
        resolved_type, floor_sensing = resolve_startup_domain_rand_config(
            "sysid_floor_sensing_no_push"
        )

        assert resolved_type == "sysid_floor_sensing_no_push"
        assert floor_sensing.enable is True
        assert floor_sensing.friction_target_geom_names == FLOOR_FRICTION_TARGET_GEOM_NAMES
        assert floor_sensing.com_displacement_range == (0.0, 0.0)
        assert floor_sensing.push_robots is False
        assert floor_sensing.obs_noise_level > floor_only.obs_noise_level
        assert floor_sensing.encoder_bias_range != floor_only.encoder_bias_range
        assert floor_sensing.joint_damping_scale_range == (1.0, 1.0)
        assert floor_sensing.joint_armature_scale_range == (1.0, 1.0)
        assert floor_sensing.joint_friction_range == (0.0, 0.0)

    def test_disabled_preset_remains_no_dr(self):
        resolved_type, cfg = resolve_startup_domain_rand_config("disabled")

        assert resolved_type == "disabled"
        assert cfg.enable is False

    def test_floor_only_patch_randomizes_only_target_surface_friction(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
        )
        try:
            resolved_type, cfg = resolve_startup_domain_rand_config("sysid_floor_only_no_push")
            patch = sample_startup_domain_rand_patch(
                env.mjModel,
                cfg,
                rng=np.random.RandomState(123),
                dr_config_type=resolved_type,
                dr_seed=123,
                base_body_id=env._base_body_id,
                nominal_kp=env._nominal_kp,
                nominal_kd=env._nominal_kd,
                num_joints=env.num_joints,
            )

            target_geom_ids = _target_friction_geom_ids(env.mjModel, FLOOR_FRICTION_TARGET_GEOM_NAMES)
            changed_geom_ids = {
                geom_id
                for geom_id in range(env.mjModel.ngeom)
                if not np.isclose(
                    patch["dr_patch_geom_friction"][geom_id, 0],
                    env.mjModel.geom_friction[geom_id, 0],
                )
            }

            assert target_geom_ids
            assert changed_geom_ids == target_geom_ids
            np.testing.assert_allclose(
                patch["dr_patch_geom_friction"][:, 1:],
                env.mjModel.geom_friction[:, 1:],
            )
            np.testing.assert_allclose(patch["dr_patch_body_mass"], env.mjModel.body_mass)
            np.testing.assert_allclose(patch["dr_patch_body_ipos"], env.mjModel.body_ipos)
            np.testing.assert_allclose(patch["dr_patch_dof_damping"], env.mjModel.dof_damping)
            np.testing.assert_allclose(patch["dr_patch_dof_armature"], env.mjModel.dof_armature)
            np.testing.assert_allclose(
                patch["dr_patch_dof_frictionloss"],
                env.mjModel.dof_frictionloss,
            )
            np.testing.assert_allclose(patch["dr_encoder_bias"], 0.0)
            assert patch["dr_applied_fields"].tolist() == ["geom_friction"]
        finally:
            env.close()

    def test_floor_sensing_patch_differs_from_floor_only_only_in_encoder_bias(self):
        env = QuadrupedVelocityTrackingEnv(
            robot="go2",
            scene="flat",
            domain_rand_cfg=DomainRandomizationConfig.disabled(),
        )
        try:
            _, floor_only_cfg = resolve_startup_domain_rand_config("sysid_floor_only_no_push")
            _, floor_sensing_cfg = resolve_startup_domain_rand_config(
                "sysid_floor_sensing_no_push"
            )

            patch_floor_only = sample_startup_domain_rand_patch(
                env.mjModel,
                floor_only_cfg,
                rng=np.random.RandomState(321),
                dr_config_type="sysid_floor_only_no_push",
                dr_seed=321,
                base_body_id=env._base_body_id,
                nominal_kp=env._nominal_kp,
                nominal_kd=env._nominal_kd,
                num_joints=env.num_joints,
            )
            patch_floor_sensing = sample_startup_domain_rand_patch(
                env.mjModel,
                floor_sensing_cfg,
                rng=np.random.RandomState(321),
                dr_config_type="sysid_floor_sensing_no_push",
                dr_seed=321,
                base_body_id=env._base_body_id,
                nominal_kp=env._nominal_kp,
                nominal_kd=env._nominal_kd,
                num_joints=env.num_joints,
            )

            np.testing.assert_allclose(
                patch_floor_only["dr_patch_geom_friction"],
                patch_floor_sensing["dr_patch_geom_friction"],
            )
            np.testing.assert_allclose(
                patch_floor_only["dr_patch_body_mass"],
                patch_floor_sensing["dr_patch_body_mass"],
            )
            np.testing.assert_allclose(
                patch_floor_only["dr_patch_body_ipos"],
                patch_floor_sensing["dr_patch_body_ipos"],
            )
            np.testing.assert_allclose(
                patch_floor_only["dr_patch_dof_damping"],
                patch_floor_sensing["dr_patch_dof_damping"],
            )
            np.testing.assert_allclose(
                patch_floor_only["dr_patch_dof_armature"],
                patch_floor_sensing["dr_patch_dof_armature"],
            )
            np.testing.assert_allclose(
                patch_floor_only["dr_patch_dof_frictionloss"],
                patch_floor_sensing["dr_patch_dof_frictionloss"],
            )
            assert patch_floor_only["dr_applied_fields"].tolist() == ["geom_friction"]
            assert patch_floor_sensing["dr_applied_fields"].tolist() == [
                "geom_friction",
                "encoder_bias",
            ]
            assert np.allclose(patch_floor_only["dr_encoder_bias"], 0.0)
            assert not np.allclose(patch_floor_sensing["dr_encoder_bias"], 0.0)
        finally:
            env.close()


class TestSysIdAwareGenerationSmoke:
    def test_generator_uses_floor_only_preset_without_joint_dynamics_dr(self):
        traj = generate_dr_mpx_trajectory(
            seed=7,
            domain_rand_config_type="sysid_floor_only_no_push",
            mpc=FakeQuadrupedMPC(),
            episode_length=2,
            verbose=0,
            render=False,
        )

        assert traj["dr_config_type"] == "sysid_floor_only_no_push"
        assert traj["dr_applied_fields"].tolist() == ["geom_friction"]
        assert np.allclose(traj["dr_encoder_bias"], 0.0)
        assert traj["dr_summary_geom_friction"] == pytest.approx(
            float(traj["dr_summary_geom_friction"])
        )
