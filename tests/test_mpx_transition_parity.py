import json
from types import SimpleNamespace

import numpy as np
import pytest

import mpx.config.config_go2 as go2_config
from mpc_rl.common.mpc_inject_callbacks import (
    _validate_quadruped_direct_action_interface,
)
from mpc_rl.envs.action_interfaces import (
    DEFAULT_ACTION_INTERFACE_ID,
    MPX_BOUND_ACTION_INTERFACE_ID,
    MPX_BOUND_ENV_STEP_MODE,
    resolve_action_interface,
)
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.planner.check_mpx_transition_parity import (
    DEFAULT_TOLERANCES,
    MPX_BOUND_ACTION_INTERFACE_TOLERANCES,
    _make_replay_env,
    _metric_passes,
    _numeric_metrics,
    _replay_saved_actions,
)
from mpc_rl.planner.gen_traj_data_mpx_dr import (
    ENV_STEP_LPF_INVERSE_MODE,
    _configure_mpc_duty_factor,
    _mpx_to_env_step_raw_action,
    generate_trajectory,
)


class _ZeroMPC:
    def __init__(self):
        self.robot_height = go2_config.robot_height
        self.duty_factor = 0.5

    def reset(self, qpos, qvel):
        del qpos, qvel

    def run(self, qpos, qvel, mpx_input, contact):
        del qpos, qvel, mpx_input, contact
        return (
            np.zeros(go2_config.n_joints, dtype=np.float64),
            np.asarray(go2_config.q0, dtype=np.float64),
            np.zeros(go2_config.n_joints, dtype=np.float64),
        )


class _LargeFeedforwardMPC(_ZeroMPC):
    def run(self, qpos, qvel, mpx_input, contact):
        del qpos, qvel, mpx_input, contact
        return (
            np.full(go2_config.n_joints, 100.0, dtype=np.float64),
            np.asarray(go2_config.q0, dtype=np.float64),
            np.zeros(go2_config.n_joints, dtype=np.float64),
        )


def test_command_ramp_smoothly_transitions_gait_parameters():
    mpc = SimpleNamespace(duty_factor=0.5, step_height=0.03)

    _configure_mpc_duty_factor(
        np.array([0.25, 0.0, 0.0]),
        mpc,
        moving_duty_factor=0.5,
        moving_step_height=0.06,
        move_on_any_nonzero_command=True,
        transition_progress=0.5,
    )

    assert mpc.duty_factor == pytest.approx(0.75)
    assert mpc.step_height == pytest.approx(0.03)

    _configure_mpc_duty_factor(
        np.array([0.5, 0.0, 0.0]),
        mpc,
        moving_duty_factor=0.5,
        moving_step_height=0.06,
        move_on_any_nonzero_command=True,
        transition_progress=1.0,
    )

    assert mpc.duty_factor == pytest.approx(0.5)
    assert mpc.step_height == pytest.approx(0.06)


def test_versioned_action_interface_preserves_default_and_exposes_opt_in_mode():
    default = resolve_action_interface(DEFAULT_ACTION_INTERFACE_ID)
    mpx_bound = resolve_action_interface(MPX_BOUND_ACTION_INTERFACE_ID)

    assert default.action_dim == mpx_bound.action_dim == 12
    assert default.action_scale == 0.5
    assert default.action_lpf_cutoff_hz == 5.0
    assert default.required_mpx_conversion_mode is None
    assert mpx_bound.action_scale == 1.0
    assert mpx_bound.action_lpf_cutoff_hz is None
    assert mpx_bound.required_mpx_conversion_mode == MPX_BOUND_ENV_STEP_MODE
    assert mpx_bound.mpx_target_formula == (
        "q_target = q_des + tau_ff / kp_realized"
    )
    assert mpx_bound.generation_advance == "env.step(raw_action)"


def test_parity_replay_env_uses_selected_scale_one_no_lpf_interface():
    env = _make_replay_env(
        "disabled", {}, MPX_BOUND_ACTION_INTERFACE_ID
    )
    try:
        assert env.action_scale == 1.0
        assert env.action_lpf_cutoff_hz is None
        assert env.action_lpf_alpha == 1.0
    finally:
        env.close()


def test_numeric_metrics_report_max_rms_and_step_drift():
    expected = np.zeros((2, 2), dtype=np.float64)
    actual = np.array([[1.0, -1.0], [2.0, 0.0]], dtype=np.float64)

    metrics = _numeric_metrics(expected, actual)

    assert metrics["shape_match"] is True
    assert metrics["finite"] is True
    assert metrics["max_abs"] == 2.0
    assert metrics["rms"] == np.sqrt(1.5)
    assert metrics["per_control_step_max_abs"] == [1.0, 2.0]


def test_numeric_metrics_reject_shape_and_nonfinite_values():
    shape_mismatch = _numeric_metrics(np.zeros(2), np.zeros(3))
    nonfinite = _numeric_metrics(np.zeros(2), np.array([0.0, np.nan]))

    assert shape_mismatch["shape_match"] is False
    assert nonfinite["finite"] is False
    assert not _metric_passes(nonfinite, {"max_abs": 1.0, "rms": 1.0})


def test_predeclared_contract_covers_required_fields_and_scenarios():
    contract = json.loads(DEFAULT_TOLERANCES.read_text(encoding="utf-8"))

    assert contract["declared_before_measurement"] is True
    assert contract["conversion_mode_under_test"] == (
        "inferred_action_direct_torque_v1"
    )
    assert {scenario["name"] for scenario in contract["scenarios"]} == {
        "nominal",
        "deterministic_nontrivial_dr",
    }
    assert contract["scenarios"][1]["require_push_event"] is True
    assert set(contract["numeric_fields"]) == {
        "qpos",
        "qvel",
        "policy_observation",
        "privileged_observation",
        "reward",
        "applied_torque",
        "push_delta_qvel",
    }
    assert contract["exact_fields"]["termination_mismatch_count"] == 0
    assert contract["mask_tolerances"]["replay_action_clipping_fraction"] == 0.0


def test_v2_changes_push_coverage_without_changing_thresholds():
    v1 = json.loads(DEFAULT_TOLERANCES.read_text(encoding="utf-8"))
    v2_path = DEFAULT_TOLERANCES.with_name("transition_parity_tolerances_v2.json")
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))

    for key in (
        "numeric_fields",
        "exact_fields",
        "mask_tolerances",
        "generation_action_conversion_limits",
        "one_step_control_steps",
        "short_rollout_control_steps",
        "rollout_seed",
        "dr_seed_offset",
    ):
        assert v2[key] == v1[key]
    assert v2["thresholds_changed_from_v1"] is False
    assert len(v2["scenarios"][1]["deterministic_push_schedule"]) == 2


def test_env_step_evaluation_changes_only_conversion_mode_from_v2():
    v2_path = DEFAULT_TOLERANCES.with_name("transition_parity_tolerances_v2.json")
    env_step_path = DEFAULT_TOLERANCES.with_name(
        "transition_parity_env_step_tolerances_v2.json"
    )
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    env_step = json.loads(env_step_path.read_text(encoding="utf-8"))

    metadata_keys = {
        "declaration_id",
        "supersedes_for_push_coverage",
        "inherits_thresholds_from",
        "declared_at",
        "declared_before_v2_measurement",
        "declared_before_measurement",
        "thresholds_changed_from_v1",
        "thresholds_changed_from_v2",
        "conversion_mode_under_test",
        "acceptance_rule",
    }
    assert {
        key: value for key, value in env_step.items() if key not in metadata_keys
    } == {key: value for key, value in v2.items() if key not in metadata_keys}
    assert env_step["conversion_mode_under_test"] == "env_step_lpf_inverse_v1"
    assert env_step["thresholds_changed_from_v2"] is False


def test_scale_one_no_lpf_declaration_preserves_replay_thresholds_and_is_narrow():
    env_step_v2_path = DEFAULT_TOLERANCES.with_name(
        "transition_parity_env_step_tolerances_v2.json"
    )
    v2 = json.loads(env_step_v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(
        MPX_BOUND_ACTION_INTERFACE_TOLERANCES.read_text(encoding="utf-8")
    )

    assert v3["declared_before_measurement"] is True
    assert v3["replay_thresholds_changed_from_v2"] is False
    assert v3["action_conversion_limits_changed_from_v2"] is True
    assert v3["action_interface_id"] == MPX_BOUND_ACTION_INTERFACE_ID
    assert v3["action_interface"] == resolve_action_interface(
        MPX_BOUND_ACTION_INTERFACE_ID
    ).to_dict()
    assert v3["conversion_mode_under_test"] == MPX_BOUND_ENV_STEP_MODE
    assert v3["actuator_limits_modified"] is False
    for key in (
        "numeric_fields",
        "exact_fields",
        "mask_tolerances",
        "one_step_control_steps",
        "short_rollout_control_steps",
        "simulation_substeps_per_control_step",
        "rollout_seed",
        "dr_seed_offset",
        "scenarios",
    ):
        assert v3[key] == v2[key]
    assert v3["generation_action_conversion_limits_by_scenario"] == {
        "nominal": {
            "clipped_element_fraction": 0.0,
            "max_clip_magnitude": 0.0,
        },
        "deterministic_nontrivial_dr": {
            "clipped_element_fraction": 0.02,
            "max_clip_magnitude": 0.86,
        },
    }
    assert v3["required_clipping_diagnostics"] == ["scenario", "joint"]


def test_env_step_mapping_inverts_lpf_with_realized_gains():
    env = QuadrupedVelocityTrackingEnv(
        robot="go2",
        scene="flat",
        domain_rand_cfg=DomainRandomizationConfig(
            enable=False, push_robots=False
        ),
        enable_substep_diagnostics=True,
    )
    try:
        env.reset(seed=8)
        previous_filtered = env._filtered_q_target.copy()
        q_des = env.default_joint_pos + 0.01
        tau_ff = env.kp * 0.005

        raw_action, desired_filtered, raw_target = (
            _mpx_to_env_step_raw_action(env, tau_ff, q_des)
        )

        np.testing.assert_allclose(
            desired_filtered, q_des + tau_ff / env.kp
        )
        np.testing.assert_allclose(
            previous_filtered
            + env.action_lpf_alpha * (raw_target - previous_filtered),
            desired_filtered,
        )
        assert np.all(np.abs(raw_action) < 1.0)
        _, _, _, _, info = env.step(raw_action)
        np.testing.assert_allclose(
            info["substep_diagnostics"]["filtered_q_target"],
            desired_filtered,
            atol=1e-14,
            rtol=0.0,
        )
    finally:
        env.close()


def test_env_step_generator_transitions_replay_exactly():
    trajectory = generate_trajectory(
        seed=19,
        domain_rand_config_type="disabled",
        mpc=_ZeroMPC(),
        episode_length=3,
        verbose=0,
        render=False,
        action_conversion_mode=ENV_STEP_LPF_INVERSE_MODE,
    )
    replay = _replay_saved_actions(
        trajectory,
        domain_rand_config_type="disabled",
        deterministic_push_schedule={},
        seed=19,
        control_steps=3,
    )

    assert trajectory["action_conversion_mode"] == ENV_STEP_LPF_INVERSE_MODE
    assert replay["control_steps"] == 3
    np.testing.assert_array_equal(
        trajectory["next_qpos_ctrl"], replay["qpos"]
    )
    np.testing.assert_array_equal(
        trajectory["next_qvel_ctrl"], replay["qvel"]
    )
    np.testing.assert_array_equal(
        trajectory["next_policy_obs"], replay["policy_observation"]
    )
    np.testing.assert_array_equal(
        trajectory["applied_torques_ctrl"], replay["applied_torque"]
    )


def test_mpx_bound_interface_records_env_step_actions_and_replays_exactly():
    trajectory = generate_trajectory(
        seed=23,
        domain_rand_config_type="disabled",
        mpc=_ZeroMPC(),
        episode_length=3,
        verbose=0,
        render=False,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )

    assert trajectory["schema_version"] == 2
    assert trajectory["action_interface_id"] == MPX_BOUND_ACTION_INTERFACE_ID
    assert trajectory["action_conversion_mode"] == MPX_BOUND_ENV_STEP_MODE
    assert trajectory["action_scale"] == 1.0
    assert trajectory["action_lpf_cutoff_hz_is_none"] is True
    assert trajectory["action_lpf_cutoff_hz_value"] == 0.0
    assert trajectory["action_lpf_alpha"] == 1.0
    np.testing.assert_array_equal(
        trajectory["actions"], np.clip(trajectory["raw_actions"], -1.0, 1.0)
    )
    np.testing.assert_array_equal(
        trajectory["action_clipping_mask"],
        trajectory["raw_actions"] != trajectory["actions"],
    )

    replay = _replay_saved_actions(
        trajectory,
        domain_rand_config_type="disabled",
        deterministic_push_schedule={},
        seed=23,
        control_steps=3,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )
    np.testing.assert_array_equal(trajectory["next_qpos_ctrl"], replay["qpos"])
    np.testing.assert_array_equal(trajectory["next_qvel_ctrl"], replay["qvel"])
    np.testing.assert_array_equal(
        trajectory["next_policy_obs"], replay["policy_observation"]
    )


def test_mpx_bound_interface_records_clipped_raw_action_diagnostics():
    trajectory = generate_trajectory(
        seed=29,
        domain_rand_config_type="disabled",
        mpc=_LargeFeedforwardMPC(),
        episode_length=1,
        verbose=0,
        render=False,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )

    assert np.any(trajectory["action_clipping_mask"])
    np.testing.assert_array_equal(
        trajectory["actions"], np.clip(trajectory["raw_actions"], -1.0, 1.0)
    )
    np.testing.assert_allclose(
        trajectory["action_clipping_magnitude"],
        np.abs(trajectory["raw_actions"] - trajectory["actions"]),
    )


def test_direct_loader_rejects_action_interface_metadata_mismatch():
    trajectory = generate_trajectory(
        seed=31,
        domain_rand_config_type="disabled",
        mpc=_ZeroMPC(),
        episode_length=1,
        verbose=0,
        render=False,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )

    _validate_quadruped_direct_action_interface(
        trajectory,
        expected_action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )
    with pytest.raises(ValueError, match="action-interface mismatch"):
        _validate_quadruped_direct_action_interface(
            trajectory,
            expected_action_interface_id=DEFAULT_ACTION_INTERFACE_ID,
        )


def test_mpx_bound_interface_rejects_non_env_step_conversion():
    with pytest.raises(ValueError, match="requires action_conversion_mode"):
        generate_trajectory(
            seed=37,
            domain_rand_config_type="disabled",
            mpc=_ZeroMPC(),
            episode_length=1,
            verbose=0,
            render=False,
            action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
            action_conversion_mode=ENV_STEP_LPF_INVERSE_MODE,
        )
