import json

import numpy as np

import mpx.config.config_go2 as go2_config
from mpc_rl.envs.domain_randomization import DomainRandomizationConfig
from mpc_rl.envs.velocity_tracking_env import QuadrupedVelocityTrackingEnv
from mpc_rl.planner.check_mpx_transition_parity import (
    DEFAULT_TOLERANCES,
    _metric_passes,
    _numeric_metrics,
    _replay_saved_actions,
)
from mpc_rl.planner.gen_traj_data_mpx_dr import (
    ENV_STEP_LPF_INVERSE_MODE,
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
