import json

import numpy as np

from mpc_rl.planner.check_mpx_transition_parity import (
    DEFAULT_TOLERANCES,
    _metric_passes,
    _numeric_metrics,
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
