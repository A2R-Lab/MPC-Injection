import inspect
import json

import numpy as np
import pytest

import mpx.config.config_go2 as go2_config
from mpc_rl.envs.action_interfaces import (
    DEFAULT_ACTION_INTERFACE_ID,
    MPX_BOUND_ACTION_INTERFACE_ID,
)
from mpc_rl.planner.check_mpx_transition_parity import (
    validate_full_saved_action_replay,
)
from mpc_rl.planner.gen_traj_data_mpx_dr import (
    _bounding_filename,
    gen_traj_quadruped_dr,
    generate_trajectory,
)
from mpc_rl.planner.mpx_bounding_data import (
    build_command_schedule,
    make_commissioning_acceptance_declaration,
    measured_bound_metrics,
    validate_bounding_trajectory_data,
    validate_acceptance_declaration,
)


class RecordingBoundMPC:
    def __init__(
        self,
        config=go2_config,
        *,
        gait="bound_front_first",
        duty_factor=0.5,
        step_frequency_hz=3.0,
        step_height_m=0.03,
        enable_planned_contact_diagnostics=True,
        **kwargs,
    ):
        del kwargs
        self.config = config
        self.robot_height = go2_config.robot_height
        self.gait_name = gait
        self.initial_phase = np.array([0.0, 0.0, 0.5, 0.5])
        self.duty_factor = duty_factor
        self.step_freq = step_frequency_hz
        self.step_height = step_height_m
        self.enable_planned_contact_diagnostics = (
            enable_planned_contact_diagnostics
        )
        self.reset_count = 0
        self.inputs = []
        self._planned = None

    @property
    def gait_parameters(self):
        return {
            "gait_name": self.gait_name,
            "initial_phase": self.initial_phase.copy(),
            "duty_factor": self.duty_factor,
            "step_frequency_hz": self.step_freq,
            "step_height_m": self.step_height,
        }

    @property
    def planned_contact_schedule(self):
        return None if self._planned is None else self._planned.copy()

    def reset(self, qpos, qvel):
        del qpos, qvel
        self.reset_count += 1
        self.inputs = []
        self._planned = None

    def run(self, qpos, qvel, mpx_input, contact):
        del qpos, qvel, contact
        self.inputs.append(np.asarray(mpx_input).copy())
        self._planned = np.tile(np.array([[1, 1, 0, 0]], dtype=np.int8), (26, 1))
        return (
            np.zeros(go2_config.n_joints, dtype=np.float64),
            np.asarray(go2_config.q0, dtype=np.float64),
            np.zeros(go2_config.n_joints, dtype=np.float64),
        )


def _short_declaration(episode_length=4, ramp_steps=2):
    return make_commissioning_acceptance_declaration(
        target_command=[0.5, 0.0, 0.0],
        episode_length=episode_length,
        ramp_control_steps=ramp_steps,
        gait_name="bound_front_first",
        gait_duty_factor=0.5,
        gait_step_frequency_hz=3.0,
        gait_step_height_m=0.03,
    )


def test_fixed_speed_schedule_is_exact_ramp_then_hold():
    schedule = build_command_schedule(
        [0.5, 0.0, 0.0], episode_length=1000, ramp_control_steps=50
    )

    np.testing.assert_array_equal(
        schedule[:50, 0], np.arange(1, 51, dtype=np.float64) / 100.0
    )
    np.testing.assert_array_equal(schedule[50:, 0], 0.5)
    np.testing.assert_array_equal(schedule[:, 1:], 0.0)


@pytest.mark.parametrize(
    ("episode_length", "ramp_steps", "message"),
    [(0, 0, "positive"), (10, -1, "0 <="), (10, 11, "episode_length")],
)
def test_command_schedule_rejects_invalid_lengths(
    episode_length, ramp_steps, message
):
    with pytest.raises(ValueError, match=message):
        build_command_schedule(
            [0.5, 0.0, 0.0],
            episode_length=episode_length,
            ramp_control_steps=ramp_steps,
        )


def test_fixed_schedule_propagates_to_env_mpc_and_pickle_free_metadata():
    controller = RecordingBoundMPC()
    trajectory = generate_trajectory(
        seed=41,
        domain_rand_config_type="disabled",
        mpc=controller,
        episode_length=4,
        verbose=0,
        fixed_command=[0.5, 0.0, 0.0],
        command_ramp_control_steps=2,
        gait="bound_front_first",
        duty_factor=0.5,
        step_frequency_hz=3.0,
        step_height_m=0.03,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )
    expected = build_command_schedule(
        [0.5, 0.0, 0.0], episode_length=4, ramp_control_steps=2
    )

    np.testing.assert_array_equal(trajectory["commands_ctrl"], expected)
    np.testing.assert_array_equal(
        trajectory["commands"], np.repeat(expected, 4, axis=0).T
    )
    np.testing.assert_array_equal(
        trajectory["configured_command_schedule_ctrl"], expected
    )
    np.testing.assert_array_equal(
        np.asarray(controller.inputs)[:, :2], expected[:, :2]
    )
    assert trajectory["gait_name"] == "bound_front_first"
    assert trajectory["gait_duty_factor"] == 0.5
    assert trajectory["gait_step_frequency_hz"] == 3.0
    assert trajectory["gait_step_height_m"] == 0.03
    assert trajectory["domain_rand_config_type"] == "disabled"
    assert trajectory["action_interface_id"] == MPX_BOUND_ACTION_INTERFACE_ID
    assert json.loads(trajectory["generation_settings_json"])["target_command"] == [
        0.5,
        0.0,
        0.0,
    ]
    assert all(np.asarray(value).dtype != object for value in trajectory.values())


def test_shared_controller_reset_reproduces_fixed_schedule_actions():
    controller = RecordingBoundMPC()
    kwargs = dict(
        seed=43,
        domain_rand_config_type="disabled",
        mpc=controller,
        episode_length=4,
        verbose=0,
        fixed_command=[0.5, 0.0, 0.0],
        command_ramp_control_steps=2,
        gait="bound_front_first",
        duty_factor=0.5,
        step_frequency_hz=3.0,
        step_height_m=0.03,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )

    first = generate_trajectory(**kwargs)
    second = generate_trajectory(**kwargs)

    assert controller.reset_count == 2
    np.testing.assert_array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(
        first["planned_contact_schedule_ctrl"],
        second["planned_contact_schedule_ctrl"],
    )


def test_batch_constructs_controller_with_requested_instance_gait(
    tmp_path, monkeypatch
):
    constructed = []
    constructed_configs = []

    class CapturingMPC(RecordingBoundMPC):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs.copy())
            constructed_configs.append(args[0])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "mpc_rl.planner.gen_traj_data_mpx_dr.mpc_wrapper.MPCControllerWrapper",
        CapturingMPC,
    )
    gen_traj_quadruped_dr(
        num_trajectories=1,
        episode_length=2,
        start_seed=3,
        output_dir=tmp_path,
        max_attempts=1,
        verbose=0,
        domain_rand_config_type="disabled",
        fixed_command=[0.5, 0.0, 0.0],
        command_ramp_control_steps=1,
        gait="bound_front_first",
        duty_factor=0.5,
        step_frequency_hz=3.0,
        step_height_m=0.03,
        mpx_qrot_pitch_cost=25_000.0,
        mpx_qomega_pitch_cost=250.0,
        mpx_qp_height_cost=2_500.0,
        mpx_qdp_vertical_cost=4_000.0,
        mpx_qleg_vertical_cost=50_000.0,
        mpx_robot_height_m=0.24,
        mpx_swing_clearance_speed_m_per_s=0.2,
        joint_kp={
            "hip_joint": 20.0,
            "thigh_joint": 30.0,
            "calf_joint": 40.0,
        },
        joint_kd={
            "hip_joint": 1.0,
            "thigh_joint": 2.0,
            "calf_joint": 2.0,
        },
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )

    assert constructed == [
        {
            "use_go2_sysid": True,
            "gait": "bound_front_first",
            "duty_factor": 0.5,
            "step_frequency_hz": 3.0,
            "step_height_m": 0.03,
            "enable_planned_contact_diagnostics": True,
        }
    ]
    assert float(np.asarray(constructed_configs[0].Qrot)[1, 1]) == 25_000.0
    assert float(np.asarray(constructed_configs[0].Qrot)[0, 0]) == 5_000.0
    assert float(np.asarray(constructed_configs[0].Qomega)[1, 1]) == 250.0
    assert float(np.asarray(constructed_configs[0].Qomega)[0, 0]) == 500.0
    assert float(np.asarray(constructed_configs[0].Qp)[2, 2]) == 2_500.0
    assert float(np.asarray(constructed_configs[0].Qp)[1, 1]) == 100_000.0
    assert float(np.asarray(constructed_configs[0].Qdp)[2, 2]) == 4_000.0
    assert float(np.asarray(constructed_configs[0].Qdp)[0, 0]) == 2_000.0
    np.testing.assert_array_equal(
        np.diag(np.asarray(constructed_configs[0].Qleg))[2::3],
        np.full(4, 50_000.0),
    )
    assert float(np.asarray(constructed_configs[0].Qleg)[0, 0]) == 10_000.0
    assert float(constructed_configs[0].robot_height) == 0.24
    assert float(np.asarray(constructed_configs[0].p0)[2]) == pytest.approx(0.24)
    assert float(np.asarray(go2_config.Qrot)[1, 1]) == 50_000.0
    filename = next(tmp_path.glob("*.npz")).name
    assert "bound_front_first" in filename
    assert "vx0p5" in filename
    manifest = json.loads(
        (tmp_path / "generation_manifest.jsonl").read_text(encoding="utf-8")
    )
    assert manifest["result"] == "passed"
    assert manifest["passed"] is True
    assert manifest["failure_reasons"] == []
    assert (
        manifest["generation_settings"]["controller"]["mpx_qrot_pitch_cost"]
        == 25_000.0
    )
    assert (
        manifest["generation_settings"]["controller"][
            "mpx_qomega_pitch_cost"
        ]
        == 250.0
    )
    assert (
        manifest["generation_settings"]["controller"]["mpx_qp_height_cost"]
        == 2_500.0
    )
    assert (
        manifest["generation_settings"]["controller"][
            "mpx_qdp_vertical_cost"
        ]
        == 4_000.0
    )
    assert (
        manifest["generation_settings"]["controller"][
            "mpx_qleg_vertical_cost"
        ]
        == 50_000.0
    )
    assert (
        manifest["generation_settings"]["controller"]["mpx_robot_height_m"]
        == 0.24
    )
    assert (
        manifest["generation_settings"]["controller"][
            "mpx_swing_clearance_speed_m_per_s"
        ]
        == 0.2
    )
    assert manifest["generation_settings"]["controller"]["environment_kp"] == [
        20.0,
        30.0,
        40.0,
    ] * 4
    assert manifest["generation_settings"]["controller"]["environment_kd"] == [
        1.0,
        2.0,
        2.0,
    ] * 4


def test_legacy_generator_defaults_remain_legacy():
    signature = inspect.signature(gen_traj_quadruped_dr)

    assert signature.parameters["fixed_command"].default is None
    assert signature.parameters["command_ramp_control_steps"].default == 0
    assert signature.parameters["gait"].default is None
    assert signature.parameters["mpx_qrot_pitch_cost"].default is None
    assert signature.parameters["mpx_qomega_pitch_cost"].default is None
    assert signature.parameters["mpx_qp_height_cost"].default is None
    assert signature.parameters["mpx_qdp_vertical_cost"].default is None
    assert signature.parameters["mpx_qleg_vertical_cost"].default is None
    assert signature.parameters["mpx_robot_height_m"].default is None
    assert (
        signature.parameters["mpx_swing_clearance_speed_m_per_s"].default
        is None
    )
    assert signature.parameters["acceptance_declaration"].default is None
    assert signature.parameters["domain_rand_config_type"].default == (
        "sysid_dyn20_mjlab"
    )
    assert signature.parameters["action_interface_id"].default == (
        DEFAULT_ACTION_INTERFACE_ID
    )


def test_commissioning_declaration_preserves_per_joint_type_gains():
    declaration = _short_declaration()
    declaration["controller"]["joint_kp_arg"] = {
        "hip_joint": 20.0,
        "thigh_joint": 30.0,
        "calf_joint": 40.0,
    }
    declaration["controller"]["joint_kd_arg"] = {
        "hip_joint": 1.0,
        "thigh_joint": 2.0,
        "calf_joint": 2.0,
    }

    normalized = validate_acceptance_declaration(declaration)

    assert normalized["controller"]["joint_kp_arg"]["thigh_joint"] == 30.0
    assert normalized["controller"]["joint_kd_arg"]["thigh_joint"] == 2.0


def test_measured_classifier_accepts_repeated_symmetric_alternation():
    states = []
    for _ in range(12):
        states.extend([[1, 1, 0, 0]] * 4)
        states.extend([[1, 1, 1, 1]] * 2)
        states.extend([[0, 0, 1, 1]] * 4)
        states.extend([[1, 1, 1, 1]] * 2)
    metrics = measured_bound_metrics(states, contact_debounce_substeps=2)

    assert metrics["front_pair_agreement"] == 1.0
    assert metrics["rear_pair_agreement"] == 1.0
    assert metrics["complete_front_rear_cycles"] >= 10
    assert metrics["front_only_fraction"] >= 0.05
    assert metrics["rear_only_fraction"] >= 0.05
    assert metrics["paired_interval_alternation_fraction"] == 1.0
    assert metrics["diagonal_lateral_only_fraction"] == 0.0


def test_acceptance_declaration_rejects_dr_and_wrong_interface():
    declaration = _short_declaration()
    declaration["domain_rand_config_type"] = "sysid_dyn20_mjlab"
    with pytest.raises(ValueError, match="DR disabled"):
        validate_acceptance_declaration(declaration)

    declaration = _short_declaration()
    declaration["action_interface_id"] = DEFAULT_ACTION_INTERFACE_ID
    with pytest.raises(ValueError, match="accepted MPX action interface"):
        validate_acceptance_declaration(declaration)


def test_validated_stages_enforce_disjoint_seed_namespaces(tmp_path):
    with pytest.raises(ValueError, match="commissioning start_seed"):
        gen_traj_quadruped_dr(
            num_trajectories=1,
            episode_length=4,
            start_seed=100_000,
            output_dir=tmp_path,
            max_attempts=1,
            verbose=0,
            acceptance_declaration=_short_declaration(),
            stage="commissioning",
        )

def test_full_saved_action_replay_covers_all_steps_and_contacts():
    declaration = _short_declaration()
    trajectory = generate_trajectory(
        seed=47,
        domain_rand_config_type="disabled",
        mpc=RecordingBoundMPC(),
        episode_length=4,
        verbose=0,
        fixed_command=[0.5, 0.0, 0.0],
        command_ramp_control_steps=2,
        gait="bound_front_first",
        duty_factor=0.5,
        step_frequency_hz=3.0,
        step_height_m=0.03,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
    )

    replay = validate_full_saved_action_replay(trajectory, declaration)

    assert replay["passed"] is True
    assert replay["replayed_control_steps"] == 4
    assert replay["foot_contact_mismatch_count"] == 0
    assert replay["non_foot_contact_mismatch_count"] == 0
    assert replay["replay_action_clipping_fraction"] == 0.0


def test_schema_integrity_and_operational_predicate_accept_complete_evidence():
    episode_length = 12
    declaration = make_commissioning_acceptance_declaration(
        target_command=[0.5, 0.0, 0.0],
        episode_length=episode_length,
        ramp_control_steps=0,
        gait_name="bound_front_first",
        gait_duty_factor=0.5,
        gait_step_frequency_hz=3.0,
        gait_step_height_m=0.03,
        acceptance_overrides={"min_complete_front_rear_cycles": 1},
    )
    trajectory = generate_trajectory(
        seed=53,
        domain_rand_config_type="disabled",
        mpc=RecordingBoundMPC(),
        episode_length=episode_length,
        verbose=0,
        fixed_command=[0.5, 0.0, 0.0],
        command_ramp_control_steps=0,
        gait="bound_front_first",
        duty_factor=0.5,
        step_frequency_hz=3.0,
        step_height_m=0.03,
        action_interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
        provenance={
            "generator_root_commit": "0" * 40,
            "generator_mpx_commit": "1" * 40,
            "generator_primal_dual_ilqr_commit": "2" * 40,
        },
    )
    contact_cycle = (
        [[1, 1, 0, 0]] * 4
        + [[0, 0, 1, 1]] * 4
        + [[1, 1, 0, 0]] * 4
    )
    trajectory["foot_contacts_substeps"] = np.asarray(
        contact_cycle * 4, dtype=bool
    ).reshape(episode_length, 4, 4)

    result = validate_bounding_trajectory_data(
        trajectory, declaration, replay_report={"passed": True}
    )

    assert result["passed"] is True
    assert all(result["threshold_checks"].values())


def test_bounding_filename_encodes_gait_and_fixed_speed():
    filename = _bounding_filename(
        {
            "gait_name": "bound_front_first",
            "target_command": np.array([0.5, 0.0, 0.0]),
        },
        seed=7,
        episode_length=1000,
    )

    assert filename == (
        "quadruped_bound_bound_front_first_vx0p5_vy0_wz0_"
        "seed_000007_ep_1000.npz"
    )
