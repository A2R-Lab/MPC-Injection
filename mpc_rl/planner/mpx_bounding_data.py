"""Validation and reporting primitives for fixed-speed MPX bounding data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from mpc_rl.envs.action_interfaces import MPX_BOUND_ACTION_INTERFACE_ID


BOUNDING_DATA_SCHEMA_VERSION = 1
BOUNDING_EXPERIMENT_ID = "nominal_vx0p5_bound_v1"
BOUNDING_GAITS = ("bound_front_first", "bound_hind_first")
SIMULATION_SUBSTEPS_PER_CONTROL_STEP = 4
BOUNDING_STAGE_SEED_RANGES = {
    "commissioning": (0, 99_999),
    "pilot": (100_000, 199_999),
    "production": (200_000, 2_147_483_647),
}

DEFAULT_BOUNDING_ACCEPTANCE = {
    "contact_debounce_substeps": 2,
    "min_front_pair_agreement": 0.8,
    "min_rear_pair_agreement": 0.8,
    "min_complete_front_rear_cycles": 10,
    "min_front_only_fraction": 0.05,
    "min_rear_only_fraction": 0.05,
    "min_paired_interval_alternation_fraction": 0.8,
    "max_diagonal_lateral_only_fraction": 0.1,
    "max_action_clipped_element_fraction": 0.01,
    "max_action_clip_magnitude": 0.1,
    "max_abs_pitch_rad": 0.5,
    "max_abs_roll_rad": 0.5,
    "min_base_height_m": 0.1,
}


def _json_scalar(data: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"metadata {key!r} must be scalar, got shape {value.shape}")
    return value.item()


def _finite_vector(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape {shape}")
    return array


def build_command_schedule(
    target_command: Any,
    *,
    episode_length: int,
    ramp_control_steps: int,
) -> np.ndarray:
    """Build the deterministic ramp/hold schedule applied at control rate."""
    target = _finite_vector(target_command, name="target_command", shape=(3,))
    if isinstance(episode_length, bool) or int(episode_length) != episode_length:
        raise ValueError("episode_length must be an integer")
    if isinstance(ramp_control_steps, bool) or int(ramp_control_steps) != ramp_control_steps:
        raise ValueError("ramp_control_steps must be an integer")
    episode_length = int(episode_length)
    ramp_control_steps = int(ramp_control_steps)
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if not 0 <= ramp_control_steps <= episode_length:
        raise ValueError(
            "ramp_control_steps must satisfy 0 <= value <= episode_length"
        )

    schedule = np.repeat(target[None, :], episode_length, axis=0)
    if ramp_control_steps:
        fractions = (
            np.arange(1, ramp_control_steps + 1, dtype=np.float64)
            / float(ramp_control_steps)
        )
        schedule[:ramp_control_steps] = fractions[:, None] * target[None, :]
    return schedule


def command_schedule_id(
    target_command: Any, *, episode_length: int, ramp_control_steps: int
) -> str:
    target = _finite_vector(target_command, name="target_command", shape=(3,))
    descriptor = {
        "control_frequency_hz": 50,
        "episode_length": int(episode_length),
        "ramp_control_steps": int(ramp_control_steps),
        "target_command": target.tolist(),
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"linear_ramp_hold_v1_{digest}"


def make_commissioning_acceptance_declaration(
    *,
    target_command: Any,
    episode_length: int,
    ramp_control_steps: int,
    gait_name: str,
    gait_duty_factor: float,
    gait_step_frequency_hz: float,
    gait_step_height_m: float,
    joint_kp: float | None = None,
    joint_kd: float | None = None,
    max_pitch_rad: float = 0.5,
    max_roll_rad: float = 0.5,
    min_base_height_m: float = 0.1,
    mpx_qrot_pitch_cost: float | None = None,
    mpx_qomega_pitch_cost: float | None = None,
    mpx_robot_height_m: float | None = None,
    acceptance_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an explicitly unfrozen declaration for commissioning attempts."""
    target = _finite_vector(target_command, name="target_command", shape=(3,))
    if gait_name not in BOUNDING_GAITS:
        raise ValueError(f"gait_name must be one of {BOUNDING_GAITS}")
    acceptance = dict(DEFAULT_BOUNDING_ACCEPTANCE)
    if acceptance_overrides:
        unknown = sorted(set(acceptance_overrides) - set(acceptance))
        if unknown:
            raise ValueError(f"unknown acceptance overrides: {unknown}")
        acceptance.update(acceptance_overrides)
    return {
        "schema_version": BOUNDING_DATA_SCHEMA_VERSION,
        "declaration_id": "commissioning_candidate_unfrozen",
        "frozen": False,
        "experiment_id": BOUNDING_EXPERIMENT_ID,
        "episode_length": int(episode_length),
        "simulation_substeps_per_control_step": (
            SIMULATION_SUBSTEPS_PER_CONTROL_STEP
        ),
        "target_command": target.tolist(),
        "ramp_control_steps": int(ramp_control_steps),
        "action_interface_id": MPX_BOUND_ACTION_INTERFACE_ID,
        "domain_rand_config_type": "disabled",
        "use_go2_sysid": True,
        "gait": {
            "name": gait_name,
            "duty_factor": float(gait_duty_factor),
            "step_frequency_hz": float(gait_step_frequency_hz),
            "step_height_m": float(gait_step_height_m),
        },
        "controller": {
            "joint_kp_arg": None if joint_kp is None else float(joint_kp),
            "joint_kd_arg": None if joint_kd is None else float(joint_kd),
            "max_pitch_rad": float(max_pitch_rad),
            "max_roll_rad": float(max_roll_rad),
            "min_base_height_m": float(min_base_height_m),
            "mpx_qrot_pitch_cost": (
                None
                if mpx_qrot_pitch_cost is None
                else float(mpx_qrot_pitch_cost)
            ),
            "mpx_qomega_pitch_cost": (
                None
                if mpx_qomega_pitch_cost is None
                else float(mpx_qomega_pitch_cost)
            ),
            "realized_environment_kp": None,
            "realized_environment_kd": None,
            "mpx_weight_matrix_sha256": None,
            "mpx_robot_height_m": (
                None
                if mpx_robot_height_m is None
                else float(mpx_robot_height_m)
            ),
        },
        "acceptance": acceptance,
    }


def validate_acceptance_declaration(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a commissioning or frozen acceptance declaration."""
    required = {
        "schema_version",
        "declaration_id",
        "frozen",
        "experiment_id",
        "episode_length",
        "simulation_substeps_per_control_step",
        "target_command",
        "ramp_control_steps",
        "action_interface_id",
        "domain_rand_config_type",
        "use_go2_sysid",
        "gait",
        "controller",
        "acceptance",
    }
    missing = sorted(required - set(declaration))
    if missing:
        raise ValueError(f"acceptance declaration is missing {missing}")
    normalized = json.loads(
        json.dumps(declaration, sort_keys=True, allow_nan=False)
    )
    if int(normalized["schema_version"]) != BOUNDING_DATA_SCHEMA_VERSION:
        raise ValueError("unsupported bounding acceptance declaration schema")
    if normalized["experiment_id"] != BOUNDING_EXPERIMENT_ID:
        raise ValueError("unexpected bounding experiment_id")
    if not isinstance(normalized["frozen"], bool):
        raise ValueError("acceptance declaration frozen field must be boolean")
    if normalized["action_interface_id"] != MPX_BOUND_ACTION_INTERFACE_ID:
        raise ValueError("bounding data requires the accepted MPX action interface")
    if normalized["domain_rand_config_type"] != "disabled":
        raise ValueError("bounding data requires nominal dynamics with DR disabled")
    if normalized["use_go2_sysid"] is not True:
        raise ValueError("bounding data requires the repository Go2 sysID model")
    if int(normalized["simulation_substeps_per_control_step"]) != 4:
        raise ValueError("bounding data requires four 200 Hz substeps per control step")
    schedule = build_command_schedule(
        normalized["target_command"],
        episode_length=int(normalized["episode_length"]),
        ramp_control_steps=int(normalized["ramp_control_steps"]),
    )
    del schedule
    if normalized["frozen"]:
        if int(normalized["episode_length"]) != 1000:
            raise ValueError("frozen production trajectories must contain 1000 steps")
        if int(normalized["ramp_control_steps"]) != 50:
            raise ValueError("frozen production command ramp must contain 50 steps")
        if not np.array_equal(
            np.asarray(normalized["target_command"], dtype=np.float64),
            np.asarray([0.5, 0.0, 0.0], dtype=np.float64),
        ):
            raise ValueError("frozen production target must be [0.5, 0.0, 0.0]")

    gait = normalized["gait"]
    if set(gait) != {
        "name",
        "duty_factor",
        "step_frequency_hz",
        "step_height_m",
    }:
        raise ValueError("acceptance declaration gait fields are incomplete")
    if gait["name"] not in BOUNDING_GAITS:
        raise ValueError(f"gait name must be one of {BOUNDING_GAITS}")
    duty = float(gait["duty_factor"])
    frequency = float(gait["step_frequency_hz"])
    height = float(gait["step_height_m"])
    if not np.isfinite(duty) or not 0.0 < duty <= 1.0:
        raise ValueError("gait duty factor must satisfy 0 < value <= 1")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("gait step frequency must be positive")
    if not np.isfinite(height) or height < 0.0:
        raise ValueError("gait step height must be non-negative")

    controller = normalized["controller"]
    expected_controller_keys = {
        "joint_kp_arg",
        "joint_kd_arg",
        "max_pitch_rad",
        "max_roll_rad",
        "min_base_height_m",
        "mpx_qrot_pitch_cost",
        "mpx_qomega_pitch_cost",
        "realized_environment_kp",
        "realized_environment_kd",
        "mpx_weight_matrix_sha256",
        "mpx_robot_height_m",
    }
    if set(controller) != expected_controller_keys:
        raise ValueError("acceptance declaration controller fields are incomplete")
    for key in ("joint_kp_arg", "joint_kd_arg"):
        value = controller[key]
        if value is not None and (not np.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"controller {key} must be null or non-negative")
    pitch_cost = controller["mpx_qrot_pitch_cost"]
    if pitch_cost is not None and (
        not np.isfinite(float(pitch_cost)) or float(pitch_cost) < 0.0
    ):
        raise ValueError(
            "controller mpx_qrot_pitch_cost must be null or non-negative"
        )
    pitch_rate_cost = controller["mpx_qomega_pitch_cost"]
    if pitch_rate_cost is not None and (
        not np.isfinite(float(pitch_rate_cost))
        or float(pitch_rate_cost) < 0.0
    ):
        raise ValueError(
            "controller mpx_qomega_pitch_cost must be null or non-negative"
        )
    robot_height = controller["mpx_robot_height_m"]
    if robot_height is not None and (
        not np.isfinite(float(robot_height)) or float(robot_height) <= 0.0
    ):
        raise ValueError(
            "controller mpx_robot_height_m must be null or positive"
        )
    for key in ("max_pitch_rad", "max_roll_rad", "min_base_height_m"):
        value = float(controller[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"controller {key} must be non-negative")
    if normalized["frozen"]:
        for key in ("realized_environment_kp", "realized_environment_kd"):
            values = np.asarray(controller[key], dtype=np.float64)
            if values.shape != (12,) or not np.all(np.isfinite(values)):
                raise ValueError(f"frozen controller {key} must contain 12 finite values")
        weight_hash = controller["mpx_weight_matrix_sha256"]
        if (
            not isinstance(weight_hash, str)
            or len(weight_hash) != 64
            or any(character not in "0123456789abcdef" for character in weight_hash)
        ):
            raise ValueError("frozen controller MPX weight hash must be SHA-256")
        robot_height = controller["mpx_robot_height_m"]
        if robot_height is None or not np.isfinite(float(robot_height)):
            raise ValueError("frozen controller robot height must be finite")

    acceptance = normalized["acceptance"]
    missing_acceptance = sorted(set(DEFAULT_BOUNDING_ACCEPTANCE) - set(acceptance))
    if missing_acceptance:
        raise ValueError(
            f"acceptance declaration is missing thresholds {missing_acceptance}"
        )
    if int(acceptance["contact_debounce_substeps"]) < 1:
        raise ValueError("contact debounce must be at least one substep")
    for key in (
        "min_front_pair_agreement",
        "min_rear_pair_agreement",
        "min_front_only_fraction",
        "min_rear_only_fraction",
        "min_paired_interval_alternation_fraction",
        "max_diagonal_lateral_only_fraction",
        "max_action_clipped_element_fraction",
    ):
        value = float(acceptance[key])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"acceptance threshold {key} must be in [0, 1]")
    for key in (
        "max_action_clip_magnitude",
        "max_abs_pitch_rad",
        "max_abs_roll_rad",
        "min_base_height_m",
    ):
        value = float(acceptance[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"acceptance threshold {key} must be non-negative")
    if int(acceptance["min_complete_front_rear_cycles"]) < 1:
        raise ValueError("at least one complete front/rear cycle must be required")
    return normalized


def freeze_acceptance_declaration(
    commissioning_declaration: Mapping[str, Any],
    passing_trajectory: Mapping[str, Any],
    *,
    declaration_id: str,
    visual_inspection_report: str,
) -> dict[str, Any]:
    """Freeze a passing controller and the plan's 1,000-step production schedule."""
    declaration = validate_acceptance_declaration(commissioning_declaration)
    if declaration["frozen"]:
        raise ValueError("commissioning declaration is already frozen")
    if not declaration_id or declaration_id == "commissioning_candidate_unfrozen":
        raise ValueError("a new versioned declaration_id is required")
    if not visual_inspection_report:
        raise ValueError("a retained visual inspection report is required")

    declaration["declaration_id"] = declaration_id
    declaration["frozen"] = True
    declaration["episode_length"] = 1000
    declaration["ramp_control_steps"] = 50
    declaration["target_command"] = [0.5, 0.0, 0.0]
    controller = declaration["controller"]
    controller["realized_environment_kp"] = np.asarray(
        passing_trajectory["environment_kp"], dtype=np.float64
    ).tolist()
    controller["realized_environment_kd"] = np.asarray(
        passing_trajectory["environment_kd"], dtype=np.float64
    ).tolist()
    controller["mpx_weight_matrix_sha256"] = str(
        _json_scalar(passing_trajectory, "controller_weight_matrix_sha256")
    )
    controller["mpx_robot_height_m"] = float(
        _json_scalar(passing_trajectory, "controller_robot_height_m")
    )
    declaration["frozen_from"] = {
        "commissioning_seed": int(_json_scalar(passing_trajectory, "seed")),
        "generator_root_commit": str(
            _json_scalar(passing_trajectory, "generator_root_commit")
        ),
        "generator_mpx_commit": str(
            _json_scalar(passing_trajectory, "generator_mpx_commit")
        ),
        "generator_primal_dual_ilqr_commit": str(
            _json_scalar(
                passing_trajectory, "generator_primal_dual_ilqr_commit"
            )
        ),
        "visual_inspection_report": visual_inspection_report,
    }
    declaration["non_rejecting_metrics"] = [
        "achieved_velocity_tracking",
        "lateral_velocity_and_displacement",
        "yaw_rate_error_and_drift",
        "torque_saturation",
        "measured_flight",
    ]
    return validate_acceptance_declaration(declaration)


def _debounce_contact_states(contacts: np.ndarray, persistence: int) -> np.ndarray:
    contacts = np.asarray(contacts, dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != 4 or contacts.shape[0] == 0:
        raise ValueError("contacts must have non-empty shape (substeps, 4)")
    if persistence < 1:
        raise ValueError("persistence must be at least one")
    if persistence == 1:
        return contacts.copy()

    debounced = np.empty_like(contacts)
    accepted = contacts[0].copy()
    debounced[0] = accepted
    candidate = accepted.copy()
    candidate_start = 0
    candidate_count = 0
    for index in range(1, contacts.shape[0]):
        state = contacts[index]
        if np.array_equal(state, accepted):
            candidate = accepted.copy()
            candidate_count = 0
            candidate_start = index
            debounced[index] = accepted
            continue
        if candidate_count == 0 or not np.array_equal(state, candidate):
            candidate = state.copy()
            candidate_start = index
            candidate_count = 1
        else:
            candidate_count += 1
        debounced[index] = accepted
        if candidate_count >= persistence:
            accepted = candidate.copy()
            debounced[candidate_start : index + 1] = accepted
            candidate_count = 0
    return debounced


def _state_segments(states: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    starts = [0]
    for index in range(1, states.shape[0]):
        if not np.array_equal(states[index], states[index - 1]):
            starts.append(index)
    starts.append(states.shape[0])
    return [
        (starts[index], starts[index + 1], states[starts[index]].copy())
        for index in range(len(starts) - 1)
    ]


def measured_bound_metrics(
    contacts: Any, *, contact_debounce_substeps: int
) -> dict[str, Any]:
    """Compute the frozen measured-contact classifier inputs and decisions."""
    raw = np.asarray(contacts, dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != 4 or raw.shape[0] == 0:
        raise ValueError("measured contacts must have non-empty shape (substeps, 4)")
    debounced = _debounce_contact_states(raw, int(contact_debounce_substeps))

    front_only = np.all(raw == np.array([1, 1, 0, 0], dtype=bool), axis=1)
    rear_only = np.all(raw == np.array([0, 0, 1, 1], dtype=bool), axis=1)
    all_four = np.all(raw, axis=1)
    flight = ~np.any(raw, axis=1)
    bad_two_foot_states = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    )
    diagonal_or_lateral = np.any(
        np.all(raw[:, None, :] == bad_two_foot_states[None, :, :], axis=2),
        axis=1,
    )

    paired_intervals: list[dict[str, Any]] = []
    for start, end, state in _state_segments(debounced):
        if np.array_equal(state, np.array([1, 1, 0, 0], dtype=bool)):
            support = "front"
        elif np.array_equal(state, np.array([0, 0, 1, 1], dtype=bool)):
            support = "rear"
        else:
            continue
        paired_intervals.append(
            {"support": support, "start_substep": start, "end_substep": end}
        )

    alternating = [
        paired_intervals[index]["support"]
        != paired_intervals[index - 1]["support"]
        for index in range(1, len(paired_intervals))
    ]
    alternating_count = int(np.count_nonzero(alternating))
    alternation_fraction = (
        float(np.mean(alternating)) if alternating else 0.0
    )
    complete_cycles = alternating_count // 2

    cycle_windows: list[list[int]] = []
    half_cycle_count = 0
    cycle_start = None
    for index in range(1, len(paired_intervals)):
        if paired_intervals[index]["support"] == paired_intervals[index - 1]["support"]:
            half_cycle_count = 0
            cycle_start = None
            continue
        if half_cycle_count == 0:
            cycle_start = paired_intervals[index - 1]["start_substep"]
        half_cycle_count += 1
        if half_cycle_count == 2 and cycle_start is not None:
            cycle_windows.append(
                [cycle_start, paired_intervals[index]["start_substep"]]
            )
            half_cycle_count = 0
            cycle_start = paired_intervals[index]["start_substep"]

    return {
        "target_speed_substeps": int(raw.shape[0]),
        "contact_debounce_substeps": int(contact_debounce_substeps),
        "front_pair_agreement": float(np.mean(raw[:, 0] == raw[:, 1])),
        "rear_pair_agreement": float(np.mean(raw[:, 2] == raw[:, 3])),
        "front_only_fraction": float(np.mean(front_only)),
        "rear_only_fraction": float(np.mean(rear_only)),
        "all_four_fraction": float(np.mean(all_four)),
        "flight_fraction": float(np.mean(flight)),
        "diagonal_lateral_only_fraction": float(np.mean(diagonal_or_lateral)),
        "paired_support_interval_count": len(paired_intervals),
        "paired_interval_alternation_count": alternating_count,
        "paired_interval_alternation_fraction": alternation_fraction,
        "complete_front_rear_cycles": complete_cycles,
        "paired_support_intervals": paired_intervals,
        "cycle_windows_substeps": cycle_windows,
    }


def _body_velocity_series(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(np.roll(qpos[:, 3:7], -1, axis=1))
    return rotations.inv().apply(qvel[:, :3])


def _distribution_metrics(values: np.ndarray, *, target: float = 0.0) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    errors = values - float(target)
    return {
        "mean": float(np.mean(values)),
        "signed_bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def trajectory_quality_metrics(
    data: Mapping[str, Any], declaration: Mapping[str, Any]
) -> dict[str, Any]:
    """Report contact, tracking, posture, clipping, saturation, and solver quality."""
    declaration = validate_acceptance_declaration(declaration)
    total_steps = int(declaration["episode_length"])
    ramp_steps = int(declaration["ramp_control_steps"])
    hold_substep_start = ramp_steps * SIMULATION_SUBSTEPS_PER_CONTROL_STEP
    completed_steps = int(_json_scalar(data, "completed_control_steps"))
    completed_substeps = completed_steps * SIMULATION_SUBSTEPS_PER_CONTROL_STEP
    contacts = np.asarray(data["foot_contacts_substeps"], dtype=bool).reshape(-1, 4)
    hold_contacts = contacts[hold_substep_start:completed_substeps]
    if hold_contacts.shape[0] == 0:
        raise ValueError("trajectory contains no completed target-speed hold substeps")

    acceptance = declaration["acceptance"]
    contact = measured_bound_metrics(
        hold_contacts,
        contact_debounce_substeps=int(acceptance["contact_debounce_substeps"]),
    )

    qpos_sim = np.asarray(data["qpos"], dtype=np.float64).T[1 : completed_substeps + 1]
    qvel_sim = np.asarray(data["qvel"], dtype=np.float64).T[1 : completed_substeps + 1]
    hold_qpos = qpos_sim[hold_substep_start:]
    hold_qvel = qvel_sim[hold_substep_start:]
    body_velocity = _body_velocity_series(hold_qpos, hold_qvel)
    target_command = np.asarray(declaration["target_command"], dtype=np.float64)

    cycle_vx = []
    for start, end in contact["cycle_windows_substeps"]:
        if end > start:
            cycle_vx.append(float(np.mean(body_velocity[start:end, 0])))

    yaw = Rotation.from_quat(np.roll(hold_qpos[:, 3:7], -1, axis=1)).as_euler(
        "xyz"
    )[:, 2]
    yaw = np.unwrap(yaw)
    initial_rotation = Rotation.from_quat(np.roll(hold_qpos[0, 3:7], -1))
    displacement_initial_body = initial_rotation.inv().apply(
        hold_qpos[-1, :3] - hold_qpos[0, :3]
    )
    yaw_rate_error = hold_qvel[:, 5] - target_command[2]

    all_qpos = np.asarray(data["qpos"], dtype=np.float64).T[: completed_substeps + 1]
    euler = Rotation.from_quat(np.roll(all_qpos[:, 3:7], -1, axis=1)).as_euler(
        "xyz"
    )
    clipping_mask = np.asarray(data["action_clipping_mask"], dtype=bool)
    clipping_magnitude = np.asarray(
        data["action_clipping_magnitude"], dtype=np.float64
    )
    raw_actions = np.asarray(data["raw_actions"], dtype=np.float64)
    saturation_mask = np.asarray(data["torque_saturation_mask"], dtype=bool)
    saturation_magnitude = np.asarray(
        data["torque_saturation_magnitude"], dtype=np.float64
    )
    solver_finite = np.asarray(data["solver_finite_ctrl"], dtype=bool)
    joint_names = np.asarray(data["joint_names"]).astype(str)

    clipping_by_joint = []
    saturation_by_joint = []
    for joint_index, joint_name in enumerate(joint_names.tolist()):
        joint_clip = clipping_mask[:, joint_index]
        joint_clip_magnitude = clipping_magnitude[:, joint_index]
        joint_raw = raw_actions[:, joint_index]
        joint_saturation = saturation_mask[..., joint_index]
        joint_saturation_magnitude = saturation_magnitude[..., joint_index]
        clipping_by_joint.append(
            {
                "joint_name": joint_name,
                "element_fraction": float(np.mean(joint_clip)),
                "max_magnitude": float(np.max(joint_clip_magnitude)),
                "max_abs_raw_action": float(np.max(np.abs(joint_raw))),
            }
        )
        saturation_by_joint.append(
            {
                "joint_name": joint_name,
                "element_fraction": float(np.mean(joint_saturation)),
                "max_requested_torque_overshoot_nm": float(
                    np.max(joint_saturation_magnitude)
                ),
            }
        )

    return {
        "requested_control_steps": total_steps,
        "completed_control_steps": completed_steps,
        "contact": contact,
        "tracking": {
            "forward_velocity_m_per_s": _distribution_metrics(
                body_velocity[:, 0], target=float(target_command[0])
            ),
            "lateral_velocity_m_per_s": _distribution_metrics(
                body_velocity[:, 1], target=float(target_command[1])
            ),
            "lateral_displacement_m": float(displacement_initial_body[1]),
            "yaw_rate_error_rad_per_s": _distribution_metrics(yaw_rate_error),
            "accumulated_yaw_drift_rad": float(yaw[-1] - yaw[0]),
            "per_gait_cycle_mean_forward_velocity_m_per_s": cycle_vx,
        },
        "posture": {
            "min_base_height_m": float(np.min(all_qpos[:, 2])),
            "max_abs_roll_rad": float(np.max(np.abs(euler[:, 0]))),
            "max_abs_pitch_rad": float(np.max(np.abs(euler[:, 1]))),
        },
        "action_clipping": {
            "element_count": int(np.count_nonzero(clipping_mask)),
            "total_elements": int(clipping_mask.size),
            "element_fraction": float(np.mean(clipping_mask)),
            "max_magnitude": float(np.max(clipping_magnitude)),
            "max_abs_raw_action": float(np.max(np.abs(raw_actions))),
            "by_joint": clipping_by_joint,
        },
        "torque_saturation": {
            "element_count": int(np.count_nonzero(saturation_mask)),
            "total_elements": int(saturation_mask.size),
            "element_fraction": float(np.mean(saturation_mask)),
            "max_requested_torque_overshoot_nm": float(
                np.max(saturation_magnitude)
            ),
            "by_joint": saturation_by_joint,
        },
        "solver": {
            "finite_control_steps": int(np.count_nonzero(solver_finite)),
            "all_finite": bool(solver_finite.size == completed_steps and np.all(solver_finite)),
        },
    }


def _shape_failures(
    data: Mapping[str, Any], *, episode_length: int, decimation: int
) -> list[str]:
    control_shapes = {
        "policy_obs": (episode_length, 45),
        "next_policy_obs": (episode_length, 45),
        "privileged_obs": (episode_length, 3),
        "next_privileged_obs": (episode_length, 3),
        "raw_actions": (episode_length, 12),
        "actions": (episode_length, 12),
        "action_clipping_mask": (episode_length, 12),
        "action_clipping_magnitude": (episode_length, 12),
        "rewards": (episode_length,),
        "terminated_ctrl": (episode_length,),
        "truncated_ctrl": (episode_length,),
        "commands_ctrl": (episode_length, 3),
        "measured_contacts_ctrl": (episode_length, 4),
        "next_qpos_ctrl": (episode_length, 19),
        "next_qvel_ctrl": (episode_length, 18),
        "requested_torques_ctrl": (episode_length, decimation, 12),
        "applied_torques_ctrl": (episode_length, decimation, 12),
        "torque_saturation_mask": (episode_length, decimation, 12),
        "torque_saturation_magnitude": (episode_length, decimation, 12),
        "foot_contacts_substeps": (episode_length, decimation, 4),
        "non_foot_ground_contact_substeps": (episode_length, decimation),
        "push_delta_qvel": (episode_length, 6),
        "solver_finite_ctrl": (episode_length,),
        "configured_command_schedule_ctrl": (episode_length, 3),
    }
    sim_steps = episode_length * decimation
    simulation_shapes = {
        "qpos": (19, sim_steps + 1),
        "qvel": (18, sim_steps + 1),
        "tau_applied": (12, sim_steps),
        "tau_mpx": (12, sim_steps),
        "q_des": (12, sim_steps),
        "time": (sim_steps + 1,),
        "commands": (3, sim_steps),
        "fixed_command": (3,),
        "target_command": (3,),
        "gait_initial_phase": (4,),
        "contact_order": (4,),
        "environment_kp": (12,),
        "environment_kd": (12,),
        "environment_torque_limits": (12, 2),
    }
    failures = []
    for key, expected in {**control_shapes, **simulation_shapes}.items():
        if key not in data:
            failures.append(f"missing_required_array:{key}")
            continue
        actual = np.asarray(data[key]).shape
        if actual != expected:
            failures.append(f"shape_mismatch:{key}:{actual}!={expected}")
    planned = np.asarray(data.get("planned_contact_schedule_ctrl", []))
    if planned.ndim != 3 or planned.shape[0] != episode_length or planned.shape[2] != 4:
        failures.append(
            "shape_mismatch:planned_contact_schedule_ctrl:"
            f"{planned.shape}!=(episode_length,horizon,4)"
        )
    return failures


def validate_bounding_trajectory_data(
    data: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    replay_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one in-memory trajectory against the operational predicate."""
    declaration = validate_acceptance_declaration(declaration)
    episode_length = int(declaration["episode_length"])
    decimation = int(declaration["simulation_substeps_per_control_step"])
    failures = _shape_failures(
        data, episode_length=episode_length, decimation=decimation
    )

    scalar_requirements = (
        "schema_version",
        "action_interface_id",
        "action_conversion_mode",
        "domain_rand_config_type",
        "go2_sysid_enabled",
        "episode_length",
        "completed_control_steps",
        "completed_sim_steps",
        "fell",
        "failure_reason",
        "fixed_command_enabled",
        "fixed_command",
        "target_command",
        "command_ramp_control_steps",
        "command_schedule_id",
        "gait_name",
        "gait_duty_factor",
        "gait_step_frequency_hz",
        "gait_step_height_m",
        "generation_settings_json",
        "controller_weight_matrix",
        "controller_weight_matrix_sha256",
        "controller_robot_height_m",
        "environment_max_pitch_rad",
        "environment_max_roll_rad",
        "environment_min_base_height_m",
        "generator_root_commit",
        "generator_mpx_commit",
        "generator_primal_dual_ilqr_commit",
    )
    for key in scalar_requirements:
        if key not in data:
            failures.append(f"missing_required_metadata:{key}")

    if failures:
        return {
            "passed": False,
            "failure_reasons": failures,
            "metrics": None,
            "replay": replay_report,
        }

    expected_schedule = build_command_schedule(
        declaration["target_command"],
        episode_length=episode_length,
        ramp_control_steps=int(declaration["ramp_control_steps"]),
    )
    if int(_json_scalar(data, "schema_version")) < 2:
        failures.append("unsupported_direct_transition_schema")
    if _json_scalar(data, "action_interface_id") != declaration["action_interface_id"]:
        failures.append("action_interface_mismatch")
    if _json_scalar(data, "domain_rand_config_type") != "disabled":
        failures.append("domain_randomization_not_disabled")
    if bool(_json_scalar(data, "dr_enabled")):
        failures.append("domain_randomization_bundle_enabled")
    if not bool(_json_scalar(data, "go2_sysid_enabled")):
        failures.append("go2_sysid_disabled")
    if int(_json_scalar(data, "episode_length")) != episode_length:
        failures.append("episode_length_mismatch")
    if int(_json_scalar(data, "completed_control_steps")) != episode_length:
        failures.append("incomplete_control_horizon")
    if int(_json_scalar(data, "completed_sim_steps")) != episode_length * decimation:
        failures.append("incomplete_simulation_horizon")
    if bool(_json_scalar(data, "fell")):
        failures.append("fall_or_safety_reset")
    if str(_json_scalar(data, "failure_reason")):
        failures.append("generator_failure_reason_present")
    if np.any(np.asarray(data["terminated_ctrl"], dtype=bool)):
        failures.append("terminated")
    if np.any(np.asarray(data["truncated_ctrl"], dtype=bool)):
        failures.append("truncated")
    if np.any(np.asarray(data["non_foot_ground_contact_substeps"], dtype=bool)):
        failures.append("non_foot_ground_contact")
    if not np.array_equal(np.asarray(data["commands_ctrl"]), expected_schedule):
        failures.append("applied_control_command_schedule_mismatch")
    if not np.array_equal(
        np.asarray(data["configured_command_schedule_ctrl"]), expected_schedule
    ):
        failures.append("configured_command_schedule_mismatch")
    expected_sim_schedule = np.repeat(expected_schedule, decimation, axis=0).T
    if not np.array_equal(np.asarray(data["commands"]), expected_sim_schedule):
        failures.append("applied_substep_command_schedule_mismatch")

    gait = declaration["gait"]
    gait_checks = {
        "gait_name": gait["name"],
        "gait_duty_factor": float(gait["duty_factor"]),
        "gait_step_frequency_hz": float(gait["step_frequency_hz"]),
        "gait_step_height_m": float(gait["step_height_m"]),
    }
    for key, expected in gait_checks.items():
        if _json_scalar(data, key) != expected:
            failures.append(f"{key}_mismatch")

    try:
        recorded_settings = json.loads(
            str(_json_scalar(data, "generation_settings_json"))
        )
    except (TypeError, json.JSONDecodeError):
        failures.append("generation_settings_json_invalid")
    else:
        if recorded_settings.get("command_schedule_id") != _json_scalar(
            data, "command_schedule_id"
        ):
            failures.append("generation_settings_command_schedule_mismatch")
        if recorded_settings.get("target_command") != list(
            declaration["target_command"]
        ):
            failures.append("generation_settings_target_command_mismatch")

    weights = np.asarray(data["controller_weight_matrix"], dtype=np.float64)
    recorded_weight_hash = str(
        _json_scalar(data, "controller_weight_matrix_sha256")
    )
    if hashlib.sha256(weights.tobytes(order="C")).hexdigest() != recorded_weight_hash:
        failures.append("controller_weight_matrix_checksum_mismatch")
    controller = declaration["controller"]
    scalar_controller_checks = {
        "environment_max_pitch_rad": float(controller["max_pitch_rad"]),
        "environment_max_roll_rad": float(controller["max_roll_rad"]),
        "environment_min_base_height_m": float(controller["min_base_height_m"]),
    }
    for key, expected in scalar_controller_checks.items():
        if float(_json_scalar(data, key)) != expected:
            failures.append(f"declared_controller_mismatch:{key}")
    if declaration["frozen"]:
        if not np.array_equal(
            np.asarray(data["environment_kp"], dtype=np.float64),
            np.asarray(controller["realized_environment_kp"], dtype=np.float64),
        ):
            failures.append("declared_controller_mismatch:environment_kp")
        if not np.array_equal(
            np.asarray(data["environment_kd"], dtype=np.float64),
            np.asarray(controller["realized_environment_kd"], dtype=np.float64),
        ):
            failures.append("declared_controller_mismatch:environment_kd")
        if recorded_weight_hash != controller["mpx_weight_matrix_sha256"]:
            failures.append("declared_controller_mismatch:mpx_weight_matrix")
        if float(_json_scalar(data, "controller_robot_height_m")) != float(
            controller["mpx_robot_height_m"]
        ):
            failures.append("declared_controller_mismatch:mpx_robot_height")
    for key in (
        "generator_root_commit",
        "generator_mpx_commit",
        "generator_primal_dual_ilqr_commit",
    ):
        commit = str(_json_scalar(data, key))
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            failures.append(f"invalid_provenance_commit:{key}")

    numeric_keys = (
        "qpos",
        "qvel",
        "tau_applied",
        "tau_mpx",
        "q_des",
        "time",
        "commands",
        "policy_obs",
        "next_policy_obs",
        "privileged_obs",
        "next_privileged_obs",
        "raw_actions",
        "actions",
        "action_clipping_magnitude",
        "rewards",
        "next_qpos_ctrl",
        "next_qvel_ctrl",
        "requested_torques_ctrl",
        "applied_torques_ctrl",
        "torque_saturation_magnitude",
        "push_delta_qvel",
    )
    for key in numeric_keys:
        if not np.all(np.isfinite(np.asarray(data[key]))):
            failures.append(f"nonfinite:{key}")
    actions = np.asarray(data["actions"], dtype=np.float64)
    raw_actions = np.asarray(data["raw_actions"], dtype=np.float64)
    clipping_mask = np.asarray(data["action_clipping_mask"], dtype=bool)
    clipping_magnitude = np.asarray(
        data["action_clipping_magnitude"], dtype=np.float64
    )
    if np.any((actions < -1.0) | (actions > 1.0)):
        failures.append("saved_action_out_of_range")
    if not np.array_equal(actions, np.clip(raw_actions, -1.0, 1.0)):
        failures.append("saved_action_not_clipped_raw_action")
    if not np.array_equal(clipping_mask, raw_actions != actions):
        failures.append("action_clipping_mask_inconsistent")
    if not np.array_equal(clipping_magnitude, np.abs(raw_actions - actions)):
        failures.append("action_clipping_magnitude_inconsistent")

    requested_torque = np.asarray(
        data["requested_torques_ctrl"], dtype=np.float64
    )
    applied_torque = np.asarray(data["applied_torques_ctrl"], dtype=np.float64)
    saturation_mask = np.asarray(data["torque_saturation_mask"], dtype=bool)
    saturation_magnitude = np.asarray(
        data["torque_saturation_magnitude"], dtype=np.float64
    )
    if not np.array_equal(saturation_mask, requested_torque != applied_torque):
        failures.append("torque_saturation_mask_inconsistent")
    if not np.allclose(
        saturation_magnitude,
        np.abs(requested_torque - applied_torque),
        rtol=0.0,
        atol=1e-14,
    ):
        failures.append("torque_saturation_magnitude_inconsistent")

    qpos_control = np.asarray(data["qpos"], dtype=np.float64)[
        :, decimation::decimation
    ].T
    qvel_control = np.asarray(data["qvel"], dtype=np.float64)[
        :, decimation::decimation
    ].T
    if not np.array_equal(qpos_control, np.asarray(data["next_qpos_ctrl"])):
        failures.append("control_qpos_not_equal_substep_trajectory")
    if not np.array_equal(qvel_control, np.asarray(data["next_qvel_ctrl"])):
        failures.append("control_qvel_not_equal_substep_trajectory")
    if not np.array_equal(
        np.asarray(data["policy_obs"])[:, 6:9], expected_schedule
    ):
        failures.append("policy_observation_command_schedule_mismatch")

    metrics = trajectory_quality_metrics(data, declaration)
    acceptance = declaration["acceptance"]
    contact = metrics["contact"]
    threshold_checks = {
        "front_pair_agreement": contact["front_pair_agreement"]
        >= float(acceptance["min_front_pair_agreement"]),
        "rear_pair_agreement": contact["rear_pair_agreement"]
        >= float(acceptance["min_rear_pair_agreement"]),
        "complete_front_rear_cycles": contact["complete_front_rear_cycles"]
        >= int(acceptance["min_complete_front_rear_cycles"]),
        "front_only_fraction": contact["front_only_fraction"]
        >= float(acceptance["min_front_only_fraction"]),
        "rear_only_fraction": contact["rear_only_fraction"]
        >= float(acceptance["min_rear_only_fraction"]),
        "paired_interval_alternation_fraction": contact[
            "paired_interval_alternation_fraction"
        ]
        >= float(acceptance["min_paired_interval_alternation_fraction"]),
        "diagonal_lateral_only_fraction": contact[
            "diagonal_lateral_only_fraction"
        ]
        <= float(acceptance["max_diagonal_lateral_only_fraction"]),
        "action_clipped_element_fraction": metrics["action_clipping"][
            "element_fraction"
        ]
        < float(acceptance["max_action_clipped_element_fraction"]),
        "action_clip_magnitude": metrics["action_clipping"]["max_magnitude"]
        <= float(acceptance["max_action_clip_magnitude"]),
        "base_height": metrics["posture"]["min_base_height_m"]
        >= float(acceptance["min_base_height_m"]),
        "roll": metrics["posture"]["max_abs_roll_rad"]
        <= float(acceptance["max_abs_roll_rad"]),
        "pitch": metrics["posture"]["max_abs_pitch_rad"]
        <= float(acceptance["max_abs_pitch_rad"]),
        "solver_finite": metrics["solver"]["all_finite"],
    }
    failures.extend(
        f"acceptance_failed:{name}"
        for name, passed in threshold_checks.items()
        if not passed
    )
    if replay_report is None:
        failures.append("full_saved_action_replay_not_run")
    elif not bool(replay_report.get("passed", False)):
        failures.append("full_saved_action_replay_failed")

    return {
        "passed": not failures,
        "failure_reasons": failures,
        "threshold_checks": threshold_checks,
        "metrics": metrics,
        "replay": replay_report,
    }


def load_npz_pickle_free(path: str | Path) -> dict[str, np.ndarray]:
    """Load a trajectory without permitting object arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
