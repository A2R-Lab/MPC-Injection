"""Schema-v1 validation and artifact helpers for Go2 barrel-roll data."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from mpc_rl.envs.barrel_roll_common import (
    ACTION_SCALE,
    CONTROL_DT,
    CONTROL_STEPS,
    EPISODE_HORIZON,
    FINAL_HOLD_CONTROL_STEPS,
    FINAL_HOLD_START_TIME,
    FINAL_STANCE_DURATION,
    FLIGHT_DURATION,
    INITIAL_STANCE_DURATION,
    LANDING_DURATION,
    LEGACY_ACTION_SCALE,
    LEGACY_CONTROL_STEPS,
    LEGACY_REWARD_CONFIG,
    LEGACY_SCHEMA_VERSION,
    LEGACY_SUCCESS_CONFIG,
    LATERAL_SUPPORT_DURATION,
    MANEUVER_HORIZON,
    REWARD_CONFIG,
    ROLL_DIRECTION_SIGN,
    ROLL_END_TIME,
    ROLL_START_TIME,
    SCHEMA_V2_VERSION,
    SCHEMA_V3_REWARD_CONFIG,
    SCHEMA_V3_SUCCESS_CONFIG,
    SCHEMA_V3_VERSION,
    SCHEMA_VERSION,
    SIM_DT,
    SPREAD_RANGE,
    SUCCESS_CONFIG,
    TASK_ID,
    RollProgressTracker,
    body_up_tilt_from_quaternion,
    desired_roll_at_time,
    desired_roll_rate_at_time,
    final_hold_conditions,
    legacy_maneuver_phase_at_time,
    maneuver_phase_at_time,
    reward_config_dict,
    schema_v3_maneuver_phase_at_time,
    schema_v3_terminal_failure_reason,
    standing_subscores,
    success_config_dict,
    terminal_failure_reason,
)

ROBOT_ID = "go2"
LEGACY_PHYSICS_STEPS = int(round(MANEUVER_HORIZON / SIM_DT))
PHYSICS_STEPS = int(round(EPISODE_HORIZON / SIM_DT))
MPC_DT = 0.01
MPC_NODES = 140
MPC_STATE_DIM = 61
MPC_CONTROL_DIM = 12
REPLANNING_FREQUENCY_HZ = 50.0
CONTACT_CONVENTION = "pyramidal"
CONTROLLER_MODE = "phase_limited_replanning"
WARM_START_POLICY = "shift_by_elapsed_mpx_nodes_and_terminal_pad"
ACTION_SEMANTICS = "direct_torque_transition_with_inverse_pd_residual"
DOMAIN_RANDOMIZATION = "disabled"
TRACKING_KP = np.tile(np.array([20.0, 20.0, 40.0]), 4)
TRACKING_KD = np.tile(np.array([1.0, 1.0, 2.0]), 4)
PD_KP = TRACKING_KP.copy()
PD_KD = TRACKING_KD.copy()
TORQUE_LIMITS = np.tile(np.array([[-23.7, 23.7], [-23.7, 23.7], [-45.43, 45.43]]), (4, 1))

def dimensions_for_schema(schema_version: int) -> tuple[int, int]:
    if schema_version in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION):
        return LEGACY_CONTROL_STEPS, LEGACY_PHYSICS_STEPS
    if schema_version in (SCHEMA_V3_VERSION, SCHEMA_VERSION):
        return CONTROL_STEPS, PHYSICS_STEPS
    raise ValueError(f"unsupported barrel-roll schema version {schema_version}")


def required_shapes_for_schema(schema_version: int) -> dict[str, tuple[int, ...]]:
    control_steps, physics_steps = dimensions_for_schema(schema_version)
    shapes = {
        "policy_obs": (control_steps, 45),
        "next_policy_obs": (control_steps, 45),
        "privileged_obs": (control_steps, 4),
        "next_privileged_obs": (control_steps, 4),
        "actions": (control_steps, 12),
        "rewards": (control_steps,),
        "terminated_ctrl": (control_steps,),
        "truncated_ctrl": (control_steps,),
        "qpos": (19, physics_steps + 1),
        "qvel": (18, physics_steps + 1),
        "tau_applied": (12, physics_steps),
        "tau_mpx": (12, physics_steps),
        "tau_raw": (12, physics_steps),
        "q_des": (12, physics_steps),
        "dq_des": (12, physics_steps),
        "X_updates": (control_steps, MPC_NODES + 1, MPC_STATE_DIM),
        "U_updates": (control_steps, MPC_NODES, MPC_CONTROL_DIM),
        "physics_time": (physics_steps + 1,),
        "control_time": (control_steps,),
        "phase_ctrl": (control_steps,),
        "desired_roll_ctrl": (control_steps,),
        "measured_roll_physics": (physics_steps + 1,),
        "foot_contacts": (physics_steps, 4),
        "nonfoot_contact": (physics_steps,),
        "stable_contact_streak": (control_steps,),
        "classifier_result": (control_steps,),
        "residual_actions_unclipped": (control_steps, 12),
        "action_clipped": (control_steps, 12),
        "mpx_saturation_by_actuator": (12, physics_steps),
        "applied_saturation_by_actuator": (12, physics_steps),
        "solve_seconds": (control_steps,),
        "solve_iterations": (control_steps,),
        "solve_iteration_limit": (control_steps,),
        "solve_objective_norm_sq": (control_steps,),
        "solve_constraint_norm_sq": (control_steps,),
        "solve_finite": (control_steps,),
        "solve_replanned": (control_steps,),
        "solve_phase_index": (control_steps,),
        "solve_elapsed_time": (control_steps,),
        "warm_start_shift": (control_steps,),
        "default_joint_pos": (12,),
        "pd_kp": (12,),
        "pd_kd": (12,),
        "tracking_kp": (12,),
        "tracking_kd": (12,),
        "torque_limits": (12, 2),
    }
    if schema_version == SCHEMA_V3_VERSION:
        shapes.update({
            "reward_roll_tracking": (control_steps,),
            "reward_rate_tracking": (control_steps,),
            "reward_action_change": (control_steps,),
            "reward_terminal_outcome": (control_steps,),
        })
    elif schema_version == SCHEMA_VERSION:
        shapes.update({
            "raw_foot_contacts_ctrl": (control_steps + 1, 4),
            "filtered_foot_contacts_ctrl": (control_steps, 4),
            "final_hold_streak": (control_steps,),
            "hold_rotation_valid": (control_steps,),
            "hold_foot_support_valid": (control_steps,),
            "hold_height_valid": (control_steps,),
            "hold_tilt_valid": (control_steps,),
            "hold_base_linear_speed_valid": (control_steps,),
            "hold_base_angular_speed_valid": (control_steps,),
            "hold_joint_speed_valid": (control_steps,),
            "base_height_ctrl": (control_steps,),
            "body_up_tilt_ctrl": (control_steps,),
            "base_linear_speed_ctrl": (control_steps,),
            "base_angular_speed_ctrl": (control_steps,),
            "joint_velocity_norm_ctrl": (control_steps,),
            "reward_roll_tracking": (control_steps,),
            "reward_signed_progress": (control_steps,),
            "reward_standing_score": (control_steps,),
            "reward_terminal_outcome": (control_steps,),
            "reward_foot_score": (control_steps,),
            "reward_height_score": (control_steps,),
            "reward_tilt_score": (control_steps,),
            "reward_linear_speed_score": (control_steps,),
            "reward_angular_speed_score": (control_steps,),
            "reward_joint_speed_score": (control_steps,),
        })
    return shapes


REQUIRED_SHAPES = required_shapes_for_schema(SCHEMA_VERSION)

REQUIRED_SCALARS = {
    "schema_version",
    "task_id",
    "robot_id",
    "roll_direction",
    "rollout_seed",
    "sampled_spread",
    "success",
    "failure_reason",
    "domain_randomization",
    "go2_sysid_enabled",
    "sim_dt",
    "control_dt",
    "decimation",
    "controller_steps",
    "physics_steps",
    "maneuver_horizon",
    "mpc_dt",
    "replanning_frequency_hz",
    "action_scale",
    "action_lpf_cutoff_hz",
    "action_lpf_alpha",
    "action_semantics",
    "saved_action_reproduces_lpf_transition",
    "contact_convention",
    "controller_mode",
    "warm_start_policy",
    "schedule_json",
    "success_config_json",
    "reward_config_json",
    "effective_config_json",
    "effective_config_sha256",
    "root_commit",
    "mpx_commit",
    "solver_commit",
    "gym_quadruped_commit",
    "root_worktree_dirty",
    "mpx_worktree_dirty",
    "mpx_xml_sha256",
    "rollout_xml_sha256",
    "generator_source_sha256",
    "generator_command",
    "generator_config_json",
    "runtime_versions_json",
    "final_roll",
    "final_pitch",
    "final_base_height",
    "final_roll_progress",
    "action_clip_fraction",
    "mpx_torque_saturation_fraction",
    "torque_saturation_fraction",
}

EXTENDED_REQUIRED_SCALARS = {
    "episode_horizon",
    "final_tilt",
}

SCHEMA_V4_REQUIRED_SCALARS = {
    "final_base_linear_speed",
    "final_base_angular_speed",
    "final_joint_velocity_norm",
    "terminal_final_hold_streak",
    "nonfoot_contact_count",
}


class BarrelRollValidationError(ValueError):
    """Raised when a barrel-roll dataset fails strict validation."""


@dataclass(frozen=True)
class ValidationReport:
    path: Path
    errors: tuple[str, ...]
    metrics: Mapping[str, float]

    @property
    def valid(self) -> bool:
        return not self.errors


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schedule_dict(schema_version: int = SCHEMA_VERSION) -> dict[str, float]:
    schedule = {
        "initial_stance_duration": INITIAL_STANCE_DURATION,
        "lateral_support_duration": LATERAL_SUPPORT_DURATION,
        "flight_duration": FLIGHT_DURATION,
        "landing_duration": LANDING_DURATION,
        "final_stance_duration": FINAL_STANCE_DURATION,
        "roll_start_time": ROLL_START_TIME,
        "roll_end_time": ROLL_END_TIME,
        "maneuver_horizon": MANEUVER_HORIZON,
    }
    if schema_version in (SCHEMA_V3_VERSION, SCHEMA_VERSION):
        schedule["episode_horizon"] = EPISODE_HORIZON
    elif schema_version not in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION):
        raise ValueError(f"unsupported barrel-roll schema version {schema_version}")
    return schedule


def action_scale_for_schema(schema_version: int) -> float:
    """Return the immutable action scale for a supported dataset schema."""
    if schema_version == LEGACY_SCHEMA_VERSION:
        return LEGACY_ACTION_SCALE
    if schema_version in (SCHEMA_V2_VERSION, SCHEMA_V3_VERSION, SCHEMA_VERSION):
        return ACTION_SCALE
    raise ValueError(f"unsupported barrel-roll schema version {schema_version}")


def expected_effective_config(schema_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    action_scale = action_scale_for_schema(schema_version)
    control_steps, physics_steps = dimensions_for_schema(schema_version)
    legacy_schema = schema_version in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION)
    config = {
        "action": {
            "lpf_cutoff_hz": 5.0,
            "scale": action_scale,
            "semantics": ACTION_SEMANTICS,
        },
        "contact_convention": CONTACT_CONVENTION,
        "classifier_contact_sampling": (
            "control_endpoint_raw_after_base_contact_update"
            if legacy_schema
            else (
                "diagnostic_only_control_endpoint_raw"
                if schema_version == SCHEMA_V3_VERSION
                else "current_or_genuine_previous_control_endpoint"
            )
        ),
        "controller": {
            "initial_solve_iterations": 100,
            "mode": CONTROLLER_MODE,
            "replan_earliest_stop_time": 0.86,
            "replan_stable_steps": 5,
            "replanning_frequency_hz": REPLANNING_FREQUENCY_HZ,
            "tracking_kd": TRACKING_KD.tolist(),
            "tracking_kp": TRACKING_KP.tolist(),
            "warm_start_policy": WARM_START_POLICY,
        },
        "domain_randomization": DOMAIN_RANDOMIZATION,
        "go2_sysid_enabled": True,
        "pd_kd": PD_KD.tolist(),
        "pd_kp": PD_KP.tolist(),
        "reward": reward_config_dict(schema_version),
        "robot": ROBOT_ID,
        "roll_direction": ROLL_DIRECTION_SIGN,
        "schedule": schedule_dict(schema_version),
        "schema_version": schema_version,
        "success": success_config_dict(schema_version),
        "task_id": TASK_ID,
        "timing": {
            "control_dt": CONTROL_DT,
            "control_steps": control_steps,
            "decimation": 4,
            "mpc_dt": MPC_DT,
            "physics_steps": physics_steps,
            "sim_dt": SIM_DT,
        },
        "torque_limits": TORQUE_LIMITS.tolist(),
    }
    if not legacy_schema:
        config["controller"].update({
            "maneuver_horizon": MANEUVER_HORIZON,
            "post_maneuver_execution": "terminal_padded_final_stance_same_pd",
        })
        config["timing"].update({
            "episode_horizon": EPISODE_HORIZON,
            "maneuver_horizon": MANEUVER_HORIZON,
        })
    if schema_version == SCHEMA_VERSION:
        config["failure"] = {
            "immediate": ["non_finite_state", "non_foot_ground_contact"],
            "otherwise_early_termination": False,
        }
        config["final_hold"] = {
            "control_steps": FINAL_HOLD_CONTROL_STEPS,
            "duration": FINAL_HOLD_CONTROL_STEPS * CONTROL_DT,
            "start_time": FINAL_HOLD_START_TIME,
        }
    return config


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty(repo: Path) -> bool:
    arguments = ["status", "--porcelain", "--untracked-files=no"]
    if repo.resolve() == Path(__file__).resolve().parents[2]:
        arguments.extend([
            "--",
            ".",
            ":(exclude)deploy/robots/go2/config/policy/velocity/policies/**",
        ])
    return bool(_git_value(repo, *arguments))


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_provenance_fields(
    *,
    env: Any,
    mpx_xml_path: Path,
    rollout_xml_path: Path,
    generator_source_path: Path,
    generator_command: str,
    generator_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Build fixed-width/numeric provenance from the effective frozen setup."""
    repo_root = Path(__file__).resolve().parents[2]
    mpx_repo = repo_root / "deps" / "mpx"
    solver_repo = mpx_repo / "mpx" / "primal_dual_ilqr"
    gym_repo = repo_root / "deps" / "gym-quadruped"
    effective = expected_effective_config()
    effective_json = canonical_json(effective)
    runtime = {
        "jax": _package_version("jax"),
        "mujoco": _package_version("mujoco"),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scipy": _package_version("scipy"),
    }
    fields: dict[str, Any] = {
        "robot_id": ROBOT_ID,
        "domain_randomization": DOMAIN_RANDOMIZATION,
        "sim_dt": float(env.sim_dt),
        "control_dt": float(env.control_dt),
        "decimation": int(env.decimation),
        "controller_steps": CONTROL_STEPS,
        "physics_steps": PHYSICS_STEPS,
        "maneuver_horizon": MANEUVER_HORIZON,
        "episode_horizon": EPISODE_HORIZON,
        "mpc_dt": MPC_DT,
        "replanning_frequency_hz": REPLANNING_FREQUENCY_HZ,
        "action_scale": float(env.action_scale),
        "action_lpf_cutoff_hz": float(env.action_lpf_cutoff_hz),
        "action_lpf_alpha": float(env.action_lpf_alpha),
        "action_semantics": ACTION_SEMANTICS,
        "saved_action_reproduces_lpf_transition": False,
        "contact_convention": CONTACT_CONVENTION,
        "controller_mode": CONTROLLER_MODE,
        "warm_start_policy": WARM_START_POLICY,
        "schedule_json": canonical_json(schedule_dict(SCHEMA_VERSION)),
        "success_config_json": canonical_json(success_config_dict(SCHEMA_VERSION)),
        "reward_config_json": canonical_json(reward_config_dict(SCHEMA_VERSION)),
        "effective_config_json": effective_json,
        "effective_config_sha256": sha256_bytes(effective_json.encode("utf-8")),
        "root_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "mpx_commit": _git_value(mpx_repo, "rev-parse", "HEAD"),
        "solver_commit": _git_value(solver_repo, "rev-parse", "HEAD"),
        "gym_quadruped_commit": _git_value(gym_repo, "rev-parse", "HEAD"),
        "root_worktree_dirty": _git_dirty(repo_root),
        "mpx_worktree_dirty": _git_dirty(mpx_repo),
        "mpx_xml_sha256": sha256_file(mpx_xml_path),
        "rollout_xml_sha256": sha256_file(rollout_xml_path),
        "generator_source_sha256": sha256_file(generator_source_path),
        "generator_command": generator_command,
        "generator_config_json": canonical_json(dict(generator_config)),
        "runtime_versions_json": canonical_json(runtime),
        "default_joint_pos": np.asarray(env.default_joint_pos, dtype=np.float64),
        "pd_kp": np.asarray(env.kp, dtype=np.float64),
        "pd_kd": np.asarray(env.kd, dtype=np.float64),
        "torque_limits": np.asarray(env.torque_limits, dtype=np.float64),
    }
    return {key: np.asarray(value) for key, value in fields.items()}


def atomic_save_npz(path: Path, arrays: Mapping[str, Any], *, validate: bool = True) -> None:
    """Write a compressed archive through a sibling temporary and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        if validate:
            report = validate_barrel_roll_file(temporary, logical_path=path)
            if not report.valid:
                raise BarrelRollValidationError("; ".join(report.errors))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar, got shape {value.shape}")
    return value.item()


def _json_scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    return json.loads(str(_scalar(data, key)))


def _recompute_roll(qpos: np.ndarray) -> np.ndarray:
    tracker = RollProgressTracker.from_reset_quaternion(qpos[3:7, 0])
    result = np.zeros(qpos.shape[1], dtype=np.float64)
    for index in range(1, qpos.shape[1]):
        result[index] = tracker.update(qpos[3:7, index])
    return result


def _recompute_stability(
    foot_contacts: np.ndarray, control_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    streaks = np.zeros(control_steps, dtype=np.int64)
    filtered = np.zeros((control_steps, 4), dtype=bool)
    streak = 0
    for control_index in range(control_steps):
        current = foot_contacts[(control_index + 1) * 4 - 1]
        # The commissioned environment updates its prior-contact state before
        # calling the task classifier, so the classifier's saved contact state
        # is the raw contact state at this controller boundary.
        filtered[control_index] = current
        streak = streak + 1 if np.all(filtered[control_index]) else 0
        streaks[control_index] = streak
    return streaks, filtered


def _recompute_v4_hold_data(
    data: Mapping[str, np.ndarray], post_roll: np.ndarray
) -> dict[str, Any]:
    """Recompute schema-v4 contact filtering, endpoint metrics, and hold state."""
    raw_contacts = np.asarray(data["raw_foot_contacts_ctrl"], dtype=bool)
    filtered_contacts = np.logical_or(raw_contacts[1:], raw_contacts[:-1])
    endpoint_indices = np.arange(1, CONTROL_STEPS + 1) * 4
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    qvel = np.asarray(data["qvel"], dtype=np.float64)
    base_heights = qpos[2, endpoint_indices]
    body_tilts = np.asarray(
        [
            body_up_tilt_from_quaternion(qpos[3:7, index])
            for index in endpoint_indices
        ],
        dtype=np.float64,
    )
    base_linear_speeds = np.linalg.norm(qvel[:3, endpoint_indices], axis=0)
    base_angular_speeds = np.linalg.norm(qvel[3:6, endpoint_indices], axis=0)
    joint_velocity_norms = np.linalg.norm(qvel[6:, endpoint_indices], axis=0)

    condition_values = {
        name: np.zeros(CONTROL_STEPS, dtype=bool)
        for name in (
            "rotation",
            "foot_support",
            "height",
            "tilt",
            "base_linear_speed",
            "base_angular_speed",
            "joint_speed",
        )
    }
    score_values = {
        name: np.zeros(CONTROL_STEPS, dtype=np.float64)
        for name in (
            "foot_score",
            "height_score",
            "tilt_score",
            "linear_speed_score",
            "angular_speed_score",
            "joint_speed_score",
        )
    }
    streaks = np.zeros(CONTROL_STEPS, dtype=np.int64)
    streak = 0
    for index in range(CONTROL_STEPS):
        metrics = {
            "base_height": base_heights[index],
            "body_up_tilt": body_tilts[index],
            "base_linear_speed": base_linear_speeds[index],
            "base_angular_speed": base_angular_speeds[index],
            "joint_velocity_norm": joint_velocity_norms[index],
        }
        conditions = final_hold_conditions(
            filtered_foot_contacts=filtered_contacts[index],
            roll_progress=post_roll[index],
            **metrics,
        )
        scores = standing_subscores(
            filtered_foot_contacts=filtered_contacts[index],
            **metrics,
        )
        for name, value in conditions.items():
            condition_values[name][index] = value
        for name, value in scores.items():
            score_values[name][index] = value
        streak = streak + 1 if all(conditions.values()) else 0
        streaks[index] = streak
    return {
        "filtered_contacts": filtered_contacts,
        "conditions": condition_values,
        "scores": score_values,
        "streaks": streaks,
        "base_heights": base_heights,
        "body_tilts": body_tilts,
        "base_linear_speeds": base_linear_speeds,
        "base_angular_speeds": base_angular_speeds,
        "joint_velocity_norms": joint_velocity_norms,
    }


def _recompute_v4_observations(
    data: Mapping[str, np.ndarray],
    roll_progress: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute all 45 actor and four privileged features at saved states."""
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    qvel = np.asarray(data["qvel"], dtype=np.float64)
    actions = np.asarray(data["actions"], dtype=np.float64)
    result_policy = np.zeros((CONTROL_STEPS + 1, 45), dtype=np.float64)
    result_privileged = np.zeros((CONTROL_STEPS + 1, 4), dtype=np.float64)
    previous_actions = np.vstack((np.zeros((1, 12), dtype=np.float64), actions))
    for control_index in range(CONTROL_STEPS + 1):
        state_index = control_index * 4
        quaternion_xyzw = np.roll(qpos[3:7, state_index], -1)
        rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
        time_s = control_index * CONTROL_DT
        desired = desired_roll_at_time(time_s)
        result_policy[control_index] = np.concatenate((
            qvel[3:6, state_index],
            rotation.T @ np.array([0.0, 0.0, -1.0]),
            np.array(
                [maneuver_phase_at_time(time_s), np.sin(desired), np.cos(desired)]
            ),
            qpos[7:, state_index] - data["default_joint_pos"],
            qvel[6:, state_index],
            previous_actions[control_index],
        ))
        result_privileged[control_index, :3] = (
            rotation.T @ qvel[:3, state_index]
        )
        result_privileged[control_index, 3] = roll_progress[state_index]
    return result_policy, result_privileged


def _add_mismatch(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_barrel_roll_file(
    path: Path | str, *, logical_path: Path | str | None = None
) -> ValidationReport:
    """Strictly validate one accepted versioned direct-transition archive."""
    path = Path(path)
    logical_path = path if logical_path is None else Path(logical_path)
    errors: list[str] = []
    metrics: dict[str, float] = {}
    try:
        with np.load(path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    except Exception as error:
        return ValidationReport(path, (f"archive load failed: {type(error).__name__}: {error}",), metrics)

    if "schema_version" not in data:
        return ValidationReport(path, ("missing required fields: schema_version",), metrics)
    try:
        schema_version = int(_scalar(data, "schema_version"))
        action_scale = action_scale_for_schema(schema_version)
        control_steps, physics_steps = dimensions_for_schema(schema_version)
        required_shapes = required_shapes_for_schema(schema_version)
    except (TypeError, ValueError) as error:
        return ValidationReport(path, (str(error),), metrics)

    required_scalars = set(REQUIRED_SCALARS)
    if schema_version in (SCHEMA_V3_VERSION, SCHEMA_VERSION):
        required_scalars |= EXTENDED_REQUIRED_SCALARS
    if schema_version == SCHEMA_VERSION:
        required_scalars |= SCHEMA_V4_REQUIRED_SCALARS
    missing = sorted((set(required_shapes) | required_scalars) - set(data))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
        return ValidationReport(path, tuple(errors), metrics)

    for key, value in data.items():
        if np.asarray(value).dtype.kind not in "biufcU":
            errors.append(f"{key} has forbidden dtype {np.asarray(value).dtype}")
    for key, shape in required_shapes.items():
        if data[key].shape != shape:
            errors.append(f"{key} shape mismatch: expected {shape}, got {data[key].shape}")
    for key in required_scalars:
        if np.asarray(data[key]).shape != ():
            errors.append(f"{key} must be scalar, got {np.asarray(data[key]).shape}")
    if errors:
        return ValidationReport(path, tuple(errors), metrics)

    if schema_version == SCHEMA_VERSION:
        rollout_seed = int(_scalar(data, "rollout_seed"))
        expected_filename = (
            f"go2_barrel_roll_v{SCHEMA_VERSION}_dir_pos_seed_{rollout_seed:06d}_"
            f"ep_{control_steps:03d}.npz"
        )
        if logical_path.name != expected_filename:
            errors.append(
                f"filename mismatch: expected {expected_filename!r}, "
                f"got {logical_path.name!r}"
            )

    for key, value in data.items():
        array = np.asarray(value)
        if array.dtype.kind in "fci" and not np.isfinite(array).all():
            errors.append(f"{key} contains non-finite values")

    scalar_expectations = {
        "schema_version": schema_version,
        "task_id": TASK_ID,
        "robot_id": ROBOT_ID,
        "roll_direction": ROLL_DIRECTION_SIGN,
        "success": True,
        "failure_reason": "",
        "domain_randomization": DOMAIN_RANDOMIZATION,
        "go2_sysid_enabled": True,
        "sim_dt": SIM_DT,
        "control_dt": CONTROL_DT,
        "decimation": 4,
        "controller_steps": control_steps,
        "physics_steps": physics_steps,
        "maneuver_horizon": MANEUVER_HORIZON,
        "mpc_dt": MPC_DT,
        "replanning_frequency_hz": REPLANNING_FREQUENCY_HZ,
        "action_scale": action_scale,
        "action_lpf_cutoff_hz": 5.0,
        "action_semantics": ACTION_SEMANTICS,
        "saved_action_reproduces_lpf_transition": False,
        "contact_convention": CONTACT_CONVENTION,
        "controller_mode": CONTROLLER_MODE,
        "warm_start_policy": WARM_START_POLICY,
    }
    if schema_version in (SCHEMA_V3_VERSION, SCHEMA_VERSION):
        scalar_expectations["episode_horizon"] = EPISODE_HORIZON
    for key, expected in scalar_expectations.items():
        actual = _scalar(data, key)
        if isinstance(expected, float):
            if not np.isclose(actual, expected, atol=1e-12, rtol=0.0):
                errors.append(f"{key} mismatch: expected {expected}, got {actual}")
        else:
            _add_mismatch(errors, key, actual, expected)
    spread = float(_scalar(data, "sampled_spread"))
    if not SPREAD_RANGE[0] <= spread <= SPREAD_RANGE[1]:
        errors.append(f"sampled_spread outside {SPREAD_RANGE}: {spread}")

    expected_lpf_alpha = 1.0 - np.exp(-2.0 * np.pi * 5.0 * CONTROL_DT)
    if not np.isclose(_scalar(data, "action_lpf_alpha"), expected_lpf_alpha, atol=1e-12, rtol=0.0):
        errors.append("action_lpf_alpha does not match the frozen 5 Hz online LPF")
    if not np.allclose(data["pd_kp"], PD_KP, atol=0.0, rtol=0.0):
        errors.append("pd_kp mismatch")
    if not np.allclose(data["pd_kd"], PD_KD, atol=0.0, rtol=0.0):
        errors.append("pd_kd mismatch")
    if not np.allclose(data["tracking_kp"], TRACKING_KP, atol=0.0, rtol=0.0):
        errors.append("tracking_kp mismatch")
    if not np.allclose(data["tracking_kd"], TRACKING_KD, atol=0.0, rtol=0.0):
        errors.append("tracking_kd mismatch")
    if not np.allclose(data["torque_limits"], TORQUE_LIMITS, atol=1e-6, rtol=0.0):
        errors.append("torque_limits mismatch")

    json_expectations = {
        "schedule_json": schedule_dict(schema_version),
        "success_config_json": success_config_dict(schema_version),
        "reward_config_json": reward_config_dict(schema_version),
        "effective_config_json": expected_effective_config(schema_version),
    }
    for key, expected in json_expectations.items():
        try:
            actual = _json_scalar(data, key)
        except (TypeError, json.JSONDecodeError) as error:
            errors.append(f"{key} is invalid JSON: {error}")
            continue
        if actual != expected:
            errors.append(
                f"{key} does not match the frozen schema-v{schema_version} configuration"
            )
    effective_json = str(_scalar(data, "effective_config_json"))
    if _scalar(data, "effective_config_sha256") != sha256_bytes(effective_json.encode("utf-8")):
        errors.append("effective_config_sha256 mismatch")
    for key in (
        "root_commit", "mpx_commit", "solver_commit", "gym_quadruped_commit"
    ):
        value = str(_scalar(data, key))
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            errors.append(f"{key} is not a full lowercase Git commit")
    for key in (
        "mpx_xml_sha256", "rollout_xml_sha256", "generator_source_sha256",
        "effective_config_sha256",
    ):
        value = str(_scalar(data, key))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            errors.append(f"{key} is not a SHA-256 digest")
    for key in ("generator_config_json", "runtime_versions_json"):
        try:
            value = _json_scalar(data, key)
            if not isinstance(value, dict) or not value:
                errors.append(f"{key} must encode a non-empty object")
        except (TypeError, json.JSONDecodeError) as error:
            errors.append(f"{key} is invalid JSON: {error}")
    if not str(_scalar(data, "generator_command")).strip():
        errors.append("generator_command is empty")

    physics_time = np.arange(physics_steps + 1, dtype=np.float64) * SIM_DT
    control_time = np.arange(1, control_steps + 1, dtype=np.float64) * CONTROL_DT
    phase_function = {
        LEGACY_SCHEMA_VERSION: legacy_maneuver_phase_at_time,
        SCHEMA_V2_VERSION: legacy_maneuver_phase_at_time,
        SCHEMA_V3_VERSION: schema_v3_maneuver_phase_at_time,
        SCHEMA_VERSION: maneuver_phase_at_time,
    }[schema_version]
    phase_ctrl = np.array([phase_function(t) for t in control_time])
    desired_ctrl = np.array([desired_roll_at_time(t) for t in control_time])
    if not np.allclose(data["physics_time"], physics_time, atol=1e-12, rtol=0.0):
        errors.append("physics_time mismatch")
    if not np.allclose(data["control_time"], control_time, atol=1e-12, rtol=0.0):
        errors.append("control_time mismatch")
    if not np.allclose(data["phase_ctrl"], phase_ctrl, atol=1e-12, rtol=0.0):
        errors.append("phase_ctrl mismatch")
    if not np.allclose(data["desired_roll_ctrl"], desired_ctrl, atol=1e-12, rtol=0.0):
        errors.append("desired_roll_ctrl mismatch")

    try:
        recomputed_roll = _recompute_roll(data["qpos"])
    except ValueError as error:
        errors.append(f"qpos quaternion sequence is invalid: {error}")
        return ValidationReport(path, tuple(errors), metrics)
    if not np.allclose(data["measured_roll_physics"], recomputed_roll, atol=1e-8, rtol=0.0):
        errors.append("measured_roll_physics mismatch with qpos quaternions")
    pre_roll = recomputed_roll[np.arange(control_steps) * 4]
    post_roll = recomputed_roll[np.arange(1, control_steps + 1) * 4]
    if not np.allclose(data["privileged_obs"][:, 3], pre_roll, atol=1e-8, rtol=0.0):
        errors.append("privileged_obs roll progress mismatch")
    if not np.allclose(data["next_privileged_obs"][:, 3], post_roll, atol=1e-8, rtol=0.0):
        errors.append("next_privileged_obs roll progress mismatch")

    pre_time = np.arange(control_steps, dtype=np.float64) * CONTROL_DT
    pre_phase = np.array([phase_function(t) for t in pre_time])
    pre_desired = np.array([desired_roll_at_time(t) for t in pre_time])
    expected_pre_features = np.column_stack((pre_phase, np.sin(pre_desired), np.cos(pre_desired)))
    expected_next_features = np.column_stack((phase_ctrl, np.sin(desired_ctrl), np.cos(desired_ctrl)))
    if not np.allclose(data["policy_obs"][:, 6:9], expected_pre_features, atol=1e-10, rtol=0.0):
        errors.append("policy_obs phase/desired-roll features mismatch")
    if not np.allclose(data["next_policy_obs"][:, 6:9], expected_next_features, atol=1e-10, rtol=0.0):
        errors.append("next_policy_obs phase/desired-roll features mismatch")
    if not np.array_equal(data["next_policy_obs"][:-1], data["policy_obs"][1:]):
        errors.append("policy observation transition adjacency is broken")
    if not np.array_equal(data["next_privileged_obs"][:-1], data["privileged_obs"][1:]):
        errors.append("privileged observation transition adjacency is broken")
    if schema_version == SCHEMA_VERSION:
        try:
            expected_policy, expected_privileged = _recompute_v4_observations(
                data, recomputed_roll
            )
        except ValueError as error:
            errors.append(f"schema-v4 observation recomputation failed: {error}")
        else:
            if not np.allclose(
                data["policy_obs"], expected_policy[:-1], atol=1e-9, rtol=0.0
            ):
                errors.append("policy_obs mismatch with full schema-v4 recomputation")
            if not np.allclose(
                data["next_policy_obs"], expected_policy[1:], atol=1e-9, rtol=0.0
            ):
                errors.append(
                    "next_policy_obs mismatch with full schema-v4 recomputation"
                )
            if not np.allclose(
                data["privileged_obs"], expected_privileged[:-1], atol=1e-9, rtol=0.0
            ):
                errors.append(
                    "privileged_obs mismatch with full schema-v4 recomputation"
                )
            if not np.allclose(
                data["next_privileged_obs"],
                expected_privileged[1:],
                atol=1e-9,
                rtol=0.0,
            ):
                errors.append(
                    "next_privileged_obs mismatch with full schema-v4 recomputation"
                )

    actions = data["actions"]
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        errors.append("saved actions are outside [-1, 1]")
    expected_actions = np.clip(data["residual_actions_unclipped"], -1.0, 1.0)
    expected_action_clipped = np.abs(data["residual_actions_unclipped"]) > 1.0
    if not np.allclose(actions, expected_actions, atol=1e-12, rtol=0.0):
        errors.append("saved actions do not match clipped inverse-PD residuals")
    if not np.array_equal(data["action_clipped"], expected_action_clipped):
        errors.append("action_clipped mask mismatch")

    expected_mpx_saturation = (
        (data["tau_mpx"] < data["torque_limits"][:, 0, None])
        | (data["tau_mpx"] > data["torque_limits"][:, 1, None])
    )
    expected_applied = np.clip(
        data["tau_raw"],
        data["torque_limits"][:, 0, None],
        data["torque_limits"][:, 1, None],
    )
    expected_applied_saturation = ~np.isclose(data["tau_raw"], expected_applied)
    if not np.allclose(data["tau_applied"], expected_applied, atol=1e-10, rtol=0.0):
        errors.append("tau_applied does not match clipped tau_raw")
    if not np.array_equal(data["mpx_saturation_by_actuator"], expected_mpx_saturation):
        errors.append("mpx_saturation_by_actuator mismatch")
    if not np.array_equal(data["applied_saturation_by_actuator"], expected_applied_saturation):
        errors.append("applied_saturation_by_actuator mismatch")

    action_clip_fraction = float(np.mean(expected_action_clipped))
    mpx_saturation_fraction = float(np.mean(expected_mpx_saturation))
    torque_saturation_fraction = float(np.mean(expected_applied_saturation))
    metrics.update(
        action_clip_fraction=action_clip_fraction,
        mpx_torque_saturation_fraction=mpx_saturation_fraction,
        torque_saturation_fraction=torque_saturation_fraction,
    )
    for key, expected in (
        ("action_clip_fraction", action_clip_fraction),
        ("mpx_torque_saturation_fraction", mpx_saturation_fraction),
        ("torque_saturation_fraction", torque_saturation_fraction),
    ):
        if not np.isclose(_scalar(data, key), expected, atol=1e-12, rtol=0.0):
            errors.append(f"{key} mismatch")

    hold_data = None
    if schema_version == SCHEMA_VERSION:
        expected_raw_endpoints = data["foot_contacts"][3::4]
        if not np.array_equal(
            data["raw_foot_contacts_ctrl"][1:], expected_raw_endpoints
        ):
            errors.append(
                "raw_foot_contacts_ctrl does not match physics endpoint contacts"
            )
        hold_data = _recompute_v4_hold_data(data, post_roll)
        filtered_contacts = hold_data["filtered_contacts"]
        streaks = hold_data["streaks"]
        controller_contact_streaks, _ = _recompute_stability(
            data["foot_contacts"], control_steps
        )
        if not np.array_equal(
            data["filtered_foot_contacts_ctrl"], filtered_contacts
        ):
            errors.append("filtered_foot_contacts_ctrl mismatch")
        if not np.array_equal(data["final_hold_streak"], streaks):
            errors.append("final_hold_streak mismatch")
        if not np.array_equal(
            data["stable_contact_streak"], controller_contact_streaks
        ):
            errors.append("stable_contact_streak mismatch")
        for name, expected in hold_data["conditions"].items():
            key = f"hold_{name}_valid"
            if not np.array_equal(data[key], expected):
                errors.append(f"{key} mismatch")
        for key, expected in (
            ("base_height_ctrl", hold_data["base_heights"]),
            ("body_up_tilt_ctrl", hold_data["body_tilts"]),
            ("base_linear_speed_ctrl", hold_data["base_linear_speeds"]),
            ("base_angular_speed_ctrl", hold_data["base_angular_speeds"]),
            ("joint_velocity_norm_ctrl", hold_data["joint_velocity_norms"]),
        ):
            if not np.allclose(data[key], expected, atol=1e-10, rtol=0.0):
                errors.append(f"{key} mismatch")
        nonfoot_count = int(np.count_nonzero(data["nonfoot_contact"] != ""))
        if nonfoot_count:
            errors.append("non-foot ground contact recorded")
        if int(_scalar(data, "nonfoot_contact_count")) != nonfoot_count:
            errors.append("nonfoot_contact_count mismatch")
        final_hold_slice = slice(control_steps - FINAL_HOLD_CONTROL_STEPS, None)
        for name, values in hold_data["conditions"].items():
            if not np.all(values[final_hold_slice]):
                errors.append(f"final hold condition failed: {name}")
        if int(streaks[-1]) < FINAL_HOLD_CONTROL_STEPS:
            errors.append("fewer than 25 consecutive valid final-hold endpoints")
    else:
        streaks, filtered_contacts = _recompute_stability(
            data["foot_contacts"], control_steps
        )
        if not np.array_equal(data["stable_contact_streak"], streaks):
            errors.append("stable_contact_streak mismatch")
        if (
            schema_version in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION)
            and np.any(data["nonfoot_contact"] != "")
        ):
            errors.append("non-foot ground contact recorded")

    terminated_expected = np.zeros(control_steps, dtype=bool)
    terminated_expected[-1] = True
    if not np.array_equal(data["terminated_ctrl"], terminated_expected):
        errors.append("terminated_ctrl mismatch")
    if np.any(data["truncated_ctrl"]):
        errors.append("truncated_ctrl must be false for accepted full-horizon rollouts")

    if schema_version in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION):
        expected_step = 2.0 * np.pi * CONTROL_DT / (
            ROLL_END_TIME - ROLL_START_TIME
        )
        tracking = LEGACY_REWARD_CONFIG.tracking_weight * np.exp(
            -((desired_ctrl - post_roll) / LEGACY_REWARD_CONFIG.tracking_sigma) ** 2
        )
        progress = LEGACY_REWARD_CONFIG.progress_weight * np.clip(
            ROLL_DIRECTION_SIGN * (post_roll - pre_roll) / expected_step,
            -LEGACY_REWARD_CONFIG.progress_clip,
            LEGACY_REWARD_CONFIG.progress_clip,
        )
        expected_rewards = tracking + progress
        expected_rewards[-1] += LEGACY_REWARD_CONFIG.success_bonus
    elif schema_version == SCHEMA_V3_VERSION:
        tracking = SCHEMA_V3_REWARD_CONFIG.tracking_weight * np.exp(
            -(
                (desired_ctrl - post_roll)
                / SCHEMA_V3_REWARD_CONFIG.tracking_sigma
            )
            ** 2
        )
        desired_rates = np.column_stack((
            np.array([desired_roll_rate_at_time(time_s) for time_s in control_time]),
            np.zeros(control_steps),
            np.zeros(control_steps),
        ))
        angular_velocity = data["qvel"][3:6, np.arange(1, control_steps + 1) * 4].T
        rate_errors = np.linalg.norm(angular_velocity - desired_rates, axis=1)
        rate_tracking = SCHEMA_V3_REWARD_CONFIG.rate_tracking_weight * np.exp(
            -(rate_errors / SCHEMA_V3_REWARD_CONFIG.rate_tracking_sigma) ** 2
        )
        previous_actions = np.vstack((np.zeros((1, 12)), actions[:-1]))
        action_change = SCHEMA_V3_REWARD_CONFIG.action_change_weight * np.sum(
            (actions - previous_actions) ** 2, axis=1
        )
        terminal_outcome = np.zeros(control_steps, dtype=np.float64)
        terminal_outcome[-1] = SCHEMA_V3_REWARD_CONFIG.success_bonus
        expected_components = {
            "reward_roll_tracking": tracking,
            "reward_rate_tracking": rate_tracking,
            "reward_action_change": action_change,
            "reward_terminal_outcome": terminal_outcome,
        }
        for key, expected in expected_components.items():
            if not np.allclose(data[key], expected, atol=1e-10, rtol=0.0):
                errors.append(f"{key} mismatch with frozen reward recomputation")
        expected_rewards = sum(expected_components.values())
        positive_dense = tracking + rate_tracking
        metrics.update({
            "cumulative_roll_tracking": float(np.sum(tracking)),
            "cumulative_rate_tracking": float(np.sum(rate_tracking)),
            "cumulative_action_change": float(np.sum(action_change)),
            "cumulative_positive_dense_reward": float(np.sum(positive_dense)),
        })
    else:
        assert hold_data is not None
        tracking = REWARD_CONFIG.tracking_weight * np.exp(
            -((desired_ctrl - post_roll) / REWARD_CONFIG.tracking_sigma) ** 2
        )
        expected_step = 2.0 * np.pi * CONTROL_DT / (
            ROLL_END_TIME - ROLL_START_TIME
        )
        signed_progress = REWARD_CONFIG.progress_weight * np.clip(
            ROLL_DIRECTION_SIGN * (post_roll - pre_roll) / expected_step,
            -REWARD_CONFIG.progress_clip,
            REWARD_CONFIG.progress_clip,
        )
        signed_progress[control_time > ROLL_END_TIME] = 0.0
        standing_subscores_by_name = hold_data["scores"]
        standing_score = np.mean(
            np.column_stack(tuple(standing_subscores_by_name.values())), axis=1
        )
        standing_score[control_time < REWARD_CONFIG.standing_start_time] = 0.0
        terminal_outcome = np.zeros(control_steps, dtype=np.float64)
        terminal_outcome[-1] = REWARD_CONFIG.success_bonus
        expected_components = {
            "reward_roll_tracking": tracking,
            "reward_signed_progress": signed_progress,
            "reward_standing_score": standing_score,
            "reward_terminal_outcome": terminal_outcome,
            **{
                f"reward_{name}": values
                for name, values in standing_subscores_by_name.items()
            },
        }
        for key, expected in expected_components.items():
            if not np.allclose(data[key], expected, atol=1e-10, rtol=0.0):
                errors.append(f"{key} mismatch with frozen reward recomputation")
        expected_rewards = (
            tracking + signed_progress + standing_score + terminal_outcome
        )
        metrics.update({
            "cumulative_roll_tracking": float(np.sum(tracking)),
            "cumulative_signed_progress": float(np.sum(signed_progress)),
            "cumulative_standing_score": float(np.sum(standing_score)),
            **{
                f"cumulative_{name}": float(np.sum(values))
                for name, values in standing_subscores_by_name.items()
            },
        })
    if not np.allclose(data["rewards"], expected_rewards, atol=1e-8, rtol=0.0):
        errors.append("rewards mismatch with frozen reward recomputation")

    final_rotation = Rotation.from_quat(np.roll(data["qpos"][3:7, -1], -1))
    final_roll, final_pitch, _ = final_rotation.as_euler("xyz")
    final_height = float(data["qpos"][2, -1])
    final_tilt = body_up_tilt_from_quaternion(data["qpos"][3:7, -1])
    if schema_version in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION):
        progress_ok = (
            ROLL_DIRECTION_SIGN * post_roll[-1]
            >= 2.0 * np.pi - LEGACY_SUCCESS_CONFIG.progress_tolerance
        )
        upright_ok = (
            abs(final_roll) <= LEGACY_SUCCESS_CONFIG.upright_tolerance
            and abs(final_pitch) <= LEGACY_SUCCESS_CONFIG.upright_tolerance
        )
        stable_ok = (
            streaks[-1] >= LEGACY_SUCCESS_CONFIG.stable_control_steps
            and np.all(filtered_contacts[-1])
        )
        height_ok = final_height >= LEGACY_SUCCESS_CONFIG.min_base_height
        classifier = progress_ok and upright_ok and stable_ok and height_ok
        expected_classifier_values = []
        for index in range(control_steps):
            state_index = (index + 1) * 4
            roll_error, pitch_error, _ = Rotation.from_quat(
                np.roll(data["qpos"][3:7, state_index], -1)
            ).as_euler("xyz")
            expected_classifier_values.append(
                ROLL_DIRECTION_SIGN * post_roll[index]
                >= 2.0 * np.pi - LEGACY_SUCCESS_CONFIG.progress_tolerance
                and streaks[index] >= LEGACY_SUCCESS_CONFIG.stable_control_steps
                and data["qpos"][2, state_index]
                >= LEGACY_SUCCESS_CONFIG.min_base_height
                and abs(roll_error) <= LEGACY_SUCCESS_CONFIG.upright_tolerance
                and abs(pitch_error) <= LEGACY_SUCCESS_CONFIG.upright_tolerance
            )
        expected_classifier = np.asarray(expected_classifier_values, dtype=bool)
        if not progress_ok:
            errors.append("incomplete rotation")
        if not upright_ok or not height_ok or not stable_ok:
            errors.append("unstable landing")
    elif schema_version == SCHEMA_V3_VERSION:
        failure_reason = schema_v3_terminal_failure_reason(
            post_roll[-1], final_height, final_tilt
        )
        classifier = failure_reason is None
        expected_classifier = np.zeros(control_steps, dtype=bool)
        expected_classifier[-1] = classifier
        if failure_reason is not None:
            errors.append(f"terminal classifier failed: {failure_reason}")
    else:
        assert hold_data is not None
        final_slice = slice(control_steps - FINAL_HOLD_CONTROL_STEPS, None)
        failure_counts = {
            name: int(np.count_nonzero(~values[final_slice]))
            for name, values in hold_data["conditions"].items()
        }
        failure_reason = terminal_failure_reason(failure_counts)
        classifier = bool(
            failure_reason is None
            and int(streaks[-1]) >= FINAL_HOLD_CONTROL_STEPS
        )
        expected_classifier = np.zeros(control_steps, dtype=bool)
        expected_classifier[-1] = classifier
        if not classifier:
            errors.append(
                "terminal classifier failed: "
                + (failure_reason or "final_hold_streak_too_short")
            )
    if not np.array_equal(data["classifier_result"], expected_classifier):
        errors.append("classifier_result mismatch")
    if not classifier:
        errors.append("terminal success classifier failed")
    for key, actual, expected in (
        ("final_roll", _scalar(data, "final_roll"), final_roll),
        ("final_pitch", _scalar(data, "final_pitch"), final_pitch),
        ("final_base_height", _scalar(data, "final_base_height"), final_height),
        ("final_roll_progress", _scalar(data, "final_roll_progress"), post_roll[-1]),
    ):
        if not np.isclose(actual, expected, atol=1e-8, rtol=0.0):
            errors.append(f"{key} mismatch")
    if schema_version in (SCHEMA_V3_VERSION, SCHEMA_VERSION):
        if not np.isclose(_scalar(data, "final_tilt"), final_tilt, atol=1e-8, rtol=0.0):
            errors.append("final_tilt mismatch")
        metrics.update({
            "final_roll_progress": float(post_roll[-1]),
            "final_base_height": final_height,
            "final_tilt": final_tilt,
        })
    if schema_version == SCHEMA_VERSION:
        assert hold_data is not None
        for key, expected in (
            ("final_base_linear_speed", hold_data["base_linear_speeds"][-1]),
            ("final_base_angular_speed", hold_data["base_angular_speeds"][-1]),
            ("final_joint_velocity_norm", hold_data["joint_velocity_norms"][-1]),
            ("terminal_final_hold_streak", int(streaks[-1])),
        ):
            actual = _scalar(data, key)
            if isinstance(expected, (int, np.integer)):
                if int(actual) != int(expected):
                    errors.append(f"{key} mismatch")
            elif not np.isclose(actual, expected, atol=1e-10, rtol=0.0):
                errors.append(f"{key} mismatch")
        metrics.update({
            "final_base_linear_speed": float(hold_data["base_linear_speeds"][-1]),
            "final_base_angular_speed": float(hold_data["base_angular_speeds"][-1]),
            "final_joint_velocity_norm": float(hold_data["joint_velocity_norms"][-1]),
            "final_hold_streak": float(streaks[-1]),
        })

    expected_phase_index = np.minimum(
        np.arange(control_steps, dtype=np.int64) * 2,
        MPC_NODES,
    )
    expected_elapsed = expected_phase_index.astype(np.float64) * MPC_DT
    if not np.array_equal(data["solve_phase_index"], expected_phase_index):
        errors.append("solve_phase_index mismatch")
    if not np.allclose(data["solve_elapsed_time"], expected_elapsed, atol=1e-12, rtol=0.0):
        errors.append("solve_elapsed_time mismatch")
    expected_shift = np.diff(expected_phase_index, prepend=expected_phase_index[0])
    if not np.array_equal(data["warm_start_shift"], expected_shift):
        errors.append("warm_start_shift mismatch")
    if not np.all(data["solve_finite"]):
        errors.append("non-finite solver status recorded")
    if not bool(data["solve_replanned"][0]):
        errors.append("initial MPC update was not replanned")
    if int(data["solve_iterations"][0]) != 100 or int(data["solve_iteration_limit"][0]) != 100:
        errors.append("initial solver iteration policy mismatch")
    if np.any(data["solve_iterations"][1:] > 1) or np.any(data["solve_iteration_limit"][1:] > 1):
        errors.append("subsequent solver iteration policy mismatch")
    non_replanned = ~data["solve_replanned"]
    if np.any(data["solve_iterations"][non_replanned] != 0):
        errors.append("non-replanned updates recorded solver iterations")
    if np.any(data["solve_iteration_limit"][non_replanned] != 0):
        errors.append("non-replanned updates recorded a positive iteration limit")
    if schema_version in (SCHEMA_V3_VERSION, SCHEMA_VERSION):
        tail_start = int(round(MANEUVER_HORIZON / CONTROL_DT))
        if np.any(data["solve_replanned"][tail_start:]):
            errors.append("MPC replanning continued at or after the maneuver horizon")
        for key in ("tau_mpx", "q_des", "dq_des"):
            tail = data[key].reshape(12, control_steps, 4)[:, tail_start:, :]
            if not np.array_equal(tail, np.broadcast_to(tail[:, :1, :], tail.shape)):
                errors.append(f"{key} does not repeat the terminal-padded hold")
        for key in ("X_updates", "U_updates"):
            tail = data[key][tail_start:]
            if not np.array_equal(tail, np.broadcast_to(tail[:1], tail.shape)):
                errors.append(f"{key} changes during the terminal-padded hold")

    return ValidationReport(path, tuple(errors), metrics)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def aggregate_barrel_roll_dataset(directory: Path | str) -> dict[str, Any]:
    """Validate a completed shard set and write its aggregate/checksum/summary."""
    directory = Path(directory)
    trajectory_files = sorted(directory.glob("go2_barrel_roll_v*_*.npz"))
    if not trajectory_files:
        raise BarrelRollValidationError(f"no versioned barrel-roll files in {directory}")
    reports = [validate_barrel_roll_file(path) for path in trajectory_files]
    invalid = [report for report in reports if not report.valid]
    if invalid:
        details = "; ".join(f"{item.path.name}: {', '.join(item.errors)}" for item in invalid)
        raise BarrelRollValidationError(details)

    seeds: list[int] = []
    schema_versions: set[int] = set()
    config_hashes: set[str] = set()
    clipping: list[float] = []
    mpx_saturation: list[float] = []
    torque_saturation: list[float] = []
    cumulative_roll_tracking: list[float] = []
    cumulative_rate_tracking: list[float] = []
    cumulative_action_change: list[float] = []
    cumulative_positive_dense: list[float] = []
    cumulative_signed_progress: list[float] = []
    cumulative_standing_score: list[float] = []
    cumulative_standing_subscores: dict[str, list[float]] = {
        name: []
        for name in (
            "foot_score",
            "height_score",
            "tilt_score",
            "linear_speed_score",
            "angular_speed_score",
            "joint_speed_score",
        )
    }
    final_progresses: list[float] = []
    final_heights: list[float] = []
    final_tilts: list[float] = []
    final_linear_speeds: list[float] = []
    final_angular_speeds: list[float] = []
    final_joint_speeds: list[float] = []
    final_hold_streaks: list[float] = []
    for path, report in zip(trajectory_files, reports, strict=True):
        with np.load(path, allow_pickle=False) as data:
            seeds.append(int(data["rollout_seed"]))
            schema_versions.add(int(data["schema_version"]))
            config_hashes.add(str(data["effective_config_sha256"]))
        clipping.append(report.metrics["action_clip_fraction"])
        mpx_saturation.append(report.metrics["mpx_torque_saturation_fraction"])
        torque_saturation.append(report.metrics["torque_saturation_fraction"])
        if "cumulative_rate_tracking" in report.metrics:
            cumulative_roll_tracking.append(report.metrics["cumulative_roll_tracking"])
            cumulative_rate_tracking.append(report.metrics["cumulative_rate_tracking"])
            cumulative_action_change.append(report.metrics["cumulative_action_change"])
            cumulative_positive_dense.append(
                report.metrics["cumulative_positive_dense_reward"]
            )
            final_progresses.append(report.metrics["final_roll_progress"])
            final_heights.append(report.metrics["final_base_height"])
            final_tilts.append(report.metrics["final_tilt"])
        elif "cumulative_signed_progress" in report.metrics:
            cumulative_roll_tracking.append(report.metrics["cumulative_roll_tracking"])
            cumulative_signed_progress.append(report.metrics["cumulative_signed_progress"])
            cumulative_standing_score.append(report.metrics["cumulative_standing_score"])
            for name, values in cumulative_standing_subscores.items():
                values.append(report.metrics[f"cumulative_{name}"])
            final_progresses.append(report.metrics["final_roll_progress"])
            final_heights.append(report.metrics["final_base_height"])
            final_tilts.append(report.metrics["final_tilt"])
            final_linear_speeds.append(report.metrics["final_base_linear_speed"])
            final_angular_speeds.append(report.metrics["final_base_angular_speed"])
            final_joint_speeds.append(report.metrics["final_joint_velocity_norm"])
            final_hold_streaks.append(report.metrics["final_hold_streak"])
    if len(set(seeds)) != len(seeds):
        raise BarrelRollValidationError("duplicate rollout seeds in dataset")
    if len(schema_versions) != 1:
        raise BarrelRollValidationError("dataset mixes schema versions")
    if len(config_hashes) != 1:
        raise BarrelRollValidationError("dataset mixes effective configuration hashes")

    manifest_paths = sorted(
        path for path in directory.glob("generation_manifest*.jsonl")
        if path.name != "generation_manifest_aggregate.jsonl"
    )
    manifest_records: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BarrelRollValidationError(
                    f"{manifest_path.name}:{line_number}: invalid JSON: {error}"
                ) from error
            record["worker_manifest"] = manifest_path.name
            manifest_records.append(record)
    accepted_records = [record for record in manifest_records if record.get("accepted")]
    if len(accepted_records) != len(trajectory_files):
        raise BarrelRollValidationError(
            f"manifest accepted count {len(accepted_records)} != file count {len(trajectory_files)}"
        )
    accepted_names = [record.get("trajectory_file") for record in accepted_records]
    if len(set(accepted_names)) != len(accepted_names):
        raise BarrelRollValidationError("duplicate accepted filenames in manifests")
    if set(accepted_names) != {path.name for path in trajectory_files}:
        raise BarrelRollValidationError("manifest accepted filenames do not match dataset files")
    seed_by_filename = dict(zip((path.name for path in trajectory_files), seeds, strict=True))
    for record in accepted_records:
        filename = record["trajectory_file"]
        if int(record.get("seed", -1)) != seed_by_filename[filename]:
            raise BarrelRollValidationError(
                f"manifest seed does not match archive {filename}"
            )

    aggregate_manifest_path = directory / "generation_manifest_aggregate.jsonl"
    aggregate_text = "".join(canonical_json(record) + "\n" for record in manifest_records)
    _atomic_write_text(aggregate_manifest_path, aggregate_text)

    checksum_path = directory / "checksums.sha256"
    checksum_lines = [f"{sha256_file(path)}  {path.name}\n" for path in trajectory_files]
    _atomic_write_text(checksum_path, "".join(checksum_lines))

    rejection_reasons = Counter(
        str(record.get("failure_reason") or "unknown")
        for record in manifest_records
        if not record.get("accepted")
    )
    schema_version = next(iter(schema_versions))
    control_steps, _ = dimensions_for_schema(schema_version)
    attempt_seconds = [
        float(record["attempt_seconds"])
        for record in manifest_records
        if record.get("attempt_seconds") is not None
    ]
    summary = {
        "schema_version": schema_version,
        "task_id": TASK_ID,
        "robot_id": ROBOT_ID,
        "file_count": len(trajectory_files),
        "transition_count": len(trajectory_files) * control_steps,
        "seeds": sorted(seeds),
        "attempt_count": len(manifest_records),
        "accepted_count": len(accepted_records),
        "rejected_count": len(manifest_records) - len(accepted_records),
        "acceptance_rate": len(accepted_records) / len(manifest_records),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "effective_config_sha256": next(iter(config_hashes)),
        "checksum_index": checksum_path.name,
        "checksum_index_sha256": sha256_file(checksum_path),
        "aggregate_manifest": aggregate_manifest_path.name,
        "aggregate_manifest_sha256": sha256_file(aggregate_manifest_path),
        "action_clip_fraction_mean": float(np.mean(clipping)),
        "action_clip_fraction_max": float(np.max(clipping)),
        "mpx_torque_saturation_fraction_mean": float(np.mean(mpx_saturation)),
        "torque_saturation_fraction_mean": float(np.mean(torque_saturation)),
        "trajectory_bytes": int(sum(path.stat().st_size for path in trajectory_files)),
        "attempt_seconds_sum": float(np.sum(attempt_seconds)) if attempt_seconds else None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if schema_version in (SCHEMA_V3_VERSION, SCHEMA_VERSION):
        def distribution(values: list[float]) -> dict[str, float]:
            array = np.asarray(values, dtype=np.float64)
            return {
                "min": float(np.min(array)),
                "p05": float(np.quantile(array, 0.05)),
                "median": float(np.median(array)),
                "p95": float(np.quantile(array, 0.95)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
            }

        if schema_version == SCHEMA_V3_VERSION:
            summary["reward_component_distributions"] = {
                "cumulative_roll_tracking": distribution(cumulative_roll_tracking),
                "cumulative_rate_tracking": distribution(cumulative_rate_tracking),
                "cumulative_action_change": distribution(cumulative_action_change),
                "cumulative_positive_dense_reward": distribution(cumulative_positive_dense),
            }
        else:
            summary["reward_component_distributions"] = {
                "cumulative_roll_tracking": distribution(cumulative_roll_tracking),
                "cumulative_signed_progress": distribution(cumulative_signed_progress),
                "cumulative_standing_score": distribution(cumulative_standing_score),
                **{
                    f"cumulative_{name}": distribution(values)
                    for name, values in cumulative_standing_subscores.items()
                },
            }
        summary["terminal_metric_distributions"] = {
            "roll_progress": distribution(final_progresses),
            "base_height": distribution(final_heights),
            "body_up_tilt": distribution(final_tilts),
        }
        if schema_version == SCHEMA_V3_VERSION:
            median_action_change = abs(float(np.median(cumulative_action_change)))
            median_positive_dense = float(np.median(cumulative_positive_dense))
            summary["action_change_sanity_ratio"] = (
                median_action_change / median_positive_dense
                if median_positive_dense > 0.0
                else None
            )
        else:
            summary["terminal_metric_distributions"].update({
                "base_linear_speed": distribution(final_linear_speeds),
                "base_angular_speed": distribution(final_angular_speeds),
                "joint_velocity_norm": distribution(final_joint_speeds),
                "final_hold_streak": distribution(final_hold_streaks),
            })
    summary_path = directory / "dataset_summary.json"
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def build_demonstration_report(directory: Path | str) -> dict[str, Any]:
    """Write the complete review packet for the 10-trajectory v4 MPC gate."""
    directory = Path(directory)
    repo_root = Path(__file__).resolve().parents[2]
    report_path = directory / "demonstration_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite demonstration report: {report_path}")
    trajectory_paths = sorted(directory.glob("go2_barrel_roll_v4_*.npz"))
    if len(trajectory_paths) != 10:
        raise BarrelRollValidationError(
            f"demonstration gate requires exactly 10 schema-v4 files, got {len(trajectory_paths)}"
        )
    validation_reports = [
        validate_barrel_roll_file(path) for path in trajectory_paths
    ]
    invalid = [report for report in validation_reports if not report.valid]
    if invalid:
        raise BarrelRollValidationError(
            "; ".join(
                f"{report.path.name}: {', '.join(report.errors)}"
                for report in invalid
            )
        )

    manifest_path = directory / "generation_manifest.jsonl"
    manifest_records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted_records = {
        record["trajectory_file"]: record
        for record in manifest_records
        if record.get("accepted")
    }
    if set(accepted_records) != {path.name for path in trajectory_paths}:
        raise BarrelRollValidationError(
            "demonstration manifest does not exactly match accepted archives"
        )

    trajectories = []
    for trajectory_path in trajectory_paths:
        record = accepted_records[trajectory_path.name]
        seed = int(record["seed"])
        video_path = directory / "videos" / f"mpc_seed_{seed:07d}.mp4"
        if not video_path.is_file():
            raise BarrelRollValidationError(
                f"missing demonstration video for seed {seed}: {video_path}"
            )
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            raise BarrelRollValidationError(
                f"expected one video stream for {video_path}, got {len(streams)}"
            )
        video = streams[0]
        technical = {
            "codec": video.get("codec_name"),
            "width": int(video["width"]),
            "height": int(video["height"]),
            "frame_rate": video.get("r_frame_rate"),
            "duration_seconds": float(video["duration"]),
            "frame_count": int(video["nb_read_frames"]),
        }
        if (
            technical["frame_count"] != CONTROL_STEPS
            or technical["frame_rate"] != "50/1"
            or not np.isclose(
                technical["duration_seconds"], EPISODE_HORIZON, atol=1e-9, rtol=0.0
            )
        ):
            raise BarrelRollValidationError(
                f"video timing mismatch for seed {seed}: {technical}"
            )

        with np.load(trajectory_path, allow_pickle=False) as data:
            action_clip_by_step = np.mean(data["action_clipped"], axis=1)
            mpx_saturation_by_step = np.mean(
                data["mpx_saturation_by_actuator"].reshape(
                    12, CONTROL_STEPS, 4
                ),
                axis=(0, 2),
            )
            applied_saturation_by_step = np.mean(
                data["applied_saturation_by_actuator"].reshape(
                    12, CONTROL_STEPS, 4
                ),
                axis=(0, 2),
            )
            nonfoot_by_step = data["nonfoot_contact"].reshape(CONTROL_STEPS, 4)
            progress_by_step = data["measured_roll_physics"][4::4]
            reward_names = (
                "roll_tracking",
                "signed_progress",
                "standing_score",
                "terminal_outcome",
                "foot_score",
                "height_score",
                "tilt_score",
                "linear_speed_score",
                "angular_speed_score",
                "joint_speed_score",
            )
            condition_names = (
                "rotation",
                "foot_support",
                "height",
                "tilt",
                "base_linear_speed",
                "base_angular_speed",
                "joint_speed",
            )
            endpoint_rows = []
            for index in range(CONTROL_STEPS):
                endpoint_rows.append({
                    "control_step": index + 1,
                    "time_seconds": float(data["control_time"][index]),
                    "roll_progress": float(progress_by_step[index]),
                    "raw_foot_contacts": data["raw_foot_contacts_ctrl"][
                        index + 1
                    ].tolist(),
                    "filtered_foot_contacts": data[
                        "filtered_foot_contacts_ctrl"
                    ][index].tolist(),
                    "nonfoot_contacts_by_physics_substep": [
                        str(value) for value in nonfoot_by_step[index]
                    ],
                    "hold_conditions": {
                        name: bool(data[f"hold_{name}_valid"][index])
                        for name in condition_names
                    },
                    "final_hold_streak": int(data["final_hold_streak"][index]),
                    "base_height": float(data["base_height_ctrl"][index]),
                    "body_up_tilt": float(data["body_up_tilt_ctrl"][index]),
                    "base_linear_speed": float(
                        data["base_linear_speed_ctrl"][index]
                    ),
                    "base_angular_speed": float(
                        data["base_angular_speed_ctrl"][index]
                    ),
                    "joint_velocity_norm": float(
                        data["joint_velocity_norm_ctrl"][index]
                    ),
                    "reward_components": {
                        name: float(data[f"reward_{name}"][index])
                        for name in reward_names
                    },
                    "action_clip_fraction": float(action_clip_by_step[index]),
                    "mpx_torque_saturation_fraction": float(
                        mpx_saturation_by_step[index]
                    ),
                    "applied_torque_saturation_fraction": float(
                        applied_saturation_by_step[index]
                    ),
                })
            source_keys = (
                "root_commit",
                "mpx_commit",
                "solver_commit",
                "gym_quadruped_commit",
                "mpx_xml_sha256",
                "rollout_xml_sha256",
                "generator_source_sha256",
                "effective_config_sha256",
                "generator_command",
                "generator_config_json",
                "runtime_versions_json",
            )
            trajectories.append({
                "seed": seed,
                "sampled_spread": float(data["sampled_spread"]),
                "trajectory_file": trajectory_path.name,
                "trajectory_sha256": sha256_file(trajectory_path),
                "video_file": str(video_path.relative_to(directory)),
                "video_sha256": sha256_file(video_path),
                "video_technical": technical,
                "nonfoot_contact_count": int(data["nonfoot_contact_count"]),
                "action_clip_fraction": float(data["action_clip_fraction"]),
                "mpx_torque_saturation_fraction": float(
                    data["mpx_torque_saturation_fraction"]
                ),
                "applied_torque_saturation_fraction": float(
                    data["torque_saturation_fraction"]
                ),
                "source_identities": {
                    key: np.asarray(data[key]).item() for key in source_keys
                },
                "control_endpoints": endpoint_rows,
            })

    summary_path = directory / "dataset_summary.json"
    checksum_path = directory / "checksums.sha256"
    report = {
        "gate": "schema_v4_mpc_demonstration_approval",
        "status": "awaiting_user_visual_approval",
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "attempt_count": len(manifest_records),
        "accepted_count": len(trajectory_paths),
        "rejected_attempts": [
            record for record in manifest_records if not record.get("accepted")
        ],
        "accepted_seeds": [item["seed"] for item in trajectories],
        "dataset_summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "dataset_summary_sha256": sha256_file(summary_path),
        "checksum_index_sha256": sha256_file(checksum_path),
        "manifest_sha256": sha256_file(manifest_path),
        "retained_checksum_indexes": {
            "schema_v2": sha256_file(
                repo_root / "data/go2_barrel_roll/v2/checksums.sha256"
            ),
            "schema_v3": sha256_file(
                repo_root / "data/go2_barrel_roll/v3/checksums.sha256"
            ),
        },
        "trajectories": trajectories,
    }
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return report


def validate_v3_dynamics_regression(
    v3_path: Path | str,
    retained_v2_path: Path | str,
) -> dict[str, float]:
    """Prove that schema v3 preserves the retained first 1.40 s maneuver."""
    return _validate_extended_dynamics_regression(
        v3_path,
        retained_v2_path,
        expected_schema=SCHEMA_V3_VERSION,
    )


def validate_v4_dynamics_regression(
    v4_path: Path | str,
    retained_v2_path: Path | str,
) -> dict[str, float]:
    """Prove that schema v4 preserves the retained first 1.40 s maneuver."""
    return _validate_extended_dynamics_regression(
        v4_path,
        retained_v2_path,
        expected_schema=SCHEMA_VERSION,
    )


def _validate_extended_dynamics_regression(
    candidate_path: Path | str,
    retained_v2_path: Path | str,
    *,
    expected_schema: int,
) -> dict[str, float]:
    """Compare one long-horizon schema against its retained schema-v2 prefix."""
    candidate_path = Path(candidate_path)
    retained_v2_path = Path(retained_v2_path)
    reports = (
        validate_barrel_roll_file(candidate_path),
        validate_barrel_roll_file(retained_v2_path),
    )
    invalid = [report for report in reports if not report.valid]
    if invalid:
        raise BarrelRollValidationError(
            "; ".join(
                f"{report.path}: {', '.join(report.errors)}" for report in invalid
            )
        )
    with np.load(candidate_path, allow_pickle=False) as candidate, np.load(
        retained_v2_path, allow_pickle=False
    ) as v2:
        if int(candidate["schema_version"]) != expected_schema:
            raise BarrelRollValidationError(
                f"regression candidate is not schema v{expected_schema}"
            )
        if int(v2["schema_version"]) != SCHEMA_V2_VERSION:
            raise BarrelRollValidationError("retained regression archive is not schema v2")
        if int(candidate["rollout_seed"]) != int(v2["rollout_seed"]):
            raise BarrelRollValidationError("regression archives use different rollout seeds")
        comparisons = {
            "qpos": (candidate["qpos"][:, : LEGACY_PHYSICS_STEPS + 1], v2["qpos"]),
            "qvel": (candidate["qvel"][:, : LEGACY_PHYSICS_STEPS + 1], v2["qvel"]),
            "tau_applied": (
                candidate["tau_applied"][:, :LEGACY_PHYSICS_STEPS],
                v2["tau_applied"],
            ),
            "actions": (
                candidate["actions"][:LEGACY_CONTROL_STEPS],
                v2["actions"],
            ),
        }
        maximum_absolute_errors = {}
        for name, (actual, expected) in comparisons.items():
            maximum_absolute_errors[name] = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, atol=1.0e-10, rtol=0.0):
                raise BarrelRollValidationError(
                    f"schema-v{expected_schema} first-1.40s {name} differs from "
                    "retained schema v2; "
                    f"max_abs={maximum_absolute_errors[name]:.17g}"
                )
    return maximum_absolute_errors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--demonstration-report", action="store_true")
    parser.add_argument("--compare-retained-v2", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        if path.is_dir():
            files = sorted(path.glob("go2_barrel_roll_v*_*.npz"))
        else:
            files = [path]
        for file_path in files:
            report = validate_barrel_roll_file(file_path)
            status = "VALID" if report.valid else "INVALID"
            print(
                f"{status} {file_path} "
                f"action_clip={report.metrics.get('action_clip_fraction', float('nan')):.6f} "
                f"mpx_sat={report.metrics.get('mpx_torque_saturation_fraction', float('nan')):.6f} "
                f"applied_sat={report.metrics.get('torque_saturation_fraction', float('nan')):.6f}"
            )
            for error in report.errors:
                print(f"  {error}")
            failed |= not report.valid
        if args.aggregate and path.is_dir() and not failed:
            summary = aggregate_barrel_roll_dataset(path)
            print(canonical_json(summary))
        if args.demonstration_report and path.is_dir() and not failed:
            report = build_demonstration_report(path)
            print(
                "DEMONSTRATION_REPORT "
                + canonical_json({
                    "accepted_count": report["accepted_count"],
                    "accepted_seeds": report["accepted_seeds"],
                    "status": report["status"],
                })
            )
    if args.compare_retained_v2 is not None:
        if len(args.paths) != 1 or args.paths[0].is_dir():
            parser.error(
                "--compare-retained-v2 requires exactly one schema-v3/v4 file path"
            )
        if not failed:
            with np.load(args.paths[0], allow_pickle=False) as candidate:
                candidate_schema = int(candidate["schema_version"])
            validator = {
                SCHEMA_V3_VERSION: validate_v3_dynamics_regression,
                SCHEMA_VERSION: validate_v4_dynamics_regression,
            }.get(candidate_schema)
            if validator is None:
                parser.error("--compare-retained-v2 candidate must be schema v3 or v4")
            regression = validator(args.paths[0], args.compare_retained_v2)
            print("DYNAMICS_REGRESSION " + canonical_json(regression))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
