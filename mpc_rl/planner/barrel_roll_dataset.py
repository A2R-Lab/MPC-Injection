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
    FINAL_STANCE_DURATION,
    FLIGHT_DURATION,
    INITIAL_STANCE_DURATION,
    LANDING_DURATION,
    LEGACY_ACTION_SCALE,
    LEGACY_SCHEMA_VERSION,
    LATERAL_SUPPORT_DURATION,
    MANEUVER_HORIZON,
    REWARD_CONFIG,
    ROLL_DIRECTION_SIGN,
    ROLL_END_TIME,
    ROLL_START_TIME,
    SCHEMA_VERSION,
    SIM_DT,
    SPREAD_RANGE,
    SUCCESS_CONFIG,
    TASK_ID,
    RollProgressTracker,
    desired_roll_at_time,
    maneuver_phase_at_time,
    reward_config_dict,
    success_config_dict,
)

ROBOT_ID = "go2"
PHYSICS_STEPS = 280
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

REQUIRED_SHAPES = {
    "policy_obs": (CONTROL_STEPS, 45),
    "next_policy_obs": (CONTROL_STEPS, 45),
    "privileged_obs": (CONTROL_STEPS, 4),
    "next_privileged_obs": (CONTROL_STEPS, 4),
    "actions": (CONTROL_STEPS, 12),
    "rewards": (CONTROL_STEPS,),
    "terminated_ctrl": (CONTROL_STEPS,),
    "truncated_ctrl": (CONTROL_STEPS,),
    "qpos": (19, PHYSICS_STEPS + 1),
    "qvel": (18, PHYSICS_STEPS + 1),
    "tau_applied": (12, PHYSICS_STEPS),
    "tau_mpx": (12, PHYSICS_STEPS),
    "tau_raw": (12, PHYSICS_STEPS),
    "q_des": (12, PHYSICS_STEPS),
    "dq_des": (12, PHYSICS_STEPS),
    "X_updates": (CONTROL_STEPS, MPC_NODES + 1, MPC_STATE_DIM),
    "U_updates": (CONTROL_STEPS, MPC_NODES, MPC_CONTROL_DIM),
    "physics_time": (PHYSICS_STEPS + 1,),
    "control_time": (CONTROL_STEPS,),
    "phase_ctrl": (CONTROL_STEPS,),
    "desired_roll_ctrl": (CONTROL_STEPS,),
    "measured_roll_physics": (PHYSICS_STEPS + 1,),
    "foot_contacts": (PHYSICS_STEPS, 4),
    "nonfoot_contact": (PHYSICS_STEPS,),
    "stable_contact_streak": (CONTROL_STEPS,),
    "classifier_result": (CONTROL_STEPS,),
    "residual_actions_unclipped": (CONTROL_STEPS, 12),
    "action_clipped": (CONTROL_STEPS, 12),
    "mpx_saturation_by_actuator": (12, PHYSICS_STEPS),
    "applied_saturation_by_actuator": (12, PHYSICS_STEPS),
    "solve_seconds": (CONTROL_STEPS,),
    "solve_iterations": (CONTROL_STEPS,),
    "solve_iteration_limit": (CONTROL_STEPS,),
    "solve_objective_norm_sq": (CONTROL_STEPS,),
    "solve_constraint_norm_sq": (CONTROL_STEPS,),
    "solve_finite": (CONTROL_STEPS,),
    "solve_replanned": (CONTROL_STEPS,),
    "solve_phase_index": (CONTROL_STEPS,),
    "solve_elapsed_time": (CONTROL_STEPS,),
    "warm_start_shift": (CONTROL_STEPS,),
    "default_joint_pos": (12,),
    "pd_kp": (12,),
    "pd_kd": (12,),
    "tracking_kp": (12,),
    "tracking_kd": (12,),
    "torque_limits": (12, 2),
}

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


def schedule_dict() -> dict[str, float]:
    return {
        "initial_stance_duration": INITIAL_STANCE_DURATION,
        "lateral_support_duration": LATERAL_SUPPORT_DURATION,
        "flight_duration": FLIGHT_DURATION,
        "landing_duration": LANDING_DURATION,
        "final_stance_duration": FINAL_STANCE_DURATION,
        "roll_start_time": ROLL_START_TIME,
        "roll_end_time": ROLL_END_TIME,
        "maneuver_horizon": MANEUVER_HORIZON,
    }


def action_scale_for_schema(schema_version: int) -> float:
    """Return the immutable action scale for a supported dataset schema."""
    if schema_version == LEGACY_SCHEMA_VERSION:
        return LEGACY_ACTION_SCALE
    if schema_version == SCHEMA_VERSION:
        return ACTION_SCALE
    raise ValueError(f"unsupported barrel-roll schema version {schema_version}")


def expected_effective_config(schema_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    action_scale = action_scale_for_schema(schema_version)
    return {
        "action": {
            "lpf_cutoff_hz": 5.0,
            "scale": action_scale,
            "semantics": ACTION_SEMANTICS,
        },
        "contact_convention": CONTACT_CONVENTION,
        "classifier_contact_sampling": "control_endpoint_raw_after_base_contact_update",
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
        "reward": reward_config_dict(),
        "robot": ROBOT_ID,
        "roll_direction": ROLL_DIRECTION_SIGN,
        "schedule": schedule_dict(),
        "schema_version": schema_version,
        "success": success_config_dict(),
        "task_id": TASK_ID,
        "timing": {
            "control_dt": CONTROL_DT,
            "control_steps": CONTROL_STEPS,
            "decimation": 4,
            "mpc_dt": MPC_DT,
            "physics_steps": PHYSICS_STEPS,
            "sim_dt": SIM_DT,
        },
        "torque_limits": TORQUE_LIMITS.tolist(),
    }


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty(repo: Path) -> bool:
    return bool(_git_value(repo, "status", "--porcelain", "--untracked-files=no"))


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
        "schedule_json": canonical_json(schedule_dict()),
        "success_config_json": canonical_json(success_config_dict()),
        "reward_config_json": canonical_json(reward_config_dict()),
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
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        if validate:
            report = validate_barrel_roll_file(temporary)
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


def _recompute_stability(foot_contacts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    streaks = np.zeros(CONTROL_STEPS, dtype=np.int64)
    filtered = np.zeros((CONTROL_STEPS, 4), dtype=bool)
    streak = 0
    for control_index in range(CONTROL_STEPS):
        current = foot_contacts[(control_index + 1) * 4 - 1]
        # The commissioned environment updates its prior-contact state before
        # calling the task classifier, so the classifier's saved contact state
        # is the raw contact state at this controller boundary.
        filtered[control_index] = current
        streak = streak + 1 if np.all(filtered[control_index]) else 0
        streaks[control_index] = streak
    return streaks, filtered


def _add_mismatch(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_barrel_roll_file(path: Path | str) -> ValidationReport:
    """Strictly validate one accepted versioned direct-transition archive."""
    path = Path(path)
    errors: list[str] = []
    metrics: dict[str, float] = {}
    try:
        with np.load(path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    except Exception as error:
        return ValidationReport(path, (f"archive load failed: {type(error).__name__}: {error}",), metrics)

    missing = sorted((set(REQUIRED_SHAPES) | REQUIRED_SCALARS) - set(data))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
        return ValidationReport(path, tuple(errors), metrics)

    for key, value in data.items():
        if np.asarray(value).dtype.kind not in "biufcU":
            errors.append(f"{key} has forbidden dtype {np.asarray(value).dtype}")
    for key, shape in REQUIRED_SHAPES.items():
        if data[key].shape != shape:
            errors.append(f"{key} shape mismatch: expected {shape}, got {data[key].shape}")
    for key in REQUIRED_SCALARS:
        if np.asarray(data[key]).shape != ():
            errors.append(f"{key} must be scalar, got {np.asarray(data[key]).shape}")
    if errors:
        return ValidationReport(path, tuple(errors), metrics)

    try:
        schema_version = int(_scalar(data, "schema_version"))
        action_scale = action_scale_for_schema(schema_version)
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        return ValidationReport(path, tuple(errors), metrics)

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
        "controller_steps": CONTROL_STEPS,
        "physics_steps": PHYSICS_STEPS,
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
        "schedule_json": schedule_dict(),
        "success_config_json": success_config_dict(),
        "reward_config_json": reward_config_dict(),
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

    physics_time = np.arange(PHYSICS_STEPS + 1, dtype=np.float64) * SIM_DT
    control_time = np.arange(1, CONTROL_STEPS + 1, dtype=np.float64) * CONTROL_DT
    phase_ctrl = np.array([maneuver_phase_at_time(t) for t in control_time])
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
    pre_roll = recomputed_roll[np.arange(CONTROL_STEPS) * 4]
    post_roll = recomputed_roll[np.arange(1, CONTROL_STEPS + 1) * 4]
    if not np.allclose(data["privileged_obs"][:, 3], pre_roll, atol=1e-8, rtol=0.0):
        errors.append("privileged_obs roll progress mismatch")
    if not np.allclose(data["next_privileged_obs"][:, 3], post_roll, atol=1e-8, rtol=0.0):
        errors.append("next_privileged_obs roll progress mismatch")

    pre_time = np.arange(CONTROL_STEPS, dtype=np.float64) * CONTROL_DT
    pre_phase = np.array([maneuver_phase_at_time(t) for t in pre_time])
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

    streaks, filtered_contacts = _recompute_stability(data["foot_contacts"])
    if not np.array_equal(data["stable_contact_streak"], streaks):
        errors.append("stable_contact_streak mismatch")
    if np.any(data["nonfoot_contact"] != ""):
        errors.append("non-foot ground contact recorded")

    terminated_expected = np.zeros(CONTROL_STEPS, dtype=bool)
    terminated_expected[-1] = True
    if not np.array_equal(data["terminated_ctrl"], terminated_expected):
        errors.append("terminated_ctrl mismatch")
    if np.any(data["truncated_ctrl"]):
        errors.append("truncated_ctrl must be false for accepted full-horizon rollouts")

    expected_step = 2.0 * np.pi * CONTROL_DT / (ROLL_END_TIME - ROLL_START_TIME)
    tracking = REWARD_CONFIG.tracking_weight * np.exp(
        -((desired_ctrl - post_roll) / REWARD_CONFIG.tracking_sigma) ** 2
    )
    progress = REWARD_CONFIG.progress_weight * np.clip(
        ROLL_DIRECTION_SIGN * (post_roll - pre_roll) / expected_step,
        -REWARD_CONFIG.progress_clip,
        REWARD_CONFIG.progress_clip,
    )
    expected_rewards = tracking + progress
    expected_rewards[-1] += REWARD_CONFIG.success_bonus
    if not np.allclose(data["rewards"], expected_rewards, atol=1e-8, rtol=0.0):
        errors.append("rewards mismatch with frozen reward recomputation")

    final_rotation = Rotation.from_quat(np.roll(data["qpos"][3:7, -1], -1))
    final_roll, final_pitch, _ = final_rotation.as_euler("xyz")
    final_height = float(data["qpos"][2, -1])
    progress_ok = ROLL_DIRECTION_SIGN * post_roll[-1] >= 2.0 * np.pi - SUCCESS_CONFIG.progress_tolerance
    upright_ok = (
        abs(final_roll) <= SUCCESS_CONFIG.upright_tolerance
        and abs(final_pitch) <= SUCCESS_CONFIG.upright_tolerance
    )
    stable_ok = streaks[-1] >= SUCCESS_CONFIG.stable_control_steps and np.all(filtered_contacts[-1])
    height_ok = final_height >= SUCCESS_CONFIG.min_base_height
    classifier = progress_ok and upright_ok and stable_ok and height_ok
    expected_classifier_values = []
    for index in range(CONTROL_STEPS):
        state_index = (index + 1) * 4
        roll_error, pitch_error, _ = Rotation.from_quat(
            np.roll(data["qpos"][3:7, state_index], -1)
        ).as_euler("xyz")
        expected_classifier_values.append(
            ROLL_DIRECTION_SIGN * post_roll[index]
            >= 2.0 * np.pi - SUCCESS_CONFIG.progress_tolerance
            and streaks[index] >= SUCCESS_CONFIG.stable_control_steps
            and data["qpos"][2, state_index] >= SUCCESS_CONFIG.min_base_height
            and abs(roll_error) <= SUCCESS_CONFIG.upright_tolerance
            and abs(pitch_error) <= SUCCESS_CONFIG.upright_tolerance
        )
    expected_classifier = np.asarray(expected_classifier_values, dtype=bool)
    if not np.array_equal(data["classifier_result"], expected_classifier):
        errors.append("classifier_result mismatch")
    if not progress_ok:
        errors.append("incomplete rotation")
    if not upright_ok or not height_ok or not stable_ok:
        errors.append("unstable landing")
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

    expected_phase_index = np.arange(CONTROL_STEPS, dtype=np.int64) * 2
    expected_elapsed = np.arange(CONTROL_STEPS, dtype=np.float64) * CONTROL_DT
    if not np.array_equal(data["solve_phase_index"], expected_phase_index):
        errors.append("solve_phase_index mismatch")
    if not np.allclose(data["solve_elapsed_time"], expected_elapsed, atol=1e-12, rtol=0.0):
        errors.append("solve_elapsed_time mismatch")
    expected_shift = np.full(CONTROL_STEPS, 2, dtype=np.int64)
    expected_shift[0] = 0
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
    for path, report in zip(trajectory_files, reports, strict=True):
        with np.load(path, allow_pickle=False) as data:
            seeds.append(int(data["rollout_seed"]))
            schema_versions.add(int(data["schema_version"]))
            config_hashes.add(str(data["effective_config_sha256"]))
        clipping.append(report.metrics["action_clip_fraction"])
        mpx_saturation.append(report.metrics["mpx_torque_saturation_fraction"])
        torque_saturation.append(report.metrics["torque_saturation_fraction"])
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
    accepted_names = {record.get("trajectory_file") for record in accepted_records}
    if accepted_names != {path.name for path in trajectory_files}:
        raise BarrelRollValidationError("manifest accepted filenames do not match dataset files")

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
    summary = {
        "schema_version": next(iter(schema_versions)),
        "task_id": TASK_ID,
        "robot_id": ROBOT_ID,
        "file_count": len(trajectory_files),
        "transition_count": len(trajectory_files) * CONTROL_STEPS,
        "seeds": sorted(seeds),
        "attempt_count": len(manifest_records),
        "accepted_count": len(accepted_records),
        "rejected_count": len(manifest_records) - len(accepted_records),
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
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = directory / "dataset_summary.json"
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--aggregate", action="store_true")
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
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
