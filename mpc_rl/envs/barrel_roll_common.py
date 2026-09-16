"""Shared, dependency-free semantics for the fixed Go2 barrel-roll task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

TASK_ID = "go2_barrel_roll"
LEGACY_SCHEMA_VERSION = 1
SCHEMA_V2_VERSION = 2
SCHEMA_V3_VERSION = 3
SCHEMA_VERSION = 4
ACTION_SCALE = 2.0
LEGACY_ACTION_SCALE = 0.5
ROLL_DIRECTION_SIGN = 1.0
SIM_DT = 0.005
CONTROL_DT = 0.02
MANEUVER_HORIZON = 1.40
EPISODE_HORIZON = 2.50
FINAL_HOLD_DURATION = 0.50
FINAL_HOLD_START_TIME = EPISODE_HORIZON - FINAL_HOLD_DURATION
INITIAL_STANCE_DURATION = 0.20
LATERAL_SUPPORT_DURATION = 0.20
FLIGHT_DURATION = 0.35
LANDING_DURATION = 0.10
FINAL_STANCE_DURATION = 0.55
LEGACY_CONTROL_STEPS = int(round(MANEUVER_HORIZON / CONTROL_DT))
CONTROL_STEPS = int(round(EPISODE_HORIZON / CONTROL_DT))
FINAL_HOLD_CONTROL_STEPS = int(round(FINAL_HOLD_DURATION / CONTROL_DT))
ROLL_START_TIME = INITIAL_STANCE_DURATION
ROLL_END_TIME = 0.80
ROLL_TARGET = ROLL_DIRECTION_SIGN * 2.0 * np.pi
SPREAD_RANGE = (0.0, 0.10)
HIP_JOINT_NAMES = ("FL_hip_joint", "RL_hip_joint", "FR_hip_joint", "RR_hip_joint")


@dataclass(frozen=True)
class BarrelRollSuccessConfig:
    progress_tolerance: float = 0.35
    min_base_height: float = 0.20
    max_tilt: float = 0.35
    max_base_linear_speed: float = 0.10
    max_base_angular_speed: float = 0.50
    max_joint_velocity_norm: float = 1.0
    final_hold_control_steps: int = FINAL_HOLD_CONTROL_STEPS


@dataclass(frozen=True)
class BarrelRollRewardConfig:
    tracking_sigma: float = 0.50
    tracking_weight: float = 1.0
    progress_weight: float = 0.25
    progress_clip: float = 3.0
    standing_start_time: float = MANEUVER_HORIZON
    standing_height_target: float = 0.27
    standing_height_scale: float = 0.07
    standing_tilt_scale: float = 0.35
    standing_linear_speed_scale: float = 0.10
    standing_angular_speed_scale: float = 0.50
    standing_joint_speed_scale: float = 1.0
    success_bonus: float = 25.0
    failure_penalty: float = -25.0


@dataclass(frozen=True)
class SchemaV3BarrelRollSuccessConfig:
    min_roll_progress: float = 1.75 * np.pi
    max_roll_progress: float = 2.50 * np.pi
    min_base_height: float = 0.16
    max_tilt: float = np.pi / 3.0


@dataclass(frozen=True)
class SchemaV3BarrelRollRewardConfig:
    tracking_sigma: float = 0.50
    tracking_weight: float = 1.0
    rate_tracking_sigma: float = 1.0
    rate_tracking_weight: float = 0.25
    action_change_weight: float = -0.02
    success_bonus: float = 50.0
    failure_penalty: float = -50.0


@dataclass(frozen=True)
class LegacyBarrelRollSuccessConfig:
    progress_tolerance: float = 0.35
    upright_tolerance: float = 0.35
    min_base_height: float = 0.20
    stable_control_steps: int = 5


@dataclass(frozen=True)
class LegacyBarrelRollRewardConfig:
    tracking_sigma: float = 0.50
    tracking_weight: float = 1.0
    progress_weight: float = 0.25
    success_bonus: float = 10.0
    failure_penalty: float = -10.0
    progress_clip: float = 3.0


SUCCESS_CONFIG = BarrelRollSuccessConfig()
REWARD_CONFIG = BarrelRollRewardConfig()
SCHEMA_V3_SUCCESS_CONFIG = SchemaV3BarrelRollSuccessConfig()
SCHEMA_V3_REWARD_CONFIG = SchemaV3BarrelRollRewardConfig()
LEGACY_SUCCESS_CONFIG = LegacyBarrelRollSuccessConfig()
LEGACY_REWARD_CONFIG = LegacyBarrelRollRewardConfig()

HOLD_CONDITION_PRECEDENCE = (
    "rotation",
    "foot_support",
    "height",
    "tilt",
    "base_linear_speed",
    "base_angular_speed",
    "joint_speed",
)
HOLD_FAILURE_REASONS = {
    "rotation": "rotation_out_of_band",
    "foot_support": "insufficient_foot_support",
    "height": "base_height_below_minimum",
    "tilt": "body_tilt_above_maximum",
    "base_linear_speed": "base_linear_speed_above_maximum",
    "base_angular_speed": "base_angular_speed_above_maximum",
    "joint_speed": "joint_speed_above_maximum",
}


def desired_roll_at_time(time_s: float) -> float:
    """Signed, unwrapped desired roll for the frozen global schedule."""
    phase = np.clip((time_s - ROLL_START_TIME) / (ROLL_END_TIME - ROLL_START_TIME), 0.0, 1.0)
    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
    return float(ROLL_TARGET * smooth_phase)


def desired_roll_rate_at_time(time_s: float) -> float:
    """Analytic derivative of :func:`desired_roll_at_time`."""
    duration = ROLL_END_TIME - ROLL_START_TIME
    phase = (time_s - ROLL_START_TIME) / duration
    if phase <= 0.0 or phase >= 1.0:
        return 0.0
    return float(ROLL_TARGET * 6.0 * phase * (1.0 - phase) / duration)


def maneuver_phase_at_time(time_s: float) -> float:
    """Schema-v4 actor phase, saturated when the desired roll finishes."""
    return float(np.clip(time_s / ROLL_END_TIME, 0.0, 1.0))


def schema_v3_maneuver_phase_at_time(time_s: float) -> float:
    """Frozen schema-v3 actor phase over the complete 2.50 second episode."""
    return float(np.clip(time_s / EPISODE_HORIZON, 0.0, 1.0))


def legacy_maneuver_phase_at_time(time_s: float) -> float:
    """Frozen schema-v1/v2 actor phase, saturated at the roll end time."""
    return float(np.clip(time_s / ROLL_END_TIME, 0.0, 1.0))


def _normalise_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(quat).all() or norm <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    return quat / norm


def continuous_quaternion(previous_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    """Choose the equivalent quaternion nearest the previous sample."""
    previous = _normalise_wxyz(previous_wxyz)
    current = _normalise_wxyz(current_wxyz)
    return -current if np.dot(previous, current) < 0.0 else current


def relative_twist_roll(initial_wxyz: np.ndarray, current_wxyz: np.ndarray) -> float:
    """Return the wrapped relative twist angle about the Go2 longitudinal x-axis."""
    initial_xyzw = np.roll(_normalise_wxyz(initial_wxyz), -1)
    current_xyzw = np.roll(_normalise_wxyz(current_wxyz), -1)
    rel = Rotation.from_quat(initial_xyzw).inv() * Rotation.from_quat(current_xyzw)
    x, _, _, w = rel.as_quat()
    magnitude = np.hypot(w, x)
    if magnitude <= np.finfo(np.float64).eps:
        return 0.0
    return float(2.0 * np.arctan2(x / magnitude, w / magnitude))


def unwrap_roll_increment(previous_wrapped: float, current_wrapped: float) -> float:
    return float((current_wrapped - previous_wrapped + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class RollProgressTracker:
    initial_quat_wxyz: np.ndarray
    previous_quat_wxyz: np.ndarray
    previous_wrapped_roll: float = 0.0
    progress: float = 0.0

    @classmethod
    def from_reset_quaternion(cls, quat_wxyz: np.ndarray) -> "RollProgressTracker":
        quat = _normalise_wxyz(quat_wxyz)
        return cls(quat.copy(), quat.copy())

    def update(self, quat_wxyz: np.ndarray) -> float:
        continuous = continuous_quaternion(self.previous_quat_wxyz, quat_wxyz)
        wrapped = relative_twist_roll(self.initial_quat_wxyz, continuous)
        self.progress += unwrap_roll_increment(self.previous_wrapped_roll, wrapped)
        self.previous_quat_wxyz = continuous
        self.previous_wrapped_roll = wrapped
        return float(self.progress)


def sample_symmetric_hip_spread(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    rng: np.random.Generator,
    *,
    spread: float | None = None,
) -> tuple[np.ndarray, float]:
    """Apply the locked symmetric hip spread using MuJoCo joint addresses."""
    if spread is None:
        spread = float(rng.uniform(*SPREAD_RANGE))
    else:
        spread = float(spread)
        if not np.isfinite(spread) or not SPREAD_RANGE[0] <= spread <= SPREAD_RANGE[1]:
            raise ValueError(f"spread must be within {SPREAD_RANGE}")
    sampled = np.asarray(qpos, dtype=np.float64).copy()
    signs = (1.0, 1.0, -1.0, -1.0)
    for name, sign in zip(HIP_JOINT_NAMES, signs, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"required Go2 hip joint is missing: {name}")
        sampled[int(model.jnt_qposadr[joint_id])] += sign * spread
    return sampled, spread


def assert_valid_initial_state(env: Any) -> None:
    """Reject invalid sampled starts before either MPC solve or RL rollout."""
    data, model = env.mjData, env.mjModel
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("sampled initial state is non-finite")
    for joint_id in range(1, model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        address = int(model.jnt_qposadr[joint_id])
        low, high = model.jnt_range[joint_id]
        if data.qpos[address] < low or data.qpos[address] > high:
            raise ValueError(f"sampled initial state violates joint limit {joint_id}")
    if np.any(env._get_foot_positions()[:, 2] < -1.0e-5):
        raise ValueError("sampled initial state has foot penetration")
    if find_nonfoot_ground_contact(env) is not None:
        raise ValueError("sampled initial state has non-foot ground contact")


def find_nonfoot_ground_contact(env: Any) -> str | None:
    """Return a compact violation string when a robot non-foot geom hits ground."""
    for index in range(env.mjData.ncon):
        contact = env.mjData.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(env.mjModel.geom_bodyid[geom1])
        body2 = int(env.mjModel.geom_bodyid[geom2])
        if (body1 == 0) == (body2 == 0):
            continue
        robot_geom = geom2 if body1 == 0 else geom1
        if robot_geom in env._foot_geom_id_set:
            continue
        name = mujoco.mj_id2name(env.mjModel, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
        return name or f"geom_{robot_geom}"
    return None


def body_up_tilt_from_quaternion(quat_wxyz: np.ndarray) -> float:
    """Return the angle between body-up and world-up in radians."""
    rotation = Rotation.from_quat(np.roll(_normalise_wxyz(quat_wxyz), -1))
    body_up = rotation.apply(np.array([0.0, 0.0, 1.0]))
    return float(np.arccos(np.clip(body_up[2], -1.0, 1.0)))


def schema_v3_terminal_failure_reason(
    roll_progress: float,
    base_height: float,
    body_up_tilt: float,
) -> str | None:
    """Classify a finite schema-v3 terminal state with locked precedence."""
    directed_progress = ROLL_DIRECTION_SIGN * float(roll_progress)
    if not (
        SCHEMA_V3_SUCCESS_CONFIG.min_roll_progress
        <= directed_progress
        <= SCHEMA_V3_SUCCESS_CONFIG.max_roll_progress
    ):
        return "rotation_out_of_band"
    if float(base_height) < SCHEMA_V3_SUCCESS_CONFIG.min_base_height:
        return "fallen_low_height"
    if float(body_up_tilt) > SCHEMA_V3_SUCCESS_CONFIG.max_tilt:
        return "fallen_tilt"
    return None


def final_hold_conditions(
    *,
    filtered_foot_contacts: np.ndarray,
    roll_progress: float,
    base_height: float,
    body_up_tilt: float,
    base_linear_speed: float,
    base_angular_speed: float,
    joint_velocity_norm: float,
) -> dict[str, bool]:
    """Evaluate every schema-v4 hold condition at one control endpoint."""
    directed_progress = ROLL_DIRECTION_SIGN * float(roll_progress)
    return {
        "rotation": bool(
            ROLL_TARGET - SUCCESS_CONFIG.progress_tolerance
            <= directed_progress
            <= ROLL_TARGET + SUCCESS_CONFIG.progress_tolerance
        ),
        "foot_support": bool(np.all(np.asarray(filtered_foot_contacts, dtype=bool))),
        "height": bool(float(base_height) >= SUCCESS_CONFIG.min_base_height),
        "tilt": bool(float(body_up_tilt) <= SUCCESS_CONFIG.max_tilt),
        "base_linear_speed": bool(
            float(base_linear_speed) < SUCCESS_CONFIG.max_base_linear_speed
        ),
        "base_angular_speed": bool(
            float(base_angular_speed) < SUCCESS_CONFIG.max_base_angular_speed
        ),
        "joint_speed": bool(
            float(joint_velocity_norm) < SUCCESS_CONFIG.max_joint_velocity_norm
        ),
    }


def terminal_failure_reason(condition_failures: Mapping[str, int | bool]) -> str | None:
    """Return the first failed v4 hold category using deterministic precedence."""
    for name in HOLD_CONDITION_PRECEDENCE:
        if bool(condition_failures.get(name, False)):
            return HOLD_FAILURE_REASONS[name]
    return None


def standing_subscores(
    *,
    filtered_foot_contacts: np.ndarray,
    base_height: float,
    body_up_tilt: float,
    base_linear_speed: float,
    base_angular_speed: float,
    joint_velocity_norm: float,
) -> dict[str, float]:
    """Return the six unweighted schema-v4 standing score components."""
    return {
        "foot_score": float(np.mean(np.asarray(filtered_foot_contacts, dtype=bool))),
        "height_score": float(
            np.exp(
                -(
                    (float(base_height) - REWARD_CONFIG.standing_height_target)
                    / REWARD_CONFIG.standing_height_scale
                )
                ** 2
            )
        ),
        "tilt_score": float(
            np.exp(-(float(body_up_tilt) / REWARD_CONFIG.standing_tilt_scale) ** 2)
        ),
        "linear_speed_score": float(
            np.exp(
                -(
                    float(base_linear_speed)
                    / REWARD_CONFIG.standing_linear_speed_scale
                )
                ** 2
            )
        ),
        "angular_speed_score": float(
            np.exp(
                -(
                    float(base_angular_speed)
                    / REWARD_CONFIG.standing_angular_speed_scale
                )
                ** 2
            )
        ),
        "joint_speed_score": float(
            np.exp(
                -(
                    float(joint_velocity_norm)
                    / REWARD_CONFIG.standing_joint_speed_scale
                )
                ** 2
            )
        ),
    }


def reward_config_dict(schema_version: int = SCHEMA_VERSION) -> dict[str, float]:
    if schema_version in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION):
        return asdict(LEGACY_REWARD_CONFIG)
    if schema_version == SCHEMA_V3_VERSION:
        return asdict(SCHEMA_V3_REWARD_CONFIG)
    if schema_version == SCHEMA_VERSION:
        return asdict(REWARD_CONFIG)
    raise ValueError(f"unsupported barrel-roll schema version {schema_version}")


def success_config_dict(schema_version: int = SCHEMA_VERSION) -> dict[str, float | int]:
    if schema_version in (LEGACY_SCHEMA_VERSION, SCHEMA_V2_VERSION):
        return asdict(LEGACY_SUCCESS_CONFIG)
    if schema_version == SCHEMA_V3_VERSION:
        return asdict(SCHEMA_V3_SUCCESS_CONFIG)
    if schema_version == SCHEMA_VERSION:
        return asdict(SUCCESS_CONFIG)
    raise ValueError(f"unsupported barrel-roll schema version {schema_version}")
