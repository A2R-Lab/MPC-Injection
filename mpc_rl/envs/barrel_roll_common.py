"""Shared, dependency-free semantics for the fixed Go2 barrel-roll task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

TASK_ID = "go2_barrel_roll"
LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
ACTION_SCALE = 2.0
LEGACY_ACTION_SCALE = 0.5
ROLL_DIRECTION_SIGN = 1.0
SIM_DT = 0.005
CONTROL_DT = 0.02
MANEUVER_HORIZON = 1.40
INITIAL_STANCE_DURATION = 0.20
LATERAL_SUPPORT_DURATION = 0.20
FLIGHT_DURATION = 0.35
LANDING_DURATION = 0.10
FINAL_STANCE_DURATION = 0.55
CONTROL_STEPS = int(round(MANEUVER_HORIZON / CONTROL_DT))
ROLL_START_TIME = INITIAL_STANCE_DURATION
ROLL_END_TIME = 0.80
ROLL_TARGET = ROLL_DIRECTION_SIGN * 2.0 * np.pi
SPREAD_RANGE = (0.0, 0.10)
HIP_JOINT_NAMES = ("FL_hip_joint", "RL_hip_joint", "FR_hip_joint", "RR_hip_joint")


@dataclass(frozen=True)
class BarrelRollSuccessConfig:
    progress_tolerance: float = 0.35
    upright_tolerance: float = 0.35
    min_base_height: float = 0.20
    stable_control_steps: int = 5


@dataclass(frozen=True)
class BarrelRollRewardConfig:
    tracking_sigma: float = 0.50
    tracking_weight: float = 1.0
    progress_weight: float = 0.25
    success_bonus: float = 10.0
    failure_penalty: float = -10.0
    progress_clip: float = 3.0


SUCCESS_CONFIG = BarrelRollSuccessConfig()
REWARD_CONFIG = BarrelRollRewardConfig()


def desired_roll_at_time(time_s: float) -> float:
    """Signed, unwrapped desired roll for the frozen global schedule."""
    phase = np.clip((time_s - ROLL_START_TIME) / (ROLL_END_TIME - ROLL_START_TIME), 0.0, 1.0)
    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
    return float(ROLL_TARGET * smooth_phase)


def maneuver_phase_at_time(time_s: float) -> float:
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


def reward_config_dict() -> dict[str, float]:
    return asdict(REWARD_CONFIG)


def success_config_dict() -> dict[str, float | int]:
    return asdict(SUCCESS_CONFIG)
