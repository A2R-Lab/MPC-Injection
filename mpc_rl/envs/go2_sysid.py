"""Shared Go2 system-identification utilities.

This module stores the authoritative per-joint dynamics identified in
``deploy/sys_id/report.html`` and exposes helpers to apply them to any MuJoCo
Go2 model at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np


GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS: dict[str, dict[str, float]] = {
    "FL_calf_joint": {
        "armature": 0.0424,
        "damping": 0.3302,
        "frictionloss": 0.7393,
    },
    "FL_hip_joint": {
        "armature": 0.0114,
        "damping": 0.1791,
        "frictionloss": 0.1788,
    },
    "FL_thigh_joint": {
        "armature": 0.0168,
        "damping": 0.1223,
        "frictionloss": 0.2243,
    },
    "FR_calf_joint": {
        "armature": 0.0429,
        "damping": 0.3372,
        "frictionloss": 0.5988,
    },
    "FR_hip_joint": {
        "armature": 0.0120,
        "damping": 0.1755,
        "frictionloss": 0.2014,
    },
    "FR_thigh_joint": {
        "armature": 0.0173,
        "damping": 0.1377,
        "frictionloss": 0.1609,
    },
    "RL_calf_joint": {
        "armature": 0.0438,
        "damping": 0.3472,
        "frictionloss": 0.6733,
    },
    "RL_hip_joint": {
        "armature": 0.0139,
        "damping": 0.1400,
        "frictionloss": 0.2245,
    },
    "RL_thigh_joint": {
        "armature": 0.0163,
        "damping": 0.1450,
        "frictionloss": 0.2036,
    },
    "RR_calf_joint": {
        "armature": 0.0441,
        "damping": 0.3226,
        "frictionloss": 0.7133,
    },
    "RR_hip_joint": {
        "armature": 0.0094,
        "damping": 0.1578,
        "frictionloss": 0.2662,
    },
    "RR_thigh_joint": {
        "armature": 0.0171,
        "damping": 0.1469,
        "frictionloss": 0.1628,
    },
}

_GO2_SYSID_REPORT_PATTERN = re.compile(
    r'<tr>\s*'
    r'<td class="pt_param" >(.*?)</td>\s*'
    r'<td class="pt_nominal" style="color: var\(--text-muted\);">(.*?)</td>\s*'
    r'<td class="pt_nominal">(.*?)</td>\s*'
    r'<td class="pt_[^"]+">(.*?)</td>',
    re.S,
)
_GO2_SYSID_FIELDS = ("armature", "frictionloss", "damping")


def get_go2_sysid_report_path() -> Path:
    """Return the canonical local sysID report path."""
    return Path(__file__).resolve().parents[2] / "deploy" / "sys_id" / "report.html"


def parse_go2_sysid_report(
    report_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Parse the authoritative Go2 joint-dynamics table from ``report.html``."""
    path = Path(report_path) if report_path is not None else get_go2_sysid_report_path()
    rows = _GO2_SYSID_REPORT_PATTERN.findall(path.read_text(encoding="utf-8"))
    joint_dynamics: dict[str, dict[str, float]] = {}

    for param_name, _initial, _nominal, identified in rows:
        joint_name, field_name = param_name.rsplit("_", 1)
        if field_name not in _GO2_SYSID_FIELDS:
            continue
        joint_dynamics.setdefault(joint_name, {})[field_name] = float(identified)

    expected_joint_count = len(GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS)
    if len(joint_dynamics) != expected_joint_count:
        raise ValueError(
            f"Expected {expected_joint_count} Go2 joints in {path}, found {len(joint_dynamics)}"
        )

    expected_param_count = expected_joint_count * len(_GO2_SYSID_FIELDS)
    actual_param_count = sum(len(fields) for fields in joint_dynamics.values())
    if actual_param_count != expected_param_count:
        raise ValueError(
            f"Expected {expected_param_count} identified Go2 parameters in {path}, "
            f"found {actual_param_count}"
        )

    for joint_name, expected_fields in GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS.items():
        missing_fields = sorted(set(expected_fields) - set(joint_dynamics.get(joint_name, {})))
        if missing_fields:
            raise ValueError(
                f"Missing identified fields for joint {joint_name!r} in {path}: "
                f"{', '.join(missing_fields)}"
            )

    return joint_dynamics


def looks_like_go2_model(mj_model: mujoco.MjModel) -> bool:
    """Return True when the MuJoCo model exposes the expected Go2 leg joints."""
    for joint_name in GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS:
        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) < 0:
            return False
    return True


def apply_go2_sysid_joint_dynamics(
    mj_model: mujoco.MjModel,
    *,
    joint_dynamics: dict[str, dict[str, float]] | None = None,
) -> dict[str, int]:
    """Write the identified Go2 joint dynamics into a MuJoCo model in-place."""
    params = (
        GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS if joint_dynamics is None else joint_dynamics
    )
    applied_dof_indices: dict[str, int] = {}

    for joint_name, dynamics in params.items():
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in MuJoCo model")

        dof_index = int(mj_model.jnt_dofadr[joint_id])
        if dof_index < 0:
            raise ValueError(f"Joint {joint_name!r} does not map to a valid MuJoCo DoF")

        mj_model.dof_armature[dof_index] = dynamics["armature"]
        mj_model.dof_frictionloss[dof_index] = dynamics["frictionloss"]
        mj_model.dof_damping[dof_index] = dynamics["damping"]
        applied_dof_indices[joint_name] = dof_index

    return applied_dof_indices


def get_go2_sysid_joint_dynamics_mismatches(
    mj_model: mujoco.MjModel,
    *,
    joint_dynamics: dict[str, dict[str, float]] | None = None,
    atol: float = 1e-9,
) -> list[str]:
    """Return human-readable mismatches between a model and the Go2 sysID table."""
    params = (
        GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS if joint_dynamics is None else joint_dynamics
    )
    mismatches: list[str] = []

    for joint_name, expected in params.items():
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            mismatches.append(f"missing joint {joint_name!r}")
            continue

        dof_index = int(mj_model.jnt_dofadr[joint_id])
        actual = {
            "armature": float(mj_model.dof_armature[dof_index]),
            "frictionloss": float(mj_model.dof_frictionloss[dof_index]),
            "damping": float(mj_model.dof_damping[dof_index]),
        }
        for field_name, expected_value in expected.items():
            actual_value = actual[field_name]
            if not np.isclose(actual_value, expected_value, atol=atol, rtol=0.0):
                mismatches.append(
                    f"{joint_name}.{field_name}: got {actual_value:.6f}, "
                    f"expected {expected_value:.6f}"
                )

    return mismatches


def assert_go2_sysid_joint_dynamics(
    mj_model: mujoco.MjModel,
    *,
    joint_dynamics: dict[str, dict[str, float]] | None = None,
    atol: float = 1e-9,
) -> None:
    """Raise if a MuJoCo model does not match the expected Go2 sysID dynamics."""
    mismatches = get_go2_sysid_joint_dynamics_mismatches(
        mj_model,
        joint_dynamics=joint_dynamics,
        atol=atol,
    )
    if mismatches:
        details = "; ".join(mismatches[:6])
        if len(mismatches) > 6:
            details += f"; ... ({len(mismatches)} mismatches total)"
        raise AssertionError(
            "Go2 sysID joint dynamics were not applied to this MuJoCo model: "
            f"{details}"
        )
