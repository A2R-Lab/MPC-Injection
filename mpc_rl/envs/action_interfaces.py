"""Versioned quadruped residual-position action interfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


INFERRED_ACTION_DIRECT_TORQUE_MODE = "inferred_action_direct_torque_v1"
ENV_STEP_LPF_INVERSE_MODE = "env_step_lpf_inverse_v1"
MPX_BOUND_ENV_STEP_MODE = "env_step_mpx_target_v1"

DEFAULT_ACTION_INTERFACE_ID = "residual_position_scale_0p5_lpf_5hz_v1"
MPX_BOUND_ACTION_INTERFACE_ID = "mpx_bound_scale_1_no_lpf_v1"
DIRECT_TRANSITION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class QuadrupedActionInterface:
    """Immutable environment and MPX-conversion action contract."""

    interface_id: str
    version: int
    action_dim: int
    action_scale: float
    action_lpf_cutoff_hz: float | None
    required_mpx_conversion_mode: str | None = None
    mpx_target_formula: str | None = None
    generation_advance: str | None = None

    def env_kwargs(self) -> dict[str, float | None]:
        return {
            "action_scale": self.action_scale,
            "action_lpf_cutoff_hz": self.action_lpf_cutoff_hz,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.interface_id,
            "version": self.version,
            "action_dim": self.action_dim,
            "normalized_action_bounds": [-1.0, 1.0],
            "action_scale": self.action_scale,
            "action_lpf_cutoff_hz": self.action_lpf_cutoff_hz,
            "required_mpx_conversion_mode": self.required_mpx_conversion_mode,
            "mpx_target_formula": self.mpx_target_formula,
            "generation_advance": self.generation_advance,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


ACTION_INTERFACES: Mapping[str, QuadrupedActionInterface] = MappingProxyType(
    {
        DEFAULT_ACTION_INTERFACE_ID: QuadrupedActionInterface(
            interface_id=DEFAULT_ACTION_INTERFACE_ID,
            version=1,
            action_dim=12,
            action_scale=0.5,
            action_lpf_cutoff_hz=5.0,
        ),
        MPX_BOUND_ACTION_INTERFACE_ID: QuadrupedActionInterface(
            interface_id=MPX_BOUND_ACTION_INTERFACE_ID,
            version=1,
            action_dim=12,
            action_scale=1.0,
            action_lpf_cutoff_hz=None,
            required_mpx_conversion_mode=MPX_BOUND_ENV_STEP_MODE,
            mpx_target_formula="q_target = q_des + tau_ff / kp_realized",
            generation_advance="env.step(raw_action)",
        ),
    }
)


def resolve_action_interface(
    interface_id: str | None = None,
) -> QuadrupedActionInterface:
    """Resolve a named action interface without changing the legacy default."""
    resolved_id = interface_id or DEFAULT_ACTION_INTERFACE_ID
    try:
        return ACTION_INTERFACES[resolved_id]
    except KeyError as exc:
        choices = ", ".join(ACTION_INTERFACES)
        raise ValueError(
            f"Unknown quadruped action interface {resolved_id!r}; "
            f"expected one of: {choices}"
        ) from exc


def compute_action_lpf_alpha(cutoff_hz: float | None, dt: float) -> np.float64:
    """Return the first-order target-filter coefficient for one control step."""
    if cutoff_hz is None or cutoff_hz <= 0.0:
        return np.float64(1.0)
    if dt <= 0.0:
        raise ValueError(f"LPF timestep must be positive, got {dt}")
    alpha = 1.0 - np.exp(-2.0 * np.pi * float(cutoff_hz) * float(dt))
    return np.float64(np.clip(alpha, 0.0, 1.0))


def action_interface_metadata(
    interface: QuadrupedActionInterface,
    *,
    action_lpf_alpha: float,
) -> dict[str, Any]:
    """Return pickle-free scalar metadata for a direct-transition file."""
    return {
        "schema_version": DIRECT_TRANSITION_SCHEMA_VERSION,
        "action_interface_id": interface.interface_id,
        "action_interface_version": interface.version,
        "action_interface_json": interface.to_json(),
        "action_dim": interface.action_dim,
        "action_scale": interface.action_scale,
        "action_lpf_cutoff_hz_is_none": interface.action_lpf_cutoff_hz is None,
        "action_lpf_cutoff_hz_value": (
            0.0
            if interface.action_lpf_cutoff_hz is None
            else interface.action_lpf_cutoff_hz
        ),
        "action_lpf_alpha": float(action_lpf_alpha),
    }


def _metadata_scalar(data: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(
            f"quadruped direct-transition metadata {key!r} must be scalar, "
            f"got shape {value.shape}"
        )
    return value.item()


def validate_action_interface_metadata(
    data: Mapping[str, Any],
    *,
    expected_interface_id: str | None = None,
) -> QuadrupedActionInterface:
    """Validate a schema-v2 direct trajectory against its selected interface."""
    required = (
        "schema_version",
        "action_interface_id",
        "action_interface_version",
        "action_interface_json",
        "action_dim",
        "action_scale",
        "action_lpf_cutoff_hz_is_none",
        "action_lpf_cutoff_hz_value",
        "action_lpf_alpha",
        "action_conversion_mode",
        "control_dt",
        "actions",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            "quadruped direct-transition action metadata is incomplete; "
            f"missing {missing}"
        )

    schema_version = int(_metadata_scalar(data, "schema_version"))
    if schema_version != DIRECT_TRANSITION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported quadruped direct-transition schema_version="
            f"{schema_version}; expected {DIRECT_TRANSITION_SCHEMA_VERSION}"
        )

    recorded_id = str(_metadata_scalar(data, "action_interface_id"))
    interface = resolve_action_interface(recorded_id)
    if expected_interface_id is not None and recorded_id != expected_interface_id:
        raise ValueError(
            "quadruped action-interface mismatch: "
            f"dataset={recorded_id!r}, training={expected_interface_id!r}"
        )

    recorded_version = int(_metadata_scalar(data, "action_interface_version"))
    if recorded_version != interface.version:
        raise ValueError(
            f"action interface {recorded_id!r} version mismatch: "
            f"dataset={recorded_version}, expected={interface.version}"
        )

    try:
        descriptor = json.loads(str(_metadata_scalar(data, "action_interface_json")))
    except json.JSONDecodeError as exc:
        raise ValueError("action_interface_json is not valid JSON") from exc
    if descriptor != interface.to_dict():
        raise ValueError(
            f"action interface descriptor for {recorded_id!r} does not match "
            "the repository definition"
        )

    recorded_dim = int(_metadata_scalar(data, "action_dim"))
    actions = np.asarray(data["actions"], dtype=np.float64)
    if recorded_dim != interface.action_dim or actions.ndim != 2 or actions.shape[1] != recorded_dim:
        raise ValueError(
            "quadruped action dimension mismatch: "
            f"metadata={recorded_dim}, actions_shape={actions.shape}, "
            f"interface={interface.action_dim}"
        )
    if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1.0):
        raise ValueError("quadruped direct-transition actions must be finite and in [-1, 1]")

    recorded_scale = float(_metadata_scalar(data, "action_scale"))
    if recorded_scale != interface.action_scale:
        raise ValueError(
            f"action_scale mismatch for {recorded_id!r}: "
            f"dataset={recorded_scale}, expected={interface.action_scale}"
        )

    cutoff_is_none = bool(
        _metadata_scalar(data, "action_lpf_cutoff_hz_is_none")
    )
    expected_cutoff_is_none = interface.action_lpf_cutoff_hz is None
    cutoff_value = float(_metadata_scalar(data, "action_lpf_cutoff_hz_value"))
    expected_cutoff_value = (
        0.0
        if interface.action_lpf_cutoff_hz is None
        else interface.action_lpf_cutoff_hz
    )
    if (
        cutoff_is_none != expected_cutoff_is_none
        or cutoff_value != expected_cutoff_value
    ):
        raise ValueError(
            f"action LPF cutoff mismatch for {recorded_id!r}: "
            f"dataset_is_none={cutoff_is_none}, dataset_value={cutoff_value}, "
            f"expected={interface.action_lpf_cutoff_hz}"
        )

    control_dt = float(_metadata_scalar(data, "control_dt"))
    expected_alpha = float(
        compute_action_lpf_alpha(interface.action_lpf_cutoff_hz, control_dt)
    )
    recorded_alpha = float(_metadata_scalar(data, "action_lpf_alpha"))
    if not np.isclose(recorded_alpha, expected_alpha, rtol=0.0, atol=1.0e-15):
        raise ValueError(
            f"action LPF alpha mismatch for {recorded_id!r}: "
            f"dataset={recorded_alpha}, expected={expected_alpha}"
        )

    conversion_mode = str(_metadata_scalar(data, "action_conversion_mode"))
    if (
        interface.required_mpx_conversion_mode is not None
        and conversion_mode != interface.required_mpx_conversion_mode
    ):
        raise ValueError(
            f"action conversion mismatch for {recorded_id!r}: "
            f"dataset={conversion_mode!r}, "
            f"expected={interface.required_mpx_conversion_mode!r}"
        )
    return interface
