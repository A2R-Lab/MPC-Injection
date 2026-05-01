"""Domain randomization configuration and utilities for sim-to-real transfer.

Domain randomization improves sim-to-real transfer by training agents on a wide
range of randomized environments. By varying physical parameters (friction, mass,
etc.), observation noise, and external perturbations, the policy learns to be
robust to the inevitable sim-to-real gap.

Architecture (modeled after MjLab's EventManager):
    - **Startup** (persistent per-env): Physics parameters (friction, COM,
      encoder bias) are randomized once at environment creation and stay fixed
      for the lifetime of the env instance. Each parallel env gets different
      values, but they never change across episodes. This is critical for SAC's
      off-policy replay buffer — transitions within an env come from a
      consistent MDP.
    - **Per-step**: Observation noise (i.i.d. additive uniform noise on actor
      observations only; critic sees clean ground truth).
    - **Interval**: Push perturbations with randomized timing per-env.

References:
    - mjlab (mujocolab/mjlab): randomize_field(), randomize_pd_gains(),
      push_by_setting_velocity(), EventManager startup/reset/interval modes
    - MuJoCo Playground (google-deepmind/mujoco_playground): observation noise
    - IsaacGymEnvs (isaac-sim/IsaacGymEnvs): friction randomization, push robots
    - Legged Gym / Isaac Lab: standard quadruped DR pipeline

Usage:
    # Default config (recommended starting point for Go2 sim-to-real):
    dr_cfg = DomainRandomizationConfig()

    # Disable all randomization (for debugging / ablation):
    dr_cfg = DomainRandomizationConfig.disabled()

    # Custom config:
    dr_cfg = DomainRandomizationConfig(
        friction_range=(0.3, 1.2),
        com_displacement_range=(-0.05, 0.05),
        obs_noise_level=0.5,
    )

    # Pass to environment:
    env = QuadrupedVelocityTrackingEnv(domain_rand_cfg=dr_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


DEFAULT_STARTUP_DOMAIN_RAND_PRESET = "default_no_push"
STARTUP_DOMAIN_RAND_PRESET_NAMES = (
    "default",
    "default_no_push",
    "half_no_push",
    "quarter_no_push",
    "sysid_dyn10_default",
    "sysid_dyn10_default_no_push",
    "sysid_dyn10_half_no_push",
    "sysid_dyn20_mjlab",
    "sysid_dyn20_mjlab_no_push",
    "sysid_floor_only_no_push",
    "sysid_floor_sensing_no_push",
    "disabled",
)
FLOOR_FRICTION_TARGET_GEOM_NAMES = ("ground", "floor", "hfield", "terrain")
MJLAB_CONTACT_FRICTION_TARGET_GEOM_NAMES = (
    "ground",
    "floor",
    "hfield",
    "terrain",
    "FL",
    "FR",
    "RL",
    "RR",
)
STARTUP_DOMAIN_RAND_PATCH_ARRAY_KEYS = (
    "dr_patch_geom_friction",
    "dr_patch_body_mass",
    "dr_patch_body_ipos",
    "dr_patch_dof_damping",
    "dr_patch_dof_armature",
    "dr_patch_dof_frictionloss",
    "dr_patch_actuator_ctrlrange",
    "dr_patch_actuator_forcerange",
    "dr_torque_limits",
)
STARTUP_DOMAIN_RAND_RUNTIME_ARRAY_KEYS = (
    "dr_encoder_bias",
    "dr_realized_kp",
    "dr_realized_kd",
)
STARTUP_DOMAIN_RAND_PATCH_META_KEYS = (
    "dr_enabled",
    "dr_config_type",
    "dr_seed",
    "dr_applied_fields",
    "dr_motor_strength_scale",
    "dr_added_mass_kg",
)
STARTUP_DOMAIN_RAND_PATCH_KEYS = (
    *STARTUP_DOMAIN_RAND_PATCH_META_KEYS,
    *STARTUP_DOMAIN_RAND_PATCH_ARRAY_KEYS,
    *STARTUP_DOMAIN_RAND_RUNTIME_ARRAY_KEYS,
)


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization.

    Physics parameters are applied once at env creation ("startup" mode) and
    persist across all episodes within that env instance. Observation noise is
    applied per-step, and perturbations are applied at random intervals.

    The default values are calibrated for the Unitree Go2 quadruped, matching
    the MjLab Go2 velocity task configuration.

    Attributes:
        enable: When False, no randomization is applied.

        # -- Physics parameter randomization (startup — applied once) ----
        # These modify the MuJoCo model (mjModel) fields at env creation.
        # Each parallel env gets different values, but they stay fixed
        # across episodes. This matches MjLab's "startup" event mode.

        friction_range: (min, max) absolute range for geom_friction[:, 0]
            (tangential friction). Matches MjLab operation="abs".
            (0.0, 0.0) disables.
        friction_target_geom_names: Optional tuple of geom names whose
            tangential friction should be randomized. ``None`` means apply to
            every geom for backward compatibility. The sysID-aware presets use
            this to randomize floor/contact-surface geoms only.
        added_mass_range: (min, max) kg added to the base body mass.
            Simulates payload variation. (0.0, 0.0) disables.
        com_displacement_range: (min, max) meters displacement added to
            body_ipos of the base body (x, y, z independently). Simulates
            center-of-mass shift from payload mounting. (0.0, 0.0) disables.
        encoder_bias_range: (min, max) radians of persistent bias added to
            joint position readings. Simulates encoder calibration error.
            (0.0, 0.0) disables.
        kp_scale_range: (min, max) multiplicative scale for PD proportional
            gain. (1.0, 1.0) disables.
        kd_scale_range: (min, max) multiplicative scale for PD derivative
            gain. (1.0, 1.0) disables.
        joint_damping_scale_range: (min, max) multiplicative scale for
            dof_damping (joint viscous friction). (1.0, 1.0) disables.
        joint_armature_scale_range: (min, max) multiplicative scale for
            dof_armature (rotor inertia reflected to joint). (1.0, 1.0)
            disables.
        joint_friction_range: (min, max) absolute range for dof_frictionloss
            (Coulomb friction at joints). (0.0, 0.0) disables.
        joint_friction_scale_range: (min, max) multiplicative scale for
            dof_frictionloss. (1.0, 1.0) disables. When enabled, this takes
            precedence over joint_friction_range.
        motor_strength_range: (min, max) multiplicative scale for torque
            limits. (1.0, 1.0) disables.

        # -- Observation noise (applied every step) ----------------------
        # Additive uniform noise on sensor readings to simulate real-sensor
        # noise and imperfect state estimation. Scaled by obs_noise_level.
        # Applied only to policy obs (actor), NOT privileged obs (critic).

        obs_noise_level: Master noise scale (0.0 = no noise, 1.0 = full).
        obs_noise_scales: Per-sensor noise half-widths at level=1.0.
            Keys match observation components.

        # -- External perturbations (applied at random intervals) --------
        # Random velocity kicks to the base, simulating external pushes.
        # Applied by adding to the current base velocity (additive, same
        # as MjLab's push_by_setting_velocity). Interval is randomized
        # per-episode from push_interval_range_s.

        push_robots: Whether to apply random velocity pushes.
        push_interval_range_s: (min, max) seconds between pushes. Each
            episode samples a new interval. Matches MjLab's
            interval_range_s=(1.0, 3.0).
        push_velocity_ranges: Per-DOF velocity kick ranges matching MjLab's
            6-DOF push: x, y, z (m/s) and roll, pitch, yaw (rad/s).
    """

    enable: bool = True

    # -- Physics parameter randomization (startup — applied once) --------
    # Default ranges match MjLab Go2: friction + COM + encoder bias enabled;
    # mass, damping, armature, joint friction, gains, motor strength disabled.
    friction_range: tuple[float, float] = (0.3, 1.2)
    friction_target_geom_names: tuple[str, ...] | None = None
    added_mass_range: tuple[float, float] = (0.0, 0.0)
    com_displacement_range: tuple[float, float] = (-0.05, 0.05)
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)
    kp_scale_range: tuple[float, float] = (1.0, 1.0)
    kd_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_damping_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_armature_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_friction_range: tuple[float, float] = (0.0, 0.0)
    joint_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    motor_strength_range: tuple[float, float] = (1.0, 1.0)

    # -- Observation noise -----------------------------------------------
    obs_noise_level: float = 1.0
    obs_noise_scales: dict[str, float] = field(default_factory=lambda: {
        "joint_pos": 0.01,      # radians (encoder noise)
        "joint_vel": 1.5,       # rad/s (velocity estimation noise)
        "ang_vel": 0.2,         # rad/s (IMU gyroscope noise)
        "gravity": 0.05,        # (IMU orientation estimation noise)
    })

    # -- External perturbations ------------------------------------------
    push_robots: bool = True
    push_interval_range_s: tuple[float, float] = (1.0, 3.0)
    push_velocity_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.25, 0.25),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        }
    )

    @classmethod
    def disabled(cls) -> DomainRandomizationConfig:
        """Create a config with all randomization disabled."""
        return cls(enable=False)

    @classmethod
    def default_no_push(cls) -> DomainRandomizationConfig:
        """Create the default config with external perturbations disabled."""
        return cls(push_robots=False)

    @classmethod
    def sysid_floor_only_no_push(cls) -> DomainRandomizationConfig:
        """Randomize floor friction only, leaving sysID'd dynamics untouched."""
        default_cfg = cls.default_no_push()
        return cls(
            friction_range=default_cfg.friction_range,
            friction_target_geom_names=FLOOR_FRICTION_TARGET_GEOM_NAMES,
            added_mass_range=(0.0, 0.0),
            com_displacement_range=(0.0, 0.0),
            encoder_bias_range=(0.0, 0.0),
            kp_scale_range=default_cfg.kp_scale_range,
            kd_scale_range=default_cfg.kd_scale_range,
            joint_damping_scale_range=default_cfg.joint_damping_scale_range,
            joint_armature_scale_range=default_cfg.joint_armature_scale_range,
            joint_friction_range=default_cfg.joint_friction_range,
            joint_friction_scale_range=default_cfg.joint_friction_scale_range,
            motor_strength_range=default_cfg.motor_strength_range,
            obs_noise_level=0.0,
            obs_noise_scales=default_cfg.obs_noise_scales.copy(),
            push_robots=False,
            push_interval_range_s=default_cfg.push_interval_range_s,
            push_velocity_ranges=default_cfg.push_velocity_ranges.copy(),
        )

    @classmethod
    def sysid_floor_sensing_no_push(cls) -> DomainRandomizationConfig:
        """Randomize floor friction and sensing, leaving sysID'd dynamics fixed."""
        default_cfg = cls.default_no_push()
        return cls(
            friction_range=default_cfg.friction_range,
            friction_target_geom_names=FLOOR_FRICTION_TARGET_GEOM_NAMES,
            added_mass_range=(0.0, 0.0),
            com_displacement_range=(0.0, 0.0),
            encoder_bias_range=default_cfg.encoder_bias_range,
            kp_scale_range=default_cfg.kp_scale_range,
            kd_scale_range=default_cfg.kd_scale_range,
            joint_damping_scale_range=default_cfg.joint_damping_scale_range,
            joint_armature_scale_range=default_cfg.joint_armature_scale_range,
            joint_friction_range=default_cfg.joint_friction_range,
            joint_friction_scale_range=default_cfg.joint_friction_scale_range,
            motor_strength_range=default_cfg.motor_strength_range,
            obs_noise_level=default_cfg.obs_noise_level,
            obs_noise_scales=default_cfg.obs_noise_scales.copy(),
            push_robots=False,
            push_interval_range_s=default_cfg.push_interval_range_s,
            push_velocity_ranges=default_cfg.push_velocity_ranges.copy(),
        )

    @staticmethod
    def _scaled_range(
        value_range: tuple[float, float],
        width_scale: float,
    ) -> tuple[float, float]:
        """Shrink a numeric range around its midpoint by a width scale."""
        lo, hi = value_range
        midpoint = 0.5 * (lo + hi)
        half_width = 0.5 * width_scale * (hi - lo)
        return (midpoint - half_width, midpoint + half_width)

    @classmethod
    def half_no_push(cls) -> DomainRandomizationConfig:
        """Create a reduced-strength config with pushes disabled.

        The "half" preset halves the enabled default randomization magnitudes
        while keeping default-disabled terms disabled:
            - numeric ranges are shrunk to half-width around their midpoint
            - observation noise level is halved
            - pushes are disabled entirely
        """
        default_cfg = cls()
        return cls(
            friction_range=cls._scaled_range(default_cfg.friction_range, width_scale=0.5),
            added_mass_range=cls._scaled_range(default_cfg.added_mass_range, width_scale=0.5),
            com_displacement_range=cls._scaled_range(default_cfg.com_displacement_range, width_scale=0.5),
            encoder_bias_range=cls._scaled_range(default_cfg.encoder_bias_range, width_scale=0.5),
            kp_scale_range=cls._scaled_range(default_cfg.kp_scale_range, width_scale=0.5),
            kd_scale_range=cls._scaled_range(default_cfg.kd_scale_range, width_scale=0.5),
            joint_damping_scale_range=cls._scaled_range(default_cfg.joint_damping_scale_range, width_scale=0.5),
            joint_armature_scale_range=cls._scaled_range(default_cfg.joint_armature_scale_range, width_scale=0.5),
            joint_friction_range=cls._scaled_range(default_cfg.joint_friction_range, width_scale=0.5),
            joint_friction_scale_range=cls._scaled_range(default_cfg.joint_friction_scale_range, width_scale=0.5),
            motor_strength_range=cls._scaled_range(default_cfg.motor_strength_range, width_scale=0.5),
            obs_noise_level=0.5 * default_cfg.obs_noise_level,
            obs_noise_scales=default_cfg.obs_noise_scales.copy(),
            push_robots=False,
            push_interval_range_s=cls._scaled_range(default_cfg.push_interval_range_s, width_scale=0.5),
            push_velocity_ranges={
                axis: cls._scaled_range(axis_range, width_scale=0.5)
                for axis, axis_range in default_cfg.push_velocity_ranges.items()
            },
        )

    @classmethod
    def quarter_no_push(cls) -> DomainRandomizationConfig:
        """Create a quarter-strength config with pushes disabled."""
        default_cfg = cls()
        return cls(
            friction_range=cls._scaled_range(default_cfg.friction_range, width_scale=0.25),
            added_mass_range=cls._scaled_range(default_cfg.added_mass_range, width_scale=0.25),
            com_displacement_range=cls._scaled_range(default_cfg.com_displacement_range, width_scale=0.25),
            encoder_bias_range=cls._scaled_range(default_cfg.encoder_bias_range, width_scale=0.25),
            kp_scale_range=cls._scaled_range(default_cfg.kp_scale_range, width_scale=0.25),
            kd_scale_range=cls._scaled_range(default_cfg.kd_scale_range, width_scale=0.25),
            joint_damping_scale_range=cls._scaled_range(default_cfg.joint_damping_scale_range, width_scale=0.25),
            joint_armature_scale_range=cls._scaled_range(default_cfg.joint_armature_scale_range, width_scale=0.25),
            joint_friction_range=cls._scaled_range(default_cfg.joint_friction_range, width_scale=0.25),
            joint_friction_scale_range=cls._scaled_range(default_cfg.joint_friction_scale_range, width_scale=0.25),
            motor_strength_range=cls._scaled_range(default_cfg.motor_strength_range, width_scale=0.25),
            obs_noise_level=0.25 * default_cfg.obs_noise_level,
            obs_noise_scales=default_cfg.obs_noise_scales.copy(),
            push_robots=False,
            push_interval_range_s=cls._scaled_range(default_cfg.push_interval_range_s, width_scale=0.25),
            push_velocity_ranges={
                axis: cls._scaled_range(axis_range, width_scale=0.25)
                for axis, axis_range in default_cfg.push_velocity_ranges.items()
            },
        )

    @classmethod
    def sysid_dyn10_default_no_push(cls) -> DomainRandomizationConfig:
        """Default-no-push DR plus per-joint ±10% dynamics around nominal."""
        base_cfg = cls.default_no_push()
        return cls(
            friction_range=base_cfg.friction_range,
            friction_target_geom_names=base_cfg.friction_target_geom_names,
            added_mass_range=base_cfg.added_mass_range,
            com_displacement_range=base_cfg.com_displacement_range,
            encoder_bias_range=base_cfg.encoder_bias_range,
            kp_scale_range=base_cfg.kp_scale_range,
            kd_scale_range=base_cfg.kd_scale_range,
            joint_damping_scale_range=(0.9, 1.1),
            joint_armature_scale_range=(0.9, 1.1),
            joint_friction_range=base_cfg.joint_friction_range,
            joint_friction_scale_range=(0.9, 1.1),
            motor_strength_range=base_cfg.motor_strength_range,
            obs_noise_level=base_cfg.obs_noise_level,
            obs_noise_scales=base_cfg.obs_noise_scales.copy(),
            push_robots=False,
            push_interval_range_s=base_cfg.push_interval_range_s,
            push_velocity_ranges=base_cfg.push_velocity_ranges.copy(),
        )

    @classmethod
    def sysid_dyn10_default(cls) -> DomainRandomizationConfig:
        """Default DR plus per-joint ±10% dynamics around nominal."""
        base_cfg = cls()
        return cls(
            friction_range=base_cfg.friction_range,
            friction_target_geom_names=base_cfg.friction_target_geom_names,
            added_mass_range=base_cfg.added_mass_range,
            com_displacement_range=base_cfg.com_displacement_range,
            encoder_bias_range=base_cfg.encoder_bias_range,
            kp_scale_range=base_cfg.kp_scale_range,
            kd_scale_range=base_cfg.kd_scale_range,
            joint_damping_scale_range=(0.9, 1.1),
            joint_armature_scale_range=(0.9, 1.1),
            joint_friction_range=base_cfg.joint_friction_range,
            joint_friction_scale_range=(0.9, 1.1),
            motor_strength_range=base_cfg.motor_strength_range,
            obs_noise_level=base_cfg.obs_noise_level,
            obs_noise_scales=base_cfg.obs_noise_scales.copy(),
            push_robots=base_cfg.push_robots,
            push_interval_range_s=base_cfg.push_interval_range_s,
            push_velocity_ranges=base_cfg.push_velocity_ranges.copy(),
        )

    @classmethod
    def sysid_dyn10_half_no_push(cls) -> DomainRandomizationConfig:
        """Half-no-push DR plus per-joint ±10% dynamics around nominal."""
        base_cfg = cls.half_no_push()
        return cls(
            friction_range=base_cfg.friction_range,
            friction_target_geom_names=base_cfg.friction_target_geom_names,
            added_mass_range=base_cfg.added_mass_range,
            com_displacement_range=base_cfg.com_displacement_range,
            encoder_bias_range=base_cfg.encoder_bias_range,
            kp_scale_range=base_cfg.kp_scale_range,
            kd_scale_range=base_cfg.kd_scale_range,
            joint_damping_scale_range=(0.9, 1.1),
            joint_armature_scale_range=(0.9, 1.1),
            joint_friction_range=base_cfg.joint_friction_range,
            joint_friction_scale_range=(0.9, 1.1),
            motor_strength_range=base_cfg.motor_strength_range,
            obs_noise_level=base_cfg.obs_noise_level,
            obs_noise_scales=base_cfg.obs_noise_scales.copy(),
            push_robots=False,
            push_interval_range_s=base_cfg.push_interval_range_s,
            push_velocity_ranges=base_cfg.push_velocity_ranges.copy(),
        )

    @classmethod
    def sysid_dyn20_mjlab_no_push(cls) -> DomainRandomizationConfig:
        """MjLab/IsaacLab-style Go2 DR plus per-joint ±20% sysID dynamics, no pushes."""
        base_cfg = cls.default_no_push()
        return cls(
            friction_range=(0.3, 1.6),
            friction_target_geom_names=MJLAB_CONTACT_FRICTION_TARGET_GEOM_NAMES,
            added_mass_range=(-1.0, 3.0),
            com_displacement_range=(-0.05, 0.05),
            encoder_bias_range=base_cfg.encoder_bias_range,
            kp_scale_range=(0.9, 1.1),
            kd_scale_range=(0.9, 1.1),
            joint_damping_scale_range=(0.8, 1.2),
            joint_armature_scale_range=(0.8, 1.2),
            joint_friction_range=base_cfg.joint_friction_range,
            joint_friction_scale_range=(0.8, 1.2),
            motor_strength_range=(0.9, 1.1),
            obs_noise_level=base_cfg.obs_noise_level,
            obs_noise_scales=base_cfg.obs_noise_scales.copy(),
            push_robots=False,
            push_interval_range_s=(5.0, 10.0),
            push_velocity_ranges=base_cfg.push_velocity_ranges.copy(),
        )

    @classmethod
    def sysid_dyn20_mjlab(cls) -> DomainRandomizationConfig:
        """MjLab/IsaacLab-style Go2 DR plus per-joint ±20% sysID dynamics and pushes."""
        no_push_cfg = cls.sysid_dyn20_mjlab_no_push()
        return cls(
            friction_range=no_push_cfg.friction_range,
            friction_target_geom_names=no_push_cfg.friction_target_geom_names,
            added_mass_range=no_push_cfg.added_mass_range,
            com_displacement_range=no_push_cfg.com_displacement_range,
            encoder_bias_range=no_push_cfg.encoder_bias_range,
            kp_scale_range=no_push_cfg.kp_scale_range,
            kd_scale_range=no_push_cfg.kd_scale_range,
            joint_damping_scale_range=no_push_cfg.joint_damping_scale_range,
            joint_armature_scale_range=no_push_cfg.joint_armature_scale_range,
            joint_friction_range=no_push_cfg.joint_friction_range,
            joint_friction_scale_range=no_push_cfg.joint_friction_scale_range,
            motor_strength_range=no_push_cfg.motor_strength_range,
            obs_noise_level=no_push_cfg.obs_noise_level,
            obs_noise_scales=no_push_cfg.obs_noise_scales.copy(),
            push_robots=True,
            push_interval_range_s=no_push_cfg.push_interval_range_s,
            push_velocity_ranges=no_push_cfg.push_velocity_ranges.copy(),
        )

    @classmethod
    def from_preset(cls, preset: str) -> DomainRandomizationConfig:
        """Create a config from a named preset."""
        preset_factories = {
            "default": cls,
            "default_no_push": cls.default_no_push,
            "half_no_push": cls.half_no_push,
            "quarter_no_push": cls.quarter_no_push,
            "sysid_dyn10_default": cls.sysid_dyn10_default,
            "sysid_dyn10_default_no_push": cls.sysid_dyn10_default_no_push,
            "sysid_dyn10_half_no_push": cls.sysid_dyn10_half_no_push,
            "sysid_dyn20_mjlab": cls.sysid_dyn20_mjlab,
            "sysid_dyn20_mjlab_no_push": cls.sysid_dyn20_mjlab_no_push,
            "sysid_floor_only_no_push": cls.sysid_floor_only_no_push,
            "sysid_floor_sensing_no_push": cls.sysid_floor_sensing_no_push,
            "disabled": cls.disabled,
        }
        try:
            return preset_factories[preset]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown domain randomization preset: {preset!r}. "
                f"Expected one of: {', '.join(sorted(preset_factories))}"
            ) from exc

    def to_dict(self) -> dict:
        """Serialize to dict for logging/saving."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DomainRandomizationConfig:
        """Deserialize from dict."""
        if "friction_target_geom_names" in d and d["friction_target_geom_names"] is not None:
            d = dict(d)
            d["friction_target_geom_names"] = tuple(d["friction_target_geom_names"])
        return cls(**d)


def resolve_startup_domain_rand_config(
    config_type: str | None,
    *,
    default_preset: str = DEFAULT_STARTUP_DOMAIN_RAND_PRESET,
) -> tuple[str, DomainRandomizationConfig]:
    """Resolve a startup DR preset name to a concrete config."""
    resolved_type = default_preset if config_type is None else config_type
    return resolved_type, DomainRandomizationConfig.from_preset(resolved_type)


def extract_startup_domain_rand_patch(source) -> dict | None:
    """Extract a saved startup DR patch from a dict-like source.

    Args:
        source: Mapping-like object such as a dict or np.load(...) result.

    Returns:
        A patch dict with copied arrays/scalars, or None if no DR metadata exists.
    """
    patch = {}
    for key in STARTUP_DOMAIN_RAND_PATCH_KEYS:
        if key not in source:
            continue
        value = source[key]
        if key in STARTUP_DOMAIN_RAND_PATCH_ARRAY_KEYS or key in STARTUP_DOMAIN_RAND_RUNTIME_ARRAY_KEYS:
            patch[key] = np.array(value, copy=True)
        elif key == "dr_enabled":
            patch[key] = bool(np.array(value).item())
        elif key == "dr_seed":
            patch[key] = int(np.array(value).item())
        elif key in {"dr_motor_strength_scale", "dr_added_mass_kg"}:
            patch[key] = float(np.array(value).item())
        elif key == "dr_config_type":
            patch[key] = str(np.array(value).item())
        elif key == "dr_applied_fields":
            patch[key] = np.array(value, copy=True).astype(str)

    if not patch:
        return None
    return patch


def _resolve_friction_target_geom_ids(
    mj_model: mujoco.MjModel,
    target_geom_names: tuple[str, ...] | None,
) -> np.ndarray:
    """Resolve the geom IDs whose friction should be randomized."""
    if target_geom_names is None:
        return np.arange(mj_model.ngeom, dtype=np.int32)

    normalized_targets = {name.lower() for name in target_geom_names}
    target_geom_ids = []
    for geom_id in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and geom_name.lower() in normalized_targets:
            target_geom_ids.append(geom_id)

    if not target_geom_ids:
        targets = ", ".join(sorted(normalized_targets))
        raise ValueError(
            "Could not resolve any target geoms for friction randomization. "
            f"Requested names: {targets}"
        )

    return np.asarray(target_geom_ids, dtype=np.int32)


def sample_startup_domain_rand_patch(
    mj_model: mujoco.MjModel,
    domain_rand_cfg: DomainRandomizationConfig,
    *,
    rng: np.random.RandomState,
    dr_config_type: str,
    dr_seed: int,
    base_body_id: int,
    nominal_kp: np.ndarray | None = None,
    nominal_kd: np.ndarray | None = None,
    num_joints: int | None = None,
) -> dict:
    """Sample one portable startup DR patch for a MuJoCo model.

    This covers only the subset that maps directly onto the MuJoCo plant and
    torque limits plus the startup wrapper terms that affect the policy-facing
    observation and control channels. Per-step observation noise and pushes are
    still runtime env behavior and are intentionally excluded.
    """
    joint_count = int(num_joints if num_joints is not None else mj_model.nu)
    if nominal_kp is None:
        nominal_kp = np.ones(joint_count, dtype=np.float64)
    else:
        nominal_kp = np.array(nominal_kp, copy=True, dtype=np.float64)
    if nominal_kd is None:
        nominal_kd = np.ones(joint_count, dtype=np.float64)
    else:
        nominal_kd = np.array(nominal_kd, copy=True, dtype=np.float64)

    patch = {
        "dr_enabled": bool(domain_rand_cfg.enable),
        "dr_config_type": str(dr_config_type),
        "dr_seed": int(dr_seed),
        "dr_applied_fields": np.array([], dtype="<U32"),
        "dr_motor_strength_scale": 1.0,
        "dr_added_mass_kg": 0.0,
        "dr_patch_geom_friction": mj_model.geom_friction.copy(),
        "dr_patch_body_mass": mj_model.body_mass.copy(),
        "dr_patch_body_ipos": mj_model.body_ipos.copy(),
        "dr_patch_dof_damping": mj_model.dof_damping.copy(),
        "dr_patch_dof_armature": mj_model.dof_armature.copy(),
        "dr_patch_dof_frictionloss": mj_model.dof_frictionloss.copy(),
        "dr_patch_actuator_ctrlrange": mj_model.actuator_ctrlrange.copy(),
        "dr_patch_actuator_forcerange": mj_model.actuator_forcerange.copy(),
        "dr_torque_limits": mj_model.actuator_ctrlrange.copy(),
        "dr_encoder_bias": np.zeros(joint_count, dtype=np.float64),
        "dr_realized_kp": nominal_kp.copy(),
        "dr_realized_kd": nominal_kd.copy(),
    }

    if not domain_rand_cfg.enable:
        return patch

    applied_fields = []

    lo, hi = domain_rand_cfg.friction_range
    if lo != hi:
        friction_geom_ids = _resolve_friction_target_geom_ids(
            mj_model,
            domain_rand_cfg.friction_target_geom_names,
        )
        friction_val = rng.uniform(lo, hi)
        patch["dr_patch_geom_friction"][friction_geom_ids, 0] = friction_val
        applied_fields.append("geom_friction")

    lo, hi = domain_rand_cfg.added_mass_range
    if lo != hi:
        added_mass = rng.uniform(lo, hi)
        patch["dr_patch_body_mass"][base_body_id] += added_mass
        patch["dr_added_mass_kg"] = float(added_mass)
        applied_fields.append("body_mass")

    lo, hi = domain_rand_cfg.com_displacement_range
    if lo != hi:
        com_disp = rng.uniform(lo, hi, size=3)
        patch["dr_patch_body_ipos"][base_body_id] += com_disp
        applied_fields.append("body_ipos")

    lo, hi = domain_rand_cfg.joint_damping_scale_range
    if lo != hi:
        damping_scale = rng.uniform(lo, hi, size=mj_model.dof_damping.shape)
        patch["dr_patch_dof_damping"] *= damping_scale
        applied_fields.append("dof_damping")

    lo, hi = domain_rand_cfg.joint_armature_scale_range
    if lo != hi:
        armature_scale = rng.uniform(lo, hi, size=mj_model.dof_armature.shape)
        patch["dr_patch_dof_armature"] *= armature_scale
        applied_fields.append("dof_armature")

    lo_scale, hi_scale = domain_rand_cfg.joint_friction_scale_range
    if lo_scale != hi_scale:
        friction_scale = rng.uniform(
            lo_scale, hi_scale, size=mj_model.dof_frictionloss.shape
        )
        patch["dr_patch_dof_frictionloss"] *= friction_scale
        applied_fields.append("dof_frictionloss")
    else:
        lo, hi = domain_rand_cfg.joint_friction_range
        if lo != hi:
            patch["dr_patch_dof_frictionloss"] = rng.uniform(
                lo, hi, size=mj_model.dof_frictionloss.shape
            )
            applied_fields.append("dof_frictionloss")

    lo, hi = domain_rand_cfg.encoder_bias_range
    if lo != hi:
        patch["dr_encoder_bias"] = rng.uniform(lo, hi, size=joint_count)
        applied_fields.append("encoder_bias")

    lo, hi = domain_rand_cfg.kp_scale_range
    if lo != hi:
        kp_scale = rng.uniform(lo, hi)
        patch["dr_realized_kp"] = nominal_kp * kp_scale
        applied_fields.append("kp")

    lo, hi = domain_rand_cfg.kd_scale_range
    if lo != hi:
        kd_scale = rng.uniform(lo, hi)
        patch["dr_realized_kd"] = nominal_kd * kd_scale
        applied_fields.append("kd")

    lo, hi = domain_rand_cfg.motor_strength_range
    if lo != hi:
        motor_scale = rng.uniform(lo, hi)
        patch["dr_patch_actuator_ctrlrange"] *= motor_scale
        patch["dr_patch_actuator_forcerange"] *= motor_scale
        patch["dr_torque_limits"] = patch["dr_patch_actuator_ctrlrange"].copy()
        patch["dr_motor_strength_scale"] = float(motor_scale)
        applied_fields.append("torque_limits")

    patch["dr_applied_fields"] = np.array(applied_fields, dtype="<U32")
    return patch


def apply_startup_domain_rand_patch(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    patch: dict | None,
) -> np.ndarray:
    """Apply a previously sampled startup DR patch to a MuJoCo model/data pair.

    Returns:
        The torque-limit array that should be used by the caller.
    """
    torque_limits = mj_model.actuator_ctrlrange.copy()
    if patch is None:
        return torque_limits

    array_keys = {
        "dr_patch_geom_friction": "geom_friction",
        "dr_patch_body_mass": "body_mass",
        "dr_patch_body_ipos": "body_ipos",
        "dr_patch_dof_damping": "dof_damping",
        "dr_patch_dof_armature": "dof_armature",
        "dr_patch_dof_frictionloss": "dof_frictionloss",
        "dr_patch_actuator_ctrlrange": "actuator_ctrlrange",
        "dr_patch_actuator_forcerange": "actuator_forcerange",
    }
    for key, attr in array_keys.items():
        if key in patch:
            getattr(mj_model, attr)[:] = np.array(patch[key], copy=False)

    if "dr_torque_limits" in patch:
        torque_limits = np.array(patch["dr_torque_limits"], copy=True)
        mj_model.actuator_ctrlrange[:] = torque_limits

    mujoco.mj_forward(mj_model, mj_data)
    return torque_limits
