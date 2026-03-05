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


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization.

    Physics parameters are applied once at env creation ("startup" mode) and
    persist across all episodes within that env instance. Observation noise is
    applied per-step, and perturbations are applied at random intervals.

    The default values are calibrated for the Unitree Go2 quadruped, matching
    the MjLab Go2 velocity task configuration.

    Attributes:
        enable: Master switch. When False, no randomization is applied.

        # ── Physics parameter randomization (startup — applied once) ────
        # These modify the MuJoCo model (mjModel) fields at env creation.
        # Each parallel env gets different values, but they stay fixed
        # across episodes. This matches MjLab's "startup" event mode.

        friction_range: (min, max) absolute range for geom_friction[:, 0]
            (tangential friction). Matches MjLab operation="abs".
            (0.0, 0.0) disables.
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
        motor_strength_range: (min, max) multiplicative scale for torque
            limits. (1.0, 1.0) disables.

        # ── Observation noise (applied every step) ──────────────────────
        # Additive uniform noise on sensor readings to simulate real-sensor
        # noise and imperfect state estimation. Scaled by obs_noise_level.
        # Applied only to policy obs (actor), NOT privileged obs (critic).

        obs_noise_level: Master noise scale (0.0 = no noise, 1.0 = full).
        obs_noise_scales: Per-sensor noise half-widths at level=1.0.
            Keys match observation components.

        # ── External perturbations (applied at random intervals) ────────
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

    # ── Physics parameter randomization (startup — applied once) ────────
    # Default ranges match MjLab Go2: friction + COM + encoder bias enabled;
    # mass, damping, armature, joint friction, gains, motor strength disabled.
    friction_range: tuple[float, float] = (0.3, 1.2)
    added_mass_range: tuple[float, float] = (0.0, 0.0)
    com_displacement_range: tuple[float, float] = (-0.05, 0.05)
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)
    kp_scale_range: tuple[float, float] = (1.0, 1.0)
    kd_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_damping_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_armature_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_friction_range: tuple[float, float] = (0.0, 0.0)
    motor_strength_range: tuple[float, float] = (1.0, 1.0)

    # ── Observation noise ───────────────────────────────────────────────
    obs_noise_level: float = 1.0
    obs_noise_scales: dict[str, float] = field(default_factory=lambda: {
        "joint_pos": 0.01,      # radians (encoder noise)
        "joint_vel": 1.5,       # rad/s (velocity estimation noise)
        "ang_vel": 0.2,         # rad/s (IMU gyroscope noise)
        "gravity": 0.05,        # (IMU orientation estimation noise)
    })

    # ── External perturbations ──────────────────────────────────────────
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

    def to_dict(self) -> dict:
        """Serialize to dict for logging/saving."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DomainRandomizationConfig:
        """Deserialize from dict."""
        return cls(**d)
