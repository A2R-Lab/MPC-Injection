"""Domain randomization configuration and utilities for sim-to-real transfer.

Domain randomization improves sim-to-real transfer by training agents on a wide
range of randomized environments. By varying physical parameters (friction, mass,
etc.), observation noise, and external perturbations, the policy learns to be
robust to the inevitable sim-to-real gap.

References:
    - mjlab (mujocolab/mjlab): randomize_field(), randomize_pd_gains(), push_by_setting_velocity()
    - MuJoCo Playground (google-deepmind/mujoco_playground): observation noise config
    - IsaacGymEnvs (isaac-sim/IsaacGymEnvs): friction randomization, push robots
    - Legged Gym / Isaac Lab: standard quadruped DR pipeline

Usage:
    # Default config (recommended starting point for Go2 sim-to-real):
    dr_cfg = DomainRandomizationConfig()

    # Disable all randomization (for debugging / ablation):
    dr_cfg = DomainRandomizationConfig.disabled()

    # Custom config:
    dr_cfg = DomainRandomizationConfig(
        friction_range=(0.3, 2.0),
        added_mass_range=(-1.5, 1.5),
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

    All randomization is applied per-episode at reset() time (physics parameters)
    or per-step (observation noise, perturbations). Parameters use multiplicative
    scaling or additive offsets relative to the nominal (XML) model values.

    The default values are calibrated for the Unitree Go2 quadruped based on
    common ranges from Legged Gym, Isaac Lab, mjlab, and MuJoCo Playground.

    Attributes:
        enable: Master switch. When False, no randomization is applied.

        # ── Physics parameter randomization (applied at reset) ──────────
        # These modify the MuJoCo model (mjModel) fields at the start of
        # each episode. Values are restored to nominal before re-randomizing.

        friction_range: (min, max) multiplicative scale for geom_friction[:, 0]
            (tangential friction). 1.0 = nominal. Range [0.2, 2.0] covers
            slippery tile to high-grip rubber.
        added_mass_range: (min, max) kg added to the base body mass.
            Simulates payload variation. [-1.0, 2.0] kg for Go2 (~12 kg).
        com_displacement_range: (min, max) meters displacement added to
            body_ipos of the base body (x, y, z independently). Simulates
            center-of-mass shift from payload mounting.
        kp_scale_range: (min, max) multiplicative scale for PD proportional
            gain. [0.8, 1.2] = ±20% variation.
        kd_scale_range: (min, max) multiplicative scale for PD derivative
            gain. [0.5, 2.0] = wide range since Kd is hard to measure.
        joint_damping_scale_range: (min, max) multiplicative scale for
            dof_damping (joint viscous friction). [0.8, 1.2].
        joint_armature_scale_range: (min, max) multiplicative scale for
            dof_armature (rotor inertia reflected to joint). [0.8, 1.2].
        joint_friction_range: (min, max) absolute range for dof_frictionloss
            (Coulomb friction at joints). [0.0, 0.05] Nm.

        # ── Observation noise (applied every step) ──────────────────────
        # Additive Gaussian noise on sensor readings to simulate real-sensor
        # noise and imperfect state estimation. Scaled by obs_noise_level.

        obs_noise_level: Master noise scale (0.0 = no noise, 1.0 = full noise).
        obs_noise_scales: Per-sensor noise standard deviations at level=1.0.
            Keys match the observation components.

        # ── External perturbations (applied periodically during episode) ─
        # Random velocity kicks to the base, simulating external pushes or
        # collisions. Applied by directly setting base velocity.

        push_robots: Whether to apply random velocity pushes.
        push_interval_s: Average time between pushes (seconds).
        push_vel_xy_range: (min, max) m/s for random base velocity kicks
            in the xy plane.
        push_ang_vel_range: (min, max) rad/s for random angular velocity
            kicks around z axis.

        # ── Motor strength randomization ────────────────────────────────
        motor_strength_range: (min, max) multiplicative scale for torque
            limits. Simulates motor degradation or variation. [0.85, 1.15].
    """

    enable: bool = True

    # ── Physics parameter randomization ─────────────────────────────────
    friction_range: tuple[float, float] = (0.4, 1.8)
    added_mass_range: tuple[float, float] = (-1.0, 2.0)
    com_displacement_range: tuple[float, float] = (-0.05, 0.05)
    kp_scale_range: tuple[float, float] = (0.85, 1.15)
    kd_scale_range: tuple[float, float] = (0.7, 1.3)
    joint_damping_scale_range: tuple[float, float] = (0.8, 1.2)
    joint_armature_scale_range: tuple[float, float] = (0.8, 1.2)
    joint_friction_range: tuple[float, float] = (0.0, 0.05)

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
    push_interval_s: float = 2.0
    push_vel_xy_range: tuple[float, float] = (-0.5, 0.5)
    push_ang_vel_range: tuple[float, float] = (-0.3, 0.3)

    # ── Motor strength randomization ────────────────────────────────────
    motor_strength_range: tuple[float, float] = (0.9, 1.1)

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
