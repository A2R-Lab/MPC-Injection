"""Quadruped velocity tracking environment for RL training.

This environment trains a quadruped robot to track user-commanded velocities (vx, vy, wz).
It is designed for sim2real transfer using **asymmetric actor-critic**: the actor (policy)
only receives sensor readings available on real hardware, while the critic receives
additional privileged information during training.

Observation space (Dict):
    "policy" (45-dim) - Actor observations (real-hardware available):
        - base_ang_vel (body frame, from IMU gyroscope): 3
        - projected_gravity (body frame, from IMU): 3
        - velocity_commands (vx_cmd, vy_cmd, wz_cmd): 3
        - joint_positions (relative to default pose, from encoders): 12
        - joint_velocities (from encoders): 12
        - previous_actions (from policy memory): 12

    "privileged" (3-dim) - Critic-only observations (simulation ground truth):
        - base_linear_velocity (body frame): 3

Action space (12-dim):
    Joint position targets as residuals around the default standing pose.
    raw_target = default_joint_pos + action_scale * action
    action_applied = low_pass_filter(raw_target)
    A PD controller converts targets to torques: tau = Kp*(q_target - q) + Kd*(0 - dq)

The asymmetric observation design enables:
    - The critic to accurately estimate value using ground-truth velocity (which drives
      the reward), leading to better policy gradients during training.
    - The actor to learn a policy that only depends on real-sensor data, enabling
      zero-gap sim2real transfer without teacher-student distillation.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config
from gym_quadruped.utils.mujoco.terrain import generate_terrain

from mpc_rl.envs.action_interfaces import compute_action_lpf_alpha
from mpc_rl.envs.domain_randomization import (
    DomainRandomizationConfig,
    apply_startup_domain_rand_patch as apply_startup_domain_rand_patch_to_model,
    sample_startup_domain_rand_patch,
)
from mpc_rl.envs.go2_sysid import apply_go2_sysid_joint_dynamics

log = logging.getLogger(__name__)

_GENERATION_CONTACT_GEOM_NAMES = frozenset({"ground", "floor", "hfield", "terrain"})
_GENERATION_CONTACT_FRICTION = (0.7, 0.005, 0.0)


class QuadrupedVelocityTrackingEnv(gym.Env):
    """Gymnasium environment for training a quadruped to track commanded velocities.

    Uses only real-hardware-available observations (no base linear velocity).
    Actions are joint position residuals around the default pose, converted
    to torques via a PD controller.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    _SIMPLE_REWARD_CFG = {
        "tracking_sigma": 0.25,
        "w_track_lin_vel": 1.5,
        "w_track_ang_vel": 1.5,
        "w_lin_vel_forward": 1.5,
        "w_ang_vel_forward": 1.0,
        "w_is_terminated": -10.0,
        "w_joint_acc": -3.0e-7,
        "w_action_rate": -0.03,
        "only_positive_rewards": False,
    }

    def __init__(
        self,
        robot: str = "go2",
        scene: str = "flat",
        render_mode: str | None = None,
        use_go2_sysid: bool = True,
        # Simulation parameters
        # Physics at 200 Hz (sim_dt=0.005), control at 50 Hz (decimation=4)
        # to match the real Go2 robot's 50 Hz control loop.
        # Reference: Genesis Go2 env uses dt=0.02 control with substeps=2 (100 Hz physics).
        sim_dt: float = 0.005,
        decimation: int = 4,
        # PD controller gains
        #kp: float = 40.0,
        #kd: float = 0.5,
        
        # Per-joint PD controller gains based on Unitree MjLab
        # stiffness: [20, 20, 40, 20, 20, 40, 20, 20, 40, 20, 20, 40]
        # damping:   [ 1,  1,  2,  1,  1,  2,  1,  1,  2,  1,  1,  2]
        kp: float | dict[str, float] | None = None,
        kd: float | dict[str, float] | None = None,
        
        action_scale: float = 0.5, # NOTE: mjlab uses 0.5
        action_lpf_cutoff_hz: float | None = 5.0,
        # Command ranges
        lin_vel_x_range: tuple[float, float] = (0., 0.5), #(-0.5, 0.5), # NOTE mjlab biases forward
        lin_vel_y_range: tuple[float, float] = (0., 0.), #(-0.25, 0.25),
        ang_vel_z_range: tuple[float, float] = (0., 0.), #(-0.5, 0.5),
        # Reward weights
        reward_cfg: dict[str, float] | None = None,
        # Termination thresholds
        max_pitch: float = 0.5, # 0.5 | radians 87 was too aggressive and allowed for torso tip and hit ground
        max_roll: float = 0.5, # 0.5 | radians
        min_base_height: float = 0.1,
        # Episode settings
        command_resample_interval: int = 250, # og 500
        # Domain randomization
        domain_rand_cfg: DomainRandomizationConfig | None = None,
        apply_startup_domain_rand_on_init: bool = True,
        # Simplified reward mode (for MPC-injection training)
        simple_reward: bool = False,
        # Opt-in exact control-loop diagnostics for trajectory validation
        enable_substep_diagnostics: bool = False,
        # Validation-only fixed pushes keyed by zero-based control step
        deterministic_push_schedule: dict[int, np.ndarray] | None = None,
    ):
        """Initialize the velocity tracking environment.

        Args:
            robot: Robot model name (go2, go1, mini_cheetah, aliengo, etc.).
            scene: Terrain type (flat, perlin, random_boxes, etc.).
            render_mode: Gymnasium render mode ("human", "rgb_array", or None).
            use_go2_sysid: When True, patch the Go2 MuJoCo joint dynamics with
                the identified sysID parameters. Ignored for non-Go2 robots.
            sim_dt: MuJoCo physics timestep in seconds. Default 0.005 (200 Hz)
                for stable contact dynamics.
            decimation: Number of physics steps per control step.
                control_dt = sim_dt * decimation = 0.005 * 4 = 0.02s (50 Hz),
                matching the real Go2 robot's control frequency.
            kp: Proportional gain for PD controller.
            kd: Derivative gain for PD controller.
            action_scale: Scaling factor for action residuals (radians).
            action_lpf_cutoff_hz: First-order low-pass cutoff for absolute joint
                targets before the PD controller. Set to None or <= 0 to disable.
            lin_vel_x_range: Range for commanded x velocity (m/s).
            lin_vel_y_range: Range for commanded y velocity (m/s).
            ang_vel_z_range: Range for commanded yaw rate (rad/s).
            reward_cfg: Dictionary of reward weights. See _default_reward_cfg().
            max_pitch: Maximum pitch angle before termination (radians).
            max_roll: Maximum roll angle before termination (radians).
            min_base_height: Minimum base height before termination (meters).
            command_resample_interval: Resample velocity commands every N control steps.
            simple_reward: If True, use a simplified reward function with only
                velocity tracking and termination penalty. Used when training
                with MPC injection (SAC-MPC/TD3-MPC) to provide a cleaner
                learning signal that aligns better with MPC demonstrations.
            enable_substep_diagnostics: If True, include action clipping and
                per-simulation-substep torque, contact, state, and non-foot
                contact diagnostics in ``info``. Disabled by default.
            deterministic_push_schedule: Optional validation-only mapping from
                zero-based control-step index to a six-element base velocity
                delta `[x, y, z, roll, pitch, yaw]`. When supplied, it replaces
                stochastic interval pushes. Normal training leaves it unset.
        """
        super().__init__()

        # Store configuration
        self.robot_name = robot
        self.use_go2_sysid = bool(use_go2_sysid)
        self.robot_cfg: RobotConfig = get_robot_config(robot_name=robot)
        self.render_mode = render_mode
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.control_dt = sim_dt * decimation

        # PD controller parameters
        self._kp_init = kp
        self._kd_init = kd
        self.action_scale = np.float64(action_scale)
        self.action_lpf_cutoff_hz = (
            None if action_lpf_cutoff_hz is None else np.float64(action_lpf_cutoff_hz)
        )
        self.action_lpf_alpha = self._compute_lpf_alpha(
            self.action_lpf_cutoff_hz,
            self.control_dt,
        )

        # Command ranges
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range
        self.command_resample_interval = command_resample_interval

        # Termination thresholds
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.min_base_height = min_base_height

        # Domain randomization configuration
        self.domain_rand_cfg = domain_rand_cfg or DomainRandomizationConfig()
        self._apply_startup_domain_rand_on_init = apply_startup_domain_rand_on_init
        self._startup_domain_rand_config_type = (
            "custom" if self.domain_rand_cfg.enable else "disabled"
        )
        self._startup_domain_rand_seed = -1
        self._startup_domain_rand_patch: dict | None = None

        # Simplified reward mode
        self.simple_reward = simple_reward
        self.enable_substep_diagnostics = bool(enable_substep_diagnostics)
        self._deterministic_push_schedule = self._validate_push_schedule(
            deterministic_push_schedule
        )

        # Reward configuration
        self.reward_cfg = self._default_reward_cfg()
        if reward_cfg is not None:
            self.reward_cfg.update(reward_cfg)

        # -- Load MuJoCo model --------------------------------------------
        self._load_model(scene)

        # -- Extract default joint positions from keyframe ----------------
        # Reset to keyframe 0 ("home") to get the default standing pose
        temp_data = mujoco.MjData(self.mjModel)
        mujoco.mj_resetDataKeyframe(self.mjModel, temp_data, 0)
        # If the robot config specifies a custom zero position, use that
        if self.robot_cfg.qpos0_js is not None:
            temp_data.qpos[7:] = np.array(self.robot_cfg.qpos0_js)
        self.default_joint_pos = temp_data.qpos[7:].copy().astype(np.float64)
        self.default_qpos = temp_data.qpos.copy().astype(np.float64)
        del temp_data

        # Number of actuated joints
        self.num_joints = self.mjModel.nu
        assert self.num_joints == len(self.default_joint_pos), (
            f"Mismatch: {self.num_joints} actuators vs {len(self.default_joint_pos)} joint positions"
        )

        # Build the per-joint PD gain arrays like MjLab Go2 actuator config
        # Hip joints:   Kp=20, Kd=1
        # Thigh joints: Kp=20, Kd=1
        # Calf joints:  Kp=40, Kd=2
        default_kp_map = {'hip_joint': 20.0, 'thigh_joint': 20.0, 'calf_joint': 40.0}
        default_kd_map = {'hip_joint': 1.0,  'thigh_joint': 1.0,  'calf_joint': 2.0}
        self.kp = self._build_per_joint_gains(self._kp_init, default_kp_map)
        self.kd = self._build_per_joint_gains(self._kd_init, default_kd_map)

        # Extract torque limits from actuator ctrl range
        # NOTE: For MuJoCo motor actuators, ctrl IS the torque, so ctrlrange = torque limits.
        # actuator_forcerange may be zeros if forcelimited is not enabled.
        self.torque_limits = self.mjModel.actuator_ctrlrange.copy()

        # -- Foot contact detection setup --
        # Look up MuJoCo geom IDs for foot collision geoms.
        # Used to detect ground contact for the feet_air_time reward.
        # Go2 XML defines foot geoms named "FL", "FR", "RL", "RR".
        self._foot_geom_ids = {}
        foot_geom_names = self.robot_cfg.feet_geom_names  # {'FL': 'FL', ...}
        for leg_name, geom_name in foot_geom_names.items():
            geom_id = mujoco.mj_name2id(
                self.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            assert geom_id >= 0, (
                f"Foot geom '{geom_name}' not found in MuJoCo model"
            )
            self._foot_geom_ids[leg_name] = geom_id
        self._foot_names = tuple(self._foot_geom_ids.keys())
        self._foot_name_to_idx = {
            name: i for i, name in enumerate(self._foot_names)
        }
        self._foot_geom_id_set = set(self._foot_geom_ids.values())
        self._num_feet = len(self._foot_geom_ids)

        # Match the nominal MuJoCo plant used by quadruped MPC trajectory
        # generation before capturing the baseline model parameters.
        self._set_generation_contact_friction()

        # --- Store nominal model values for domain randomization --------
        # These are the "ground truth" XML values that randomization scales/offsets.
        # Stored once at init so reset() can always start from the nominal model.
        self._nominal_friction = self.mjModel.geom_friction.copy()
        self._nominal_body_mass = self.mjModel.body_mass.copy()
        self._nominal_body_ipos = self.mjModel.body_ipos.copy()
        self._nominal_dof_damping = self.mjModel.dof_damping.copy()
        self._nominal_dof_armature = self.mjModel.dof_armature.copy()
        self._nominal_dof_frictionloss = self.mjModel.dof_frictionloss.copy()
        self._nominal_actuator_forcerange = self.mjModel.actuator_forcerange.copy()
        # Store base body ID for mass/CoM randomization
        # Go2 XML uses "base" as the root body; fall back to body ID 1 (first
        # non-world body) if lookup fails.
        _base_id = mujoco.mj_name2id(
            self.mjModel, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self._base_body_id = _base_id if _base_id >= 0 else 1
        # Nominal PD gains (for startup randomization)
        self._nominal_kp = self.kp.copy()
        self._nominal_kd = self.kd.copy()
        # Nominal torque limits (for startup randomization)
        self._nominal_torque_limits = self.torque_limits.copy()
        # Current motor strength scale (set at startup)
        self._motor_strength_scale = 1.0

        # Persistent encoder bias (set at startup, simulates calibration error)
        self._encoder_bias = np.zeros(self.num_joints, dtype=np.float64)

        # Perturbation tracking
        self._push_interval_steps = 0  # set in reset
        self._steps_since_last_push = 0

        # -- Apply startup domain randomization (once, persistent) --------
        # Physics DR is applied once here and never re-randomized across
        # episodes. Each parallel env gets different values. This matches
        # MjLab's "startup" event mode and is essential for SAC's replay
        # buffer consistency.
        if self._apply_startup_domain_rand_on_init:
            self._apply_startup_randomization()
        else:
            self._startup_domain_rand_patch = self.export_startup_domain_rand_patch(
                dr_config_type=self._startup_domain_rand_config_type,
                dr_seed=self._startup_domain_rand_seed,
            )

        # --- Define observation space (Dict: asymmetric actor-critic) -----
        # "policy" (actor): real-hardware-available sensors
        #   base_ang_vel (body frame):    3
        #   projected_gravity:            3
        #   commands (vx, vy, wz):        3
        #   joint_pos (relative):        12
        #   joint_vel:                   12
        #   previous_actions:            12
        # "privileged" (critic only): simulation ground truth
        #   base_lin_vel (body frame):    3
        self.policy_obs_dim = 3 + 3 + 3 + self.num_joints + self.num_joints + self.num_joints
        self.privileged_obs_dim = 3  # base linear velocity in body frame
        self.observation_space = spaces.Dict({
            "policy": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.policy_obs_dim,),
                dtype=np.float64,
            ),
            "privileged": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.privileged_obs_dim,),
                dtype=np.float64,
            ),
        })

        # -- Define action space ------------------------------------------
        # Actions are joint position residuals in [-1, 1], scaled by action_scale
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float64,
        )

        # -- Internal state tracking -------------------------------------
        self._commands = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(self.num_joints, dtype=np.float64)
        self._prev_last_action = np.zeros(self.num_joints, dtype=np.float64)
        self._raw_q_target = self.default_joint_pos.copy()
        self._filtered_q_target = self.default_joint_pos.copy()
        self._applied_torques = np.zeros(self.num_joints, dtype=np.float64)
        self._last_substep_diagnostics: dict[str, np.ndarray] | None = None
        self._step_count = 0
        self._steps_since_command_resample = 0
        self._fixed_commands = False  # When True, step() will NOT auto-resample commands

        # -- Feet air/contact time tracking (for locomotion rewards) --
        # Continuous timers per foot: air_time increments while in air,
        # contact_time increments while on ground; the other resets to 0.
        self._feet_air_time = np.zeros(self._num_feet, dtype=np.float64)
        self._feet_contact_time = np.zeros(self._num_feet, dtype=np.float64)
        self._last_foot_contacts = np.zeros(self._num_feet, dtype=bool)

        # -- Swing peak tracking (for feet height reward) --
        # Tracks the maximum foot height during each swing phase.
        # On first contact, the peak is compared to the target height.
        self._swing_peak = np.zeros(self._num_feet, dtype=np.float64)
        self._first_contact = np.zeros(self._num_feet, dtype=bool)
        self._current_contacts = np.zeros(self._num_feet, dtype=bool)

        # -- Joint acceleration tracking (for smooth motion penalty) --
        # Penalizing acceleration (d²q/dt²) instead of velocity encourages
        # smooth motion without discouraging joint movement itself.
        self._last_joint_vel = np.zeros(self.num_joints, dtype=np.float64)
        self._joint_acc = np.zeros(self.num_joints, dtype=np.float64)

        # -- Foot geom IDs as ordered list (for vectorized position/velocity) --
        self._foot_geom_id_list = list(self._foot_geom_ids.values())

        # -- Joint limits (soft) for dof_pos_limits penalty --
        # Soft limits at 95% of the actual joint range (same as MuJoCo Playground).
        # jnt_range shape: (njnt, 2) where col 0 = lower, col 1 = upper.
        # Skip the first joint (freejoint has no range).
        soft_factor = 0.95
        jnt_range = self.mjModel.jnt_range[1:]  # skip freejoint
        self._soft_joint_lower = jnt_range[:, 0] * soft_factor
        self._soft_joint_upper = jnt_range[:, 1] * soft_factor

        # -- Per-joint posture standard deviations (for variable_posture reward) --
        # Speed-dependent tolerance: tighter for standing, looser for walking/running.
        # Ported from unitree_rl_mjlab Go2 config.
        self._setup_posture_stds()

        # -- Reward component tracking (for logging) --
        self._reward_components: dict[str, float] = {}

        # -- Rendering ---------------------------------------------------
        self.viewer = None
        self._renderer = None  # For rgb_array mode
        self._render_camera = None  # Tracking camera for rgb_array rendering

    # =====================================================================
    # Gymnasium API
    # =====================================================================

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Execute one control step (multiple sim steps via decimation).

        Args:
            action: Joint position residuals in [-1, 1], shape (num_joints,).

        Returns:
            obs: Dict with "policy" and "privileged" observation arrays.
            reward: Scalar reward.
            terminated: Whether the episode ended due to failure.
            truncated: Whether the episode was truncated (handled by gymnasium wrapper).
            info: Additional information dictionary.
        """
        raw_action = np.asarray(action, dtype=np.float64)
        action = np.clip(raw_action, -1.0, 1.0).astype(np.float64)

        # Store previous action for action rate penalty
        self._prev_last_action = self._last_action.copy()
        self._last_action = action.copy()

        # Compute and filter joint position targets before the PD controller.
        self._raw_q_target = self.default_joint_pos + self.action_scale * action
        q_target = self._apply_action_lpf(self._raw_q_target)

        substep_diagnostics = None
        if self.enable_substep_diagnostics:
            action_clip_delta = raw_action - action
            substep_diagnostics = {
                "raw_action": raw_action.copy(),
                "clipped_action": action.copy(),
                "action_clipping_mask": action_clip_delta != 0.0,
                "action_clipping_magnitude": np.abs(action_clip_delta),
                "raw_q_target": self._raw_q_target.copy(),
                "filtered_q_target": q_target.copy(),
                "requested_torques": np.zeros(
                    (self.decimation, self.num_joints), dtype=np.float64
                ),
                "applied_torques": np.zeros(
                    (self.decimation, self.num_joints), dtype=np.float64
                ),
                "torque_saturation_mask": np.zeros(
                    (self.decimation, self.num_joints), dtype=bool
                ),
                "torque_saturation_magnitude": np.zeros(
                    (self.decimation, self.num_joints), dtype=np.float64
                ),
                "foot_contacts": np.zeros(
                    (self.decimation, self._num_feet), dtype=bool
                ),
                "non_foot_ground_contact": np.zeros(self.decimation, dtype=bool),
                "non_foot_ground_contact_body": np.full(
                    self.decimation, "", dtype="<U64"
                ),
                "non_foot_ground_contact_detail": np.full(
                    self.decimation, "", dtype="<U256"
                ),
                "qpos": np.zeros(
                    (self.decimation, self.mjModel.nq), dtype=np.float64
                ),
                "qvel": np.zeros(
                    (self.decimation, self.mjModel.nv), dtype=np.float64
                ),
                "time": np.zeros(self.decimation, dtype=np.float64),
                "push_delta_qvel": np.zeros(6, dtype=np.float64),
            }

        # Apply PD control for `decimation` simulation steps
        for substep in range(self.decimation):
            q_current = self.mjData.qpos[7:]
            dq_current = self.mjData.qvel[6:]

            # PD controller: tau = Kp * (q_target - q) + Kd * (0 - dq)
            requested_torques = (
                self.kp * (q_target - q_current) + self.kd * (0.0 - dq_current)
            )

            # Clip torques to actuator limits
            torques = np.clip(
                requested_torques,
                self.torque_limits[:, 0],
                self.torque_limits[:, 1],
            )
            self._applied_torques = torques

            self.mjData.ctrl[:] = torques
            mujoco.mj_step(self.mjModel, self.mjData)

            if substep_diagnostics is not None:
                torque_clip_delta = requested_torques - torques
                substep_diagnostics["requested_torques"][substep] = (
                    requested_torques
                )
                substep_diagnostics["applied_torques"][substep] = torques
                substep_diagnostics["torque_saturation_mask"][substep] = (
                    torque_clip_delta != 0.0
                )
                substep_diagnostics["torque_saturation_magnitude"][substep] = (
                    np.abs(torque_clip_delta)
                )
                substep_diagnostics["foot_contacts"][substep] = (
                    self._get_foot_contacts()
                )
                non_foot_contact = self._find_non_foot_ground_contact()
                if non_foot_contact is not None:
                    body_name, detail = non_foot_contact
                    substep_diagnostics["non_foot_ground_contact"][substep] = True
                    substep_diagnostics["non_foot_ground_contact_body"][substep] = (
                        body_name
                    )
                    substep_diagnostics["non_foot_ground_contact_detail"][substep] = (
                        detail
                    )
                substep_diagnostics["qpos"][substep] = self.mjData.qpos
                substep_diagnostics["qvel"][substep] = self.mjData.qvel
                substep_diagnostics["time"][substep] = self.mjData.time

        self._step_count += 1
        self._steps_since_command_resample += 1

        # Apply random perturbation (push) periodically
        base_qvel_before_push = None
        if substep_diagnostics is not None:
            base_qvel_before_push = self.mjData.qvel[:6].copy()
        self._maybe_push_robot()
        if substep_diagnostics is not None:
            substep_diagnostics["push_delta_qvel"] = (
                self.mjData.qvel[:6] - base_qvel_before_push
            )
            self._last_substep_diagnostics = substep_diagnostics

        # Update feet air time tracking (also updates swing peak & first_contact)
        self._update_feet_air_time()

        # Compute joint acceleration for smooth-motion penalty
        joint_vel_current = self.mjData.qvel[6:].copy()
        self._joint_acc = (joint_vel_current - self._last_joint_vel) / self.control_dt
        self._last_joint_vel = joint_vel_current

        # Resample commands periodically (only during training, not when commands are set externally)
        if not self._fixed_commands and self._steps_since_command_resample >= self.command_resample_interval:
            self._sample_commands()

        # Compute observation and termination
        obs = self._get_obs()
        terminated = self._check_termination()

        # Compute reward (needs terminated flag for termination cost)
        reward = self._compute_reward(action, terminated)

        truncated = False  # Handled by gymnasium's max_episode_steps
        info = self._get_info()

        # Reset swing peak for feet that just made contact
        self._swing_peak *= ~self._current_contacts

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            obs: Dict with "policy" and "privileged" observation arrays.
            info: Initial info dictionary.
        """
        super().reset(seed=seed)

        # Reset MuJoCo state completely
        mujoco.mj_resetData(self.mjModel, self.mjData)

        # Reset to keyframe ("home" standing pose)
        mujoco.mj_resetDataKeyframe(self.mjModel, self.mjData, 0)

        # In the no-DR setting, always restore the nominal contact friction so
        # resets are idempotent even if a previous code path mutated mjModel.
        if not self.domain_rand_cfg.enable:
            self.mjModel.geom_friction[:] = self._nominal_friction

        # Apply custom zero position if specified
        if self.robot_cfg.qpos0_js is not None:
            self.mjData.qpos[7:] = np.array(self.robot_cfg.qpos0_js)

        # Add small randomization to initial joint positions
        noise_scale_pos = 0.05  # radians
        self.mjData.qpos[7:] += self.np_random.uniform(
            -noise_scale_pos, noise_scale_pos, size=self.num_joints
        )

        # Small random perturbation to base orientation (roll, pitch only)
        roll_noise = self.np_random.uniform(-0.03, 0.03)
        pitch_noise = self.np_random.uniform(-0.03, 0.03)
        base_quat_wxyz = self.mjData.qpos[3:7].copy()
        base_quat_xyzw = np.roll(base_quat_wxyz, -1)
        base_rot = Rotation.from_quat(base_quat_xyzw)
        noise_rot = Rotation.from_euler("xyz", [roll_noise, pitch_noise, 0.0])
        combined_rot = noise_rot * base_rot
        combined_quat_xyzw = combined_rot.as_quat()
        self.mjData.qpos[3:7] = np.roll(combined_quat_xyzw, 1)  # back to wxyz

        # Zero all velocities and accelerations
        self.mjData.qvel[:] = 0.0
        self.mjData.qacc[:] = 0.0
        self.mjData.qacc_warmstart[:] = 0.0
        self.mjData.ctrl[:] = 0.0

        # Compute forward kinematics without advancing simulation
        mujoco.mj_forward(self.mjModel, self.mjData)

        # Add small velocity noise AFTER forward kinematics
        self.mjData.qvel[6:] = self.np_random.uniform(
            -0.05, 0.05, size=self.mjModel.nv - 6
        )

        # Reset internal state
        self._last_action = np.zeros(self.num_joints, dtype=np.float64)
        self._prev_last_action = np.zeros(self.num_joints, dtype=np.float64)
        self._raw_q_target = self.default_joint_pos.copy()
        self._filtered_q_target = self.default_joint_pos.copy()
        self._applied_torques = np.zeros(self.num_joints, dtype=np.float64)
        self._last_substep_diagnostics = None
        self._feet_air_time = np.zeros(self._num_feet, dtype=np.float64)
        self._feet_contact_time = np.zeros(self._num_feet, dtype=np.float64)
        self._last_foot_contacts = np.zeros(self._num_feet, dtype=bool)
        self._swing_peak = np.zeros(self._num_feet, dtype=np.float64)
        self._first_contact = np.zeros(self._num_feet, dtype=bool)
        self._current_contacts = np.zeros(self._num_feet, dtype=bool)
        self._reward_components = {}
        self._last_joint_vel = np.zeros(self.num_joints, dtype=np.float64)
        self._joint_acc = np.zeros(self.num_joints, dtype=np.float64)
        self._step_count = 0
        self._steps_since_command_resample = 0

        # Reset perturbation tracking (sample random interval per episode)
        self._resample_push_interval()
        self._steps_since_last_push = 0

        # Sample new velocity commands (only if not using externally fixed commands)
        if not self._fixed_commands:
            self._sample_commands()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    @staticmethod
    def _compute_lpf_alpha(cutoff_hz: float | None, dt: float) -> np.float64:
        """Return first-order LPF alpha for y += alpha * (x - y)."""
        return compute_action_lpf_alpha(cutoff_hz, dt)

    @staticmethod
    def _validate_push_schedule(
        schedule: dict[int, np.ndarray] | None,
    ) -> dict[int, np.ndarray] | None:
        """Validate and copy a deterministic validation push schedule."""
        if schedule is None:
            return None

        validated = {}
        for control_step, delta in schedule.items():
            if not isinstance(control_step, (int, np.integer)) or control_step < 0:
                raise ValueError(
                    "deterministic push steps must be non-negative integers"
                )
            delta_array = np.asarray(delta, dtype=np.float64)
            if delta_array.shape != (6,):
                raise ValueError(
                    "deterministic push deltas must have shape (6,), got "
                    f"{delta_array.shape} at step {control_step}"
                )
            if not np.all(np.isfinite(delta_array)):
                raise ValueError("deterministic push deltas must be finite")
            validated[int(control_step)] = delta_array.copy()
        return validated

    def _apply_action_lpf(self, raw_q_target: np.ndarray) -> np.ndarray:
        """Filter absolute joint-position targets before PD control."""
        if self.action_lpf_alpha >= 1.0:
            self._filtered_q_target = raw_q_target.copy()
        else:
            self._filtered_q_target += (
                self.action_lpf_alpha * (raw_q_target - self._filtered_q_target)
            )
        return self._filtered_q_target.copy()

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", else None.
        """
        if self.render_mode == "human":
            try:
                import mujoco.viewer as mj_viewer
            except (ImportError, AttributeError):
                log.warning("mujoco.viewer not available (headless environment?)")
                return None

            if self.viewer is None:
                self.viewer = mj_viewer.launch_passive(
                    self.mjModel,
                    self.mjData,
                    show_left_ui=False,
                    show_right_ui=False,
                )
                mujoco.mjv_defaultFreeCamera(self.mjModel, self.viewer.cam)
            # Update camera to follow robot
            base_pos = self.mjData.qpos[0:3]
            self.viewer.cam.lookat[:] = base_pos
            self.viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.mjModel, height=480, width=640)
            if self._render_camera is None:
                self._render_camera = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(self.mjModel, self._render_camera)
            # Update camera to follow the robot's base position
            base_pos = self.mjData.qpos[0:3]
            self._render_camera.lookat[:] = base_pos
            self._renderer.update_scene(self.mjData, camera=self._render_camera)
            return self._renderer.render()

        return None

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._render_camera = None

    # =====================================================================
    # Public API for external command control
    # =====================================================================

    def set_commands(self, vx: float, vy: float = 0.0, wz: float = 0.0):
        """Set velocity commands externally (disables automatic resampling).

        Use this during evaluation or teleoperation to override the
        automatic command resampling that occurs during training.

        Args:
            vx: Commanded linear velocity in x (m/s).
            vy: Commanded linear velocity in y (m/s).
            wz: Commanded angular velocity around z (rad/s).
        """
        self._commands[0] = np.float64(vx)
        self._commands[1] = np.float64(vy)
        self._commands[2] = np.float64(wz)
        self._fixed_commands = True

    def release_commands(self):
        """Re-enable automatic command resampling (for training)."""
        self._fixed_commands = False

    # =====================================================================
    # Internal methods
    # =====================================================================

    def _build_per_joint_gains(
        self,
        user_value: float | dict[str, float] | None,
        default_map: dict[str, float],
    ) -> np.ndarray:
        """Build a per-joint gain array from user input or per-joint-type defaults.

        Args:
            user_value: If float, use that scalar for all joints.
                If dict mapping joint-type substrings to floats, use per-type values.
                If None, use default_map (MJLab Go2 actuator config).
            default_map: Fallback mapping from joint-type substrings (e.g.
                'hip_joint', 'thigh_joint', 'calf_joint') to gain values.

        Returns:
            np.ndarray of shape (num_joints,) with per-joint gain values.
        """
        gains = np.zeros(self.num_joints, dtype=np.float64)
        if isinstance(user_value, (int, float)):
            gains[:] = float(user_value)
            return gains

        gain_map = user_value if isinstance(user_value, dict) else default_map
        for i in range(self.num_joints):
            joint_id = self.mjModel.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(
                self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            matched = False
            for pattern, val in gain_map.items():
                if pattern in joint_name:
                    gains[i] = val
                    matched = True
                    break
            if not matched:
                # Fall back to the max value in the map
                fallback = max(gain_map.values())
                log.warning(
                    f"Joint '{joint_name}' (actuator {i}) does not match any "
                    f"pattern in gain map {list(gain_map.keys())}. "
                    f"Using fallback={fallback}."
                )
                gains[i] = fallback
        return gains

    def _setup_posture_stds(self):
        """Set up per-joint standard deviations for the variable posture reward.

        Maps each actuated joint to its type (hip/thigh/calf) and assigns
        speed-dependent standard deviations that control how strictly each
        joint must stay near the default pose.

        Smaller std = tighter tolerance (less deviation allowed).
        Larger std = looser tolerance (more deviation allowed).

        Three speed regimes (based on total command speed):
            - standing: Tight tolerance for holding default pose
            - walking: Moderate tolerance for normal locomotion
            - running: Loose tolerance for large joint excursions

        Values from unitree_rl_mjlab Go2 configuration.
        """
        # Per-joint-type standard deviations (Go2 config from mjlab, with a
        # mildly tighter walking/running hip tolerance to reduce lateral bowing).
        std_map = {
            'hip_joint':   {'standing': 0.05, 'walking': 0.13, 'running': 0.13},
            'thigh_joint': {'standing': 0.1,  'walking': 0.35, 'running': 0.35},
            'calf_joint':  {'standing': 0.15, 'walking': 0.5,  'running': 0.5},
        }

        self._posture_std_standing = np.zeros(self.num_joints, dtype=np.float64)
        self._posture_std_walking = np.zeros(self.num_joints, dtype=np.float64)
        self._posture_std_running = np.zeros(self.num_joints, dtype=np.float64)

        for i in range(self.num_joints):
            # Get the joint name for this actuator
            joint_id = self.mjModel.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(
                self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )

            # Match joint name to type and assign stds
            matched = False
            for pattern, stds in std_map.items():
                if pattern in joint_name:
                    self._posture_std_standing[i] = stds['standing']
                    self._posture_std_walking[i] = stds['walking']
                    self._posture_std_running[i] = stds['running']
                    matched = True
                    break

            if not matched:
                # Unknown joint type: use moderate default tolerance
                log.warning(
                    f"Joint '{joint_name}' (actuator {i}) does not match "
                    f"any known type (hip/thigh/calf). Using default posture stds."
                )
                self._posture_std_standing[i] = 0.1
                self._posture_std_walking[i] = 0.35
                self._posture_std_running[i] = 0.5

    def _load_model(self, scene: str):
        """Load the MuJoCo model with the specified robot and scene."""
        import gym_quadruped

        gym_quad_dir = Path(gym_quadruped.__file__).parent
        base_path = gym_quad_dir / "robot_model"
        procedural_assets_path = gym_quad_dir / "utils" / "mujoco" / "assets"

        # Check for scene-specific XML first, then fall back to flat
        base_scene_path = base_path / f"scene_{scene}.xml"
        if not base_scene_path.exists():
            base_scene_path = procedural_assets_path / f"scene_{scene}.xml"

        # Generate terrain
        scene_env, self.terrain_limits = generate_terrain(
            base_scene_path, procedural_assets_path, self.robot_cfg.hip_height, scene, seed=10
        )

        # Insert robot model into scene
        root = scene_env.getroot()
        robot_xml_path = base_path / self.robot_cfg.mjcf_filename
        assert robot_xml_path.exists(), f"Robot model not found: {robot_xml_path}"

        include = ET.Element("include")
        include.attrib["file"] = str(robot_xml_path.absolute().resolve())
        root.insert(0, include)

        # Write combined scene to temp file and load
        combined_scene_path = procedural_assets_path / f"{self.robot_name}-{scene}-veltrack.xml"
        scene_env.write(combined_scene_path)

        self.mjModel = mujoco.MjModel.from_xml_path(str(combined_scene_path.absolute()))
        if self.robot_name.lower() == "go2" and self.use_go2_sysid:
            apply_go2_sysid_joint_dynamics(self.mjModel)
        self.mjData = mujoco.MjData(self.mjModel)

        # Set simulation timestep
        self.mjModel.opt.timestep = self.sim_dt

        # Apply custom zero position if specified
        if self.robot_cfg.qpos0_js is not None:
            self.mjModel.qpos0[7:] = np.array(self.robot_cfg.qpos0_js)

    def _generation_contact_geom_ids(self) -> set[int]:
        """Return floor and foot geom IDs that must match the MPC plant."""
        target_geom_ids = set(self._foot_geom_ids.values())

        for geom_id in range(self.mjModel.ngeom):
            geom_name = mujoco.mj_id2name(
                self.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if geom_name and geom_name.lower() in _GENERATION_CONTACT_GEOM_NAMES:
                target_geom_ids.add(geom_id)

        return target_geom_ids

    def _set_generation_contact_friction(self):
        """Mirror QuadrupedEnv ground/foot friction for the nominal plant."""
        friction = np.array(_GENERATION_CONTACT_FRICTION, dtype=np.float64)
        for geom_id in self._generation_contact_geom_ids():
            self.mjModel.geom_friction[geom_id, :] = friction

    def assert_generation_contact_friction_matches(self):
        """Raise if the no-DR contact friction drifts from the MPC plant."""
        expected = np.array(_GENERATION_CONTACT_FRICTION, dtype=np.float64)
        mismatches = []

        for geom_id in sorted(self._generation_contact_geom_ids()):
            geom_name = mujoco.mj_id2name(
                self.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            actual = self.mjModel.geom_friction[geom_id, :]
            if not np.allclose(actual, expected):
                label = geom_name if geom_name is not None else f"geom_{geom_id}"
                mismatches.append(
                    f"{label}: got {actual.tolist()}, expected {expected.tolist()}"
                )

        if mismatches:
            details = "; ".join(mismatches)
            raise AssertionError(
                "QuadrupedVelocityTrackingEnv contact friction no longer matches "
                f"the MPC generation plant: {details}"
            )

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Compute the observation dictionary for asymmetric actor-critic.

        Returns a Dict with two keys:
            "policy" (45-dim): Real-hardware-available observations for the actor.
                [0:3]   base angular velocity (body frame) - from IMU gyroscope
                [3:6]   projected gravity (body frame) - from IMU orientation
                [6:9]   velocity commands (vx, vy, wz) - from user input
                [9:21]  joint positions relative to default - from encoders
                [21:33] joint velocities - from encoders
                [33:45] previous actions - from policy buffer

            "privileged" (3-dim): Simulation-only ground truth for the critic.
                [0:3]   base linear velocity (body frame) - from simulation
        """
        # Base angular velocity in body frame (MuJoCo qvel[3:6] is already body frame)
        base_ang_vel = self.mjData.qvel[3:6].copy()

        # Projected gravity in body frame
        projected_gravity = self._projected_gravity()

        # Velocity commands
        commands = self._commands.copy()

        # Joint positions relative to default standing pose
        # Add persistent encoder bias (calibration error, set at startup)
        joint_pos_rel = (self.mjData.qpos[7:].copy() - self.default_joint_pos
                         + self._encoder_bias)

        # Joint velocities
        joint_vel = self.mjData.qvel[6:].copy()

        # Previous actions
        prev_actions = self._last_action.copy()

        # -- Apply observation noise for sim-to-real robustness ----------
        # Additive uniform noise on sensor readings, following MuJoCo Playground
        # and Legged Gym conventions. Noise is only applied to the policy
        # observations (what the real robot would see), NOT privileged observations.
        dr = self.domain_rand_cfg
        if dr.enable and dr.obs_noise_level > 0.0:
            noise_level = dr.obs_noise_level
            scales = dr.obs_noise_scales

            # IMU gyroscope noise
            base_ang_vel += self.np_random.uniform(
                -1, 1, size=base_ang_vel.shape
            ) * noise_level * scales.get("ang_vel", 0.0)

            # IMU orientation (gravity projection) noise
            projected_gravity += self.np_random.uniform(
                -1, 1, size=projected_gravity.shape
            ) * noise_level * scales.get("gravity", 0.0)

            # Joint encoder position noise
            joint_pos_rel += self.np_random.uniform(
                -1, 1, size=joint_pos_rel.shape
            ) * noise_level * scales.get("joint_pos", 0.0)

            # Joint encoder velocity noise
            joint_vel += self.np_random.uniform(
                -1, 1, size=joint_vel.shape
            ) * noise_level * scales.get("joint_vel", 0.0)

        # Policy observations (real-hardware available)
        policy_obs = np.concatenate([
            base_ang_vel,       # 3
            projected_gravity,  # 3
            commands,           # 3
            joint_pos_rel,      # num_joints (12)
            joint_vel,          # num_joints (12)
            prev_actions,       # num_joints (12)
        ]).astype(np.float64)

        # Privileged observations (simulation only - for critic during training)
        # NOTE: No noise on privileged obs — the critic should see ground truth.
        base_lin_vel_body = self._base_lin_vel_body()
        privileged_obs = base_lin_vel_body.astype(np.float64)

        return {
            "policy": policy_obs,
            "privileged": privileged_obs,
        }

    def _projected_gravity(self) -> np.ndarray:
        """Compute the gravity vector projected into the body frame.

        When the robot is upright, this returns approximately [0, 0, -1].
        Measurable on real hardware via IMU orientation estimate.
        """
        quat_wxyz = self.mjData.qpos[3:7]
        quat_xyzw = np.roll(quat_wxyz, -1)
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        g_world = np.array([0.0, 0.0, -1.0])
        return (R.T @ g_world).astype(np.float64)

    def _base_lin_vel_body(self) -> np.ndarray:
        """Get base linear velocity in body frame (for reward computation only).

        NOTE: This is not available on real hardware without estimation. Used
        only for computing the velocity tracking reward during training.
        """
        quat_wxyz = self.mjData.qpos[3:7]
        quat_xyzw = np.roll(quat_wxyz, -1)
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        return R.T @ self.mjData.qvel[0:3]

    def _get_foot_contacts(self) -> np.ndarray:
        """Detect which feet are in contact with the ground.

        Checks MuJoCo contact data for contacts between foot geoms and the
        ground plane (world body, id=0).

        Returns:
            Boolean array of shape (num_feet,) in order [FL, FR, RL, RR].
        """
        contacts = np.zeros(self._num_feet, dtype=bool)
        foot_id_to_idx = {
            gid: i for i, gid in enumerate(self._foot_geom_ids.values())
        }

        for i in range(self.mjData.ncon):
            contact = self.mjData.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            body1 = self.mjModel.geom_bodyid[geom1]
            body2 = self.mjModel.geom_bodyid[geom2]

            # One geom must be the ground (world body = 0)
            if body1 == 0 or body2 == 0:
                other_geom = geom2 if body1 == 0 else geom1
                if other_geom in self._foot_geom_id_set:
                    contacts[foot_id_to_idx[other_geom]] = True

        return contacts

    def _geom_body_label(self, geom_id: int) -> str:
        """Return a compact geom/body label for contact diagnostics."""
        geom_name = mujoco.mj_id2name(
            self.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        body_id = int(self.mjModel.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            self.mjModel, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        geom_label = geom_name if geom_name else f"geom_{geom_id}"
        body_label = body_name if body_name else f"body_{body_id}"
        return f"{geom_label}({body_label})"

    def _body_is_descendant_of(self, body_id: int, root_body_id: int) -> bool:
        """Return whether a body belongs to the requested MuJoCo subtree."""
        while body_id != 0:
            if body_id == root_body_id:
                return True
            body_id = int(self.mjModel.body_parentid[body_id])
        return False

    def _find_non_foot_ground_contact(self) -> tuple[str, str] | None:
        """Return the first robot non-foot/world-ground contact, if present."""
        for contact_idx in range(self.mjData.ncon):
            contact = self.mjData.contact[contact_idx]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.mjModel.geom_bodyid[geom1])
            body2 = int(self.mjModel.geom_bodyid[geom2])

            if body1 == 0 and body2 == 0:
                continue
            if body1 != 0 and body2 != 0:
                continue

            ground_geom = geom1 if body1 == 0 else geom2
            robot_geom = geom2 if body1 == 0 else geom1
            if robot_geom in self._foot_geom_id_set:
                continue

            robot_body_id = int(self.mjModel.geom_bodyid[robot_geom])
            if not self._body_is_descendant_of(
                robot_body_id, int(self._base_body_id)
            ):
                continue

            robot_body_name = mujoco.mj_id2name(
                self.mjModel, mujoco.mjtObj.mjOBJ_BODY, robot_body_id
            )
            detail = (
                f"{self._geom_body_label(robot_geom)} touched "
                f"{self._geom_body_label(ground_geom)}"
            )
            return (robot_body_name or f"body_{robot_body_id}", detail)

        return None

    def _get_foot_positions(self) -> np.ndarray:
        """Get world-frame positions of all feet. Shape (num_feet, 3)."""
        positions = np.zeros((self._num_feet, 3))
        for i, gid in enumerate(self._foot_geom_id_list):
            positions[i] = self.mjData.geom_xpos[gid]
        return positions

    def _get_foot_velocities(self) -> np.ndarray:
        """Get world-frame linear velocities of all feet. Shape (num_feet, 3)."""
        velocities = np.zeros((self._num_feet, 3))
        jacp = np.zeros((3, self.mjModel.nv))
        jacr = np.zeros((3, self.mjModel.nv))
        for i, gid in enumerate(self._foot_geom_id_list):
            mujoco.mj_jacGeom(self.mjModel, self.mjData, jacp, jacr, gid)
            velocities[i] = jacp @ self.mjData.qvel
        return velocities

    def _update_feet_air_time(self):
        """Update per-foot air time and contact time tracking.

        Maintains continuous timers for each foot:
        - air_time: increments while foot is in the air, resets on contact
        - contact_time: increments while foot is on the ground, resets on liftoff

        Also detects first_contact events (air→ground transitions) needed
        for the soft_landing reward, and tracks swing peak height.

        Contact filtering: contacts are OR-ed with previous step contacts
        to smooth noisy MuJoCo contact reporting.
        """
        # Current foot contacts
        contacts = self._get_foot_contacts()

        # Filter contacts (OR with last step to smooth noisy contact reporting)
        contact_filt = np.logical_or(contacts, self._last_foot_contacts)

        # Detect first contact: foot was in the air and just landed
        first_contact = (self._feet_air_time > 0.0) & contact_filt

        # Update per-foot timers based on filtered contacts
        for i in range(self._num_feet):
            if contact_filt[i]:
                self._feet_contact_time[i] += self.control_dt
                self._feet_air_time[i] = 0.0
            else:
                self._feet_air_time[i] += self.control_dt
                self._feet_contact_time[i] = 0.0

        # Update swing peak: track max foot z during swing phase
        foot_positions = self._get_foot_positions()
        foot_z = foot_positions[:, 2]  # z heights
        not_in_contact = ~contact_filt
        self._swing_peak = np.where(
            not_in_contact, np.maximum(self._swing_peak, foot_z), self._swing_peak
        )

        # Store for use in reward computation and swing_peak reset in step()
        self._first_contact = first_contact.copy()
        self._current_contacts = contacts.copy()

        # Store contacts for next step
        self._last_foot_contacts = contacts.copy()

    def _get_foot_contact_forces(self) -> np.ndarray:
        """Get contact force magnitudes per foot from MuJoCo contact data.

        Sums forces from all contact pairs involving each foot geom and the
        ground (world body). Returns the total force magnitude per foot.

        Returns:
            Force magnitude array of shape (num_feet,).
        """
        forces = np.zeros(self._num_feet, dtype=np.float64)
        foot_id_to_idx = {
            gid: i for i, gid in enumerate(self._foot_geom_ids.values())
        }
        result = np.zeros(6, dtype=np.float64)

        for i in range(self.mjData.ncon):
            contact = self.mjData.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            body1 = self.mjModel.geom_bodyid[geom1]
            body2 = self.mjModel.geom_bodyid[geom2]

            # One geom must be the ground (world body = 0)
            if body1 == 0 or body2 == 0:
                other_geom = geom2 if body1 == 0 else geom1
                if other_geom in self._foot_geom_id_set:
                    mujoco.mj_contactForce(
                        self.mjModel, self.mjData, i, result
                    )
                    force_magnitude = np.linalg.norm(result[:3])
                    forces[foot_id_to_idx[other_geom]] += force_magnitude

        return forces

    def _compute_reward(self, action: np.ndarray, terminated: bool) -> float:
        """Compute the reward for the current step.

        Dispatches to either the full reward or a simplified reward based on
        self.simple_reward. The simplified reward is used when training with
        MPC injection (SAC-MPC/TD3-MPC) to provide a cleaner learning signal.

        Reward formulation ported from unitree_rl_mjlab velocity tracking task,
        which is proven to produce stable quadruped locomotion gaits.

        Positive rewards (encourage desired behavior):
            - track_linear_velocity: Exponential tracking of commanded xy vel
            - track_angular_velocity: Exponential tracking of commanded yaw rate
            - lin_vel_forward: Linear forward velocity toward command
            - ang_vel_forward: Linear angular velocity toward command
            - alive: Constant per-step survival bonus
            - variable_posture: Speed-dependent default pose tracking
            - track_base_height: Exponential tracking of target base height
            - feet_air_time: Encourage trotting gait with proper timing
            - foot_gait: Encourage diagonal trot contact timing

        Penalties (discourage undesired behavior):
            - flat_orientation_l2: Penalize body tilt
            - body_ang_vel: Penalize excessive body angular velocity (world xy)
            - angular_momentum: Penalize whole-body angular momentum
            - is_terminated: Large penalty for falling
            - joint_acc_l2: Penalize jerky joint motion
            - joint_pos_limits: Penalize joints near limits
            - action_rate_l2: Penalize rapid action changes
            - feet_clearance: Penalize incorrect foot height during swing
            - feet_slip: Penalize foot sliding during contact
            - soft_landing: Penalize high impact forces at landing
            - bad_two_foot_contacts: Penalize bounding/pacing two-foot support
            - lateral_vel: Penalize sideways drift when commanded to walk
            - pitch_tilt: Penalize persistent forward/backward body tilt
        """
        if self.simple_reward:
            return self._compute_simple_reward(action, terminated)
        cfg = self.reward_cfg

        # -- Ground truth velocities (simulation only) --
        base_lin_vel_body = self._base_lin_vel_body()
        base_ang_vel_body = self.mjData.qvel[3:6].copy()  # body frame

        # -- Command magnitude for scaling locomotion-specific rewards --
        cmd_lin_norm = np.linalg.norm(self._commands[:2])
        cmd_ang_norm = abs(self._commands[2])
        total_command = cmd_lin_norm + cmd_ang_norm
        cmd_active = float(total_command > cfg["command_threshold"])

        # ------------------------------------------------------------
        # 1. Track linear velocity (weight > 0)
        #    exp(-(xy_error + 2*z_error) / sigma)
        #    Penalizes z velocity 2x to discourage bouncing.
        # ------------------------------------------------------------
        xy_error = np.sum((self._commands[:2] - base_lin_vel_body[:2]) ** 2)
        z_error = base_lin_vel_body[2] ** 2
        lin_vel_error = xy_error + 2.0 * z_error
        track_lin_vel = np.exp(-lin_vel_error / cfg["tracking_sigma"])

        # ------------------------------------------------------------
        # 2. Track angular velocity (weight > 0)
        #    exp(-(z_error + 0.05*xy_error) / sigma)
        #    Mildly penalizes xy angular velocity to discourage rocking.
        # ------------------------------------------------------------
        z_ang_error = (self._commands[2] - base_ang_vel_body[2]) ** 2
        xy_ang_error = np.sum(base_ang_vel_body[:2] ** 2)
        ang_vel_error = z_ang_error + 0.05 * xy_ang_error
        track_ang_vel = np.exp(-ang_vel_error / cfg["tracking_sigma"])

        # ------------------------------------------------------------
        # 2b. Linear forward velocity reward (weight > 0)
        #     Projects body velocity onto command direction. Provides
        #     constant gradient toward the commanded velocity, useful
        #     for off-policy algos to escape the standing-still local optimum.
        #     Clipped at command magnitude to prevent overshooting.
        # ------------------------------------------------------------
        cmd_xy = self._commands[:2]
        cmd_speed = np.linalg.norm(cmd_xy)
        if cmd_speed > 0.1:
            cmd_dir = cmd_xy / cmd_speed
            vel_proj = np.dot(base_lin_vel_body[:2], cmd_dir)
            lin_vel_forward_reward = np.clip(vel_proj, 0.0, cmd_speed)
        else:
            lin_vel_forward_reward = 0.0
        lateral_vel_penalty = (base_lin_vel_body[1] ** 2) * cmd_active

        # ------------------------------------------------------------
        # 2c. Angular forward velocity reward (weight > 0)
        #     Projects yaw rate onto command sign direction.
        #     Clipped at command magnitude.
        # ------------------------------------------------------------
        cmd_wz = self._commands[2]
        if abs(cmd_wz) > 0.1:
            wz_proj = base_ang_vel_body[2] * np.sign(cmd_wz)
            ang_vel_forward_reward = np.clip(wz_proj, 0.0, abs(cmd_wz))
        else:
            ang_vel_forward_reward = 0.0

        # ------------------------------------------------------------
        # 3. Flat orientation L2 penalty (weight < 0)
        #    sum(projected_gravity[:2]^2): non-zero when tilted.
        # ------------------------------------------------------------
        gravity_body = self._projected_gravity()
        flat_orientation = np.sum(gravity_body[:2] ** 2)

        # ------------------------------------------------------------
        # 4. Variable posture reward (weight > 0)
        #    exp(-mean(error^2 / std^2)) with speed-dependent stds.
        #    Tighter tolerance when standing, looser when walking/running.
        # ------------------------------------------------------------
        if total_command < cfg["posture_walking_threshold"]:
            std = self._posture_std_standing
        elif total_command < cfg["posture_running_threshold"]:
            std = self._posture_std_walking
        else:
            std = self._posture_std_running

        joint_pos_error = self.mjData.qpos[7:] - self.default_joint_pos
        pose_reward = np.exp(-np.mean(joint_pos_error ** 2 / (std ** 2)))

        # ------------------------------------------------------------
        # 4b. Track base height (weight > 0)
        #     Prevents low crouched gaits that satisfy velocity tracking with
        #     knees close to the ground.
        # ------------------------------------------------------------
        base_height_error = (self.mjData.qpos[2] - cfg["base_height_target"]) ** 2
        track_base_height = np.exp(-base_height_error / cfg["base_height_sigma"])

        # ------------------------------------------------------------
        # 5. Body angular velocity penalty (weight < 0)
        #    sum(ang_vel_world_xy^2): penalizes rocking in world frame.
        # ------------------------------------------------------------
        quat_wxyz = self.mjData.qpos[3:7]
        quat_xyzw = np.roll(quat_wxyz, -1)
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        ang_vel_world = R @ base_ang_vel_body
        body_ang_vel_penalty = np.sum(ang_vel_world[:2] ** 2)
        base_euler = Rotation.from_matrix(R).as_euler("xyz")
        pitch_tilt_penalty = base_euler[1] ** 2

        # ------------------------------------------------------------
        # 6. Angular momentum penalty (weight < 0)
        #    sum(angmom^2): whole-body angular momentum magnitude.
        # ------------------------------------------------------------
        angmom = self.mjData.subtree_angmom[self._base_body_id].copy()
        angular_momentum_penalty = np.sum(angmom ** 2)

        # ------------------------------------------------------------
        # 7. Termination penalty (weight < 0)
        #    Binary: 1.0 if terminated, 0.0 otherwise.
        # ------------------------------------------------------------
        termination_cost = 1.0 if terminated else 0.0

        # ------------------------------------------------------------
        # 8. Joint acceleration L2 penalty (weight < 0)
        # ------------------------------------------------------------
        joint_acc_penalty = np.sum(self._joint_acc ** 2)

        # ------------------------------------------------------------
        # 9. Joint position limits penalty (weight < 0)
        #    One-sided soft limit: penalizes joints beyond 95% of range.
        # ------------------------------------------------------------
        joint_pos = self.mjData.qpos[7:].copy()
        below_lower = np.clip(
            self._soft_joint_lower - joint_pos, a_min=0.0, a_max=None
        )
        above_upper = np.clip(
            joint_pos - self._soft_joint_upper, a_min=0.0, a_max=None
        )
        joint_pos_limits_penalty = np.sum(below_lower + above_upper)

        # ------------------------------------------------------------
        # 10. Action rate L2 penalty (weight < 0)
        #     Penalizes change in raw actions between steps.
        # ------------------------------------------------------------
        action_rate_penalty = np.sum(
            (action - self._prev_last_action) ** 2
        )

        # ------------------------------------------------------------
        # 11. Feet air time reward (weight > 0)
        #     Encourages a trotting gait: exactly 2 feet in contact
        #     (single stance) with mode time close to threshold.
        # ------------------------------------------------------------
        threshold = cfg["feet_air_time_threshold"]
        in_contact = self._feet_contact_time > 0
        in_mode_time = np.where(
            in_contact, self._feet_contact_time, self._feet_air_time
        )
        # Trotting pattern: exactly 2 of 4 feet in contact
        single_stance = np.mean(in_contact.astype(np.float64)) == 0.5
        if single_stance:
            mode_time = np.min(in_mode_time)
        else:
            mode_time = 0.0
        error = abs(mode_time - threshold)
        feet_air_time_reward = max(threshold - error, 0.0)
        feet_air_time_reward *= cmd_active

        # ------------------------------------------------------------
        # 11b. Scheduled diagonal gait reward (weight > 0)
        #      Unitree MJLab uses a phase-based foot_gait term for Go2:
        #      FR+RL in stance together, then FL+RR half a cycle later.
        #      The period is matched to the observed good 0.5 m/s trot
        #      timing from reward_shaping_progress.md.
        # ------------------------------------------------------------
        foot_gait_reward = 0.0
        bad_two_foot_contacts = 0.0
        gait_foot_names = ("FL", "FR", "RL", "RR")
        if cmd_active and all(name in self._foot_name_to_idx for name in gait_foot_names):
            phase = (
                (self._step_count * self.control_dt)
                / cfg["foot_gait_period"]
            ) % 1.0
            offsets = np.zeros(self._num_feet, dtype=np.float64)
            offsets[self._foot_name_to_idx["FR"]] = 0.0
            offsets[self._foot_name_to_idx["RL"]] = 0.0
            offsets[self._foot_name_to_idx["FL"]] = 0.5
            offsets[self._foot_name_to_idx["RR"]] = 0.5
            scheduled_stance = (
                (phase + offsets) % 1.0
            ) < cfg["foot_gait_stance_fraction"]
            gait_match = np.mean(scheduled_stance == in_contact)
            # Raw match gives 0.5 when all feet are planted because two feet are
            # scheduled for stance. Do not pay that standing local optimum.
            foot_gait_reward = max(2.0 * (gait_match - 0.5), 0.0)

            fl = in_contact[self._foot_name_to_idx["FL"]]
            fr = in_contact[self._foot_name_to_idx["FR"]]
            rl = in_contact[self._foot_name_to_idx["RL"]]
            rr = in_contact[self._foot_name_to_idx["RR"]]
            diagonal_support = (
                (fl and rr and not fr and not rl)
                or (fr and rl and not fl and not rr)
            )
            exactly_two_contacts = np.count_nonzero(in_contact) == 2
            bad_two_foot_contacts = float(
                exactly_two_contacts and not diagonal_support
            )

        # ------------------------------------------------------------
        # 12. Feet clearance penalty (weight < 0)
        #     Penalizes foot height deviation from target, weighted
        #     by foot xy velocity (only moving feet contribute).
        # ------------------------------------------------------------
        foot_positions = self._get_foot_positions()
        foot_velocities = self._get_foot_velocities()
        foot_z = foot_positions[:, 2]
        foot_vel_xy = foot_velocities[:, :2]
        vel_norm = np.linalg.norm(foot_vel_xy, axis=1)
        delta = np.abs(foot_z - cfg["foot_clearance_target"])
        feet_clearance_penalty = np.sum(delta * vel_norm) * cmd_active

        # ------------------------------------------------------------
        # 13. Feet slip penalty (weight < 0)
        #     Penalizes foot xy velocity while in contact with ground.
        # ------------------------------------------------------------
        in_contact_float = in_contact.astype(np.float64)
        vel_xy_norm_sq = np.sum(foot_vel_xy ** 2, axis=1)
        feet_slip_penalty = np.sum(vel_xy_norm_sq * in_contact_float)
        feet_slip_penalty *= cmd_active

        # ------------------------------------------------------------
        # 14. Soft landing penalty (weight < 0)
        #     Penalizes contact force magnitude at first contact
        #     (air→ground transition) to encourage soft footfalls.
        # ------------------------------------------------------------
        foot_forces = self._get_foot_contact_forces()
        landing_impact = np.sum(
            foot_forces * self._first_contact.astype(np.float64)
        )
        soft_landing_penalty = landing_impact * cmd_active

        # -- Combine all terms --
        reward = (
            cfg["w_track_lin_vel"] * track_lin_vel
            + cfg["w_track_ang_vel"] * track_ang_vel
            + cfg["w_lin_vel_forward"] * lin_vel_forward_reward
            + cfg["w_ang_vel_forward"] * ang_vel_forward_reward
            + cfg["w_lateral_vel"] * lateral_vel_penalty
            + cfg["w_alive"] * 1.0
            + cfg["w_flat_orientation"] * flat_orientation
            + cfg["w_pose"] * pose_reward
            + cfg["w_track_base_height"] * track_base_height
            + cfg["w_body_ang_vel"] * body_ang_vel_penalty
            + cfg["w_pitch_tilt"] * pitch_tilt_penalty
            + cfg["w_angular_momentum"] * angular_momentum_penalty
            + cfg["w_is_terminated"] * termination_cost
            + cfg["w_joint_acc"] * joint_acc_penalty
            + cfg["w_joint_pos_limits"] * joint_pos_limits_penalty
            + cfg["w_action_rate"] * action_rate_penalty
            + cfg["w_feet_air_time"] * feet_air_time_reward
            + cfg["w_foot_gait"] * foot_gait_reward
            + cfg["w_bad_two_foot_contacts"] * bad_two_foot_contacts
            + cfg["w_feet_clearance"] * feet_clearance_penalty
            + cfg["w_feet_slip"] * feet_slip_penalty
            + cfg["w_soft_landing"] * soft_landing_penalty
        )

        # Store individual reward terms for logging/debugging
        self._reward_components = {
            "track_lin_vel": cfg["w_track_lin_vel"] * track_lin_vel,
            "track_ang_vel": cfg["w_track_ang_vel"] * track_ang_vel,
            "lin_vel_forward": cfg["w_lin_vel_forward"] * lin_vel_forward_reward,
            "ang_vel_forward": cfg["w_ang_vel_forward"] * ang_vel_forward_reward,
            "lateral_vel": cfg["w_lateral_vel"] * lateral_vel_penalty,
            "alive": cfg["w_alive"] * 1.0,
            "flat_orientation": cfg["w_flat_orientation"] * flat_orientation,
            "pose": cfg["w_pose"] * pose_reward,
            "track_base_height": cfg["w_track_base_height"] * track_base_height,
            "body_ang_vel": cfg["w_body_ang_vel"] * body_ang_vel_penalty,
            "pitch_tilt": cfg["w_pitch_tilt"] * pitch_tilt_penalty,
            "angular_momentum": cfg["w_angular_momentum"] * angular_momentum_penalty,
            "is_terminated": cfg["w_is_terminated"] * termination_cost,
            "joint_acc": cfg["w_joint_acc"] * joint_acc_penalty,
            "joint_pos_limits": cfg["w_joint_pos_limits"] * joint_pos_limits_penalty,
            "action_rate": cfg["w_action_rate"] * action_rate_penalty,
            "feet_air_time": cfg["w_feet_air_time"] * feet_air_time_reward,
            "foot_gait": cfg["w_foot_gait"] * foot_gait_reward,
            "bad_two_foot_contacts": (
                cfg["w_bad_two_foot_contacts"] * bad_two_foot_contacts
            ),
            "feet_clearance": cfg["w_feet_clearance"] * feet_clearance_penalty,
            "feet_slip": cfg["w_feet_slip"] * feet_slip_penalty,
            "soft_landing": cfg["w_soft_landing"] * soft_landing_penalty,
        }

        if cfg.get("only_positive_rewards", False):
            reward = max(reward, 0.0)

        return float(reward)

    def _compute_simple_reward(self, action: np.ndarray, terminated: bool) -> float:
        """Simplified reward for MPC-injection training (SAC-MPC/TD3-MPC).

        Focuses on the core velocity tracking objective with minimal shaping
        to provide a cleaner learning signal that aligns with MPC demonstrations.

        Terms:
            - track_lin_vel: Exponential tracking of commanded xy velocity
            - track_ang_vel: Exponential tracking of commanded yaw rate
            - is_terminated: Large penalty for falling
        """
        cfg = self._SIMPLE_REWARD_CFG

        # -- Ground truth velocities (simulation only) --
        base_lin_vel_body = self._base_lin_vel_body()
        base_ang_vel_body = self.mjData.qvel[3:6].copy()

        # -- Linear velocity tracking --
        xy_error = np.sum((self._commands[:2] - base_lin_vel_body[:2]) ** 2)
        z_error = base_lin_vel_body[2] ** 2
        lin_vel_error = xy_error + 2.0 * z_error
        track_lin_vel = np.exp(-lin_vel_error / cfg["tracking_sigma"])

        # -- Angular velocity tracking --
        z_ang_error = (self._commands[2] - base_ang_vel_body[2]) ** 2
        xy_ang_error = np.sum(base_ang_vel_body[:2] ** 2)
        ang_vel_error = z_ang_error + 0.05 * xy_ang_error
        track_ang_vel = np.exp(-ang_vel_error / cfg["tracking_sigma"])

        # -- Linear forward velocity reward --
        cmd_xy = self._commands[:2]
        cmd_speed = np.linalg.norm(cmd_xy)
        if cmd_speed > 0.1:
            cmd_dir = cmd_xy / cmd_speed
            vel_proj = np.dot(base_lin_vel_body[:2], cmd_dir)
            lin_vel_forward_reward = np.clip(vel_proj, 0.0, cmd_speed)
        else:
            lin_vel_forward_reward = 0.0

        # -- Angular forward velocity reward --
        cmd_wz = self._commands[2]
        if abs(cmd_wz) > 0.1:
            wz_proj = base_ang_vel_body[2] * np.sign(cmd_wz)
            ang_vel_forward_reward = np.clip(wz_proj, 0.0, abs(cmd_wz))
        else:
            ang_vel_forward_reward = 0.0

        # -- Termination penalty --
        termination_cost = 1.0 if terminated else 0.0

        # -- Joint acceleration L2 penalty --
        joint_acc_penalty = np.sum(self._joint_acc ** 2)

        # -- Action rate L2 penalty --
        action_rate_penalty = np.sum(
            (action - self._prev_last_action) ** 2
        )

        reward = (
            cfg["w_track_lin_vel"] * track_lin_vel
            #+ cfg["w_track_ang_vel"] * track_ang_vel
            + cfg["w_lin_vel_forward"] * lin_vel_forward_reward
            #+ cfg["w_ang_vel_forward"] * ang_vel_forward_reward
            + cfg["w_is_terminated"] * termination_cost
            #+ cfg["w_joint_acc"] * joint_acc_penalty
            #+ cfg["w_action_rate"] * action_rate_penalty
        )

        # Store reward components for logging
        self._reward_components = {
            "track_lin_vel": cfg["w_track_lin_vel"] * track_lin_vel,
            "track_ang_vel": cfg["w_track_ang_vel"] * track_ang_vel,
            "lin_vel_forward": cfg["w_lin_vel_forward"] * lin_vel_forward_reward,
            "ang_vel_forward": cfg["w_ang_vel_forward"] * ang_vel_forward_reward,
            "is_terminated": cfg["w_is_terminated"] * termination_cost,
            "joint_acc": cfg["w_joint_acc"] * joint_acc_penalty,
            "action_rate": cfg["w_action_rate"] * action_rate_penalty,
        }

        if cfg.get("only_positive_rewards", False):
            reward = max(reward, 0.0)

        return float(reward)

    def _check_termination(self) -> bool:
        """Check if the episode should terminate (robot fell / bad state)."""
        # Get base orientation (roll, pitch, yaw)
        quat_wxyz = self.mjData.qpos[3:7]
        quat_xyzw = np.roll(quat_wxyz, -1)
        euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
        roll, pitch = euler[0], euler[1]

        # Check orientation limits
        if abs(roll) > self.max_roll or abs(pitch) > self.max_pitch:
            return True

        # Check base height
        base_height = self.mjData.qpos[2]
        if base_height < self.min_base_height:
            return True

        return False

    def _sample_commands(self):
        """Sample new velocity commands.

        Small xy velocity commands (< 0.2 m/s norm) are zeroed out to create
        a clear stand/move distinction for the feet_air_time reward.
        This follows the Legged Gym convention.
        """
        self._commands[0] = self.np_random.uniform(*self.lin_vel_x_range)
        self._commands[1] = self.np_random.uniform(*self.lin_vel_y_range)
        self._commands[2] = self.np_random.uniform(*self.ang_vel_z_range)

        # Zero small xy commands so feet_air_time reward has a clean threshold
        # between "stand" and "walk" commands
        #if np.linalg.norm(self._commands[:2]) < 0.2:
        #    self._commands[:2] = 0.0

        self._steps_since_command_resample = 0

    # =====================================================================
    # Domain Randomization
    # =====================================================================

    def restore_nominal_startup_domain_rand_state(self):
        """Restore the saved nominal startup-randomization state."""
        self.mjModel.geom_friction[:] = self._nominal_friction
        self.mjModel.body_mass[:] = self._nominal_body_mass
        self.mjModel.body_ipos[:] = self._nominal_body_ipos
        self.mjModel.dof_damping[:] = self._nominal_dof_damping
        self.mjModel.dof_armature[:] = self._nominal_dof_armature
        self.mjModel.dof_frictionloss[:] = self._nominal_dof_frictionloss
        self.mjModel.actuator_ctrlrange[:] = self._nominal_torque_limits
        self.mjModel.actuator_forcerange[:] = self._nominal_actuator_forcerange
        self.torque_limits = self._nominal_torque_limits.copy()
        self.kp = self._nominal_kp.copy()
        self.kd = self._nominal_kd.copy()
        self._encoder_bias[:] = 0.0
        self._motor_strength_scale = 1.0
        mujoco.mj_forward(self.mjModel, self.mjData)

    def sample_startup_domain_rand_bundle(
        self,
        *,
        rng: np.random.RandomState,
        dr_config_type: str,
        dr_seed: int,
    ) -> dict:
        """Sample one deterministic startup-DR bundle from the env nominal state."""
        self.restore_nominal_startup_domain_rand_state()
        return sample_startup_domain_rand_patch(
            self.mjModel,
            self.domain_rand_cfg,
            rng=rng,
            dr_config_type=dr_config_type,
            dr_seed=dr_seed,
            base_body_id=self._base_body_id,
            nominal_kp=self._nominal_kp,
            nominal_kd=self._nominal_kd,
            num_joints=self.num_joints,
        )

    def apply_startup_domain_rand_bundle(self, patch: dict | None):
        """Apply a sampled startup-DR bundle, including wrapper-side terms."""
        self.restore_nominal_startup_domain_rand_state()
        if patch is None:
            self._startup_domain_rand_config_type = "disabled"
            self._startup_domain_rand_seed = -1
            self._startup_domain_rand_patch = self.export_startup_domain_rand_patch(
                dr_config_type=self._startup_domain_rand_config_type,
                dr_seed=self._startup_domain_rand_seed,
            )
            return

        self.torque_limits = apply_startup_domain_rand_patch_to_model(
            self.mjModel,
            self.mjData,
            patch,
        )
        if "dr_encoder_bias" in patch:
            self._encoder_bias = np.array(patch["dr_encoder_bias"], copy=True, dtype=np.float64)
        if "dr_realized_kp" in patch:
            self.kp = np.array(patch["dr_realized_kp"], copy=True, dtype=np.float64)
        if "dr_realized_kd" in patch:
            self.kd = np.array(patch["dr_realized_kd"], copy=True, dtype=np.float64)
        if "dr_motor_strength_scale" in patch:
            self._motor_strength_scale = float(np.array(patch["dr_motor_strength_scale"]).item())

        mujoco.mj_forward(self.mjModel, self.mjData)
        self._startup_domain_rand_config_type = str(
            patch.get("dr_config_type", self._startup_domain_rand_config_type)
        )
        self._startup_domain_rand_seed = int(
            np.array(patch.get("dr_seed", self._startup_domain_rand_seed)).item()
        )
        self._startup_domain_rand_patch = self.export_startup_domain_rand_patch(
            dr_config_type=self._startup_domain_rand_config_type,
            dr_seed=self._startup_domain_rand_seed,
        )

    def export_startup_domain_rand_patch(
        self,
        *,
        dr_config_type: str | None = None,
        dr_seed: int | None = None,
    ) -> dict:
        """Export the realized startup-DR bundle from the live env."""
        applied_fields: list[str] = []

        def _mark_if_changed(field_name: str, current: np.ndarray, nominal: np.ndarray):
            if not np.allclose(current, nominal):
                applied_fields.append(field_name)

        _mark_if_changed("geom_friction", self.mjModel.geom_friction, self._nominal_friction)
        _mark_if_changed("body_mass", self.mjModel.body_mass, self._nominal_body_mass)
        _mark_if_changed("body_ipos", self.mjModel.body_ipos, self._nominal_body_ipos)
        _mark_if_changed("dof_damping", self.mjModel.dof_damping, self._nominal_dof_damping)
        _mark_if_changed("dof_armature", self.mjModel.dof_armature, self._nominal_dof_armature)
        _mark_if_changed(
            "dof_frictionloss",
            self.mjModel.dof_frictionloss,
            self._nominal_dof_frictionloss,
        )
        _mark_if_changed("torque_limits", self.torque_limits, self._nominal_torque_limits)
        _mark_if_changed("kp", self.kp, self._nominal_kp)
        _mark_if_changed("kd", self.kd, self._nominal_kd)
        if not np.allclose(self._encoder_bias, 0.0):
            applied_fields.append("encoder_bias")

        added_mass_kg = float(
            self.mjModel.body_mass[self._base_body_id]
            - self._nominal_body_mass[self._base_body_id]
        )
        if dr_config_type is None:
            dr_config_type = self._startup_domain_rand_config_type
        if dr_seed is None:
            dr_seed = self._startup_domain_rand_seed

        return {
            "dr_enabled": bool(self.domain_rand_cfg.enable),
            "dr_config_type": str(dr_config_type),
            "dr_seed": int(dr_seed),
            "dr_applied_fields": np.array(applied_fields, dtype="<U32"),
            "dr_motor_strength_scale": float(self._motor_strength_scale),
            "dr_added_mass_kg": added_mass_kg,
            "dr_patch_geom_friction": self.mjModel.geom_friction.copy(),
            "dr_patch_body_mass": self.mjModel.body_mass.copy(),
            "dr_patch_body_ipos": self.mjModel.body_ipos.copy(),
            "dr_patch_dof_damping": self.mjModel.dof_damping.copy(),
            "dr_patch_dof_armature": self.mjModel.dof_armature.copy(),
            "dr_patch_dof_frictionloss": self.mjModel.dof_frictionloss.copy(),
            "dr_patch_actuator_ctrlrange": self.mjModel.actuator_ctrlrange.copy(),
            "dr_patch_actuator_forcerange": self.mjModel.actuator_forcerange.copy(),
            "dr_torque_limits": self.torque_limits.copy(),
            "dr_encoder_bias": self._encoder_bias.copy(),
            "dr_realized_kp": self.kp.copy(),
            "dr_realized_kd": self.kd.copy(),
        }

    def _apply_startup_randomization(self):
        """Apply persistent physics randomization once at env creation.

        Matches MjLab's "startup" event mode: physics parameters are sampled
        once and stay fixed for the lifetime of this env instance. Each
        parallel env (created by make_vec_env) gets different values because
        each has its own np_random seeded differently.

        This is critical for SAC's off-policy replay buffer — if physics
        re-randomize every episode, the same state-action pair produces
        different transitions across episodes, making Q-learning unstable.

        Randomized parameters (matching MjLab Go2 defaults):
            - Friction coefficients (tangential, absolute)
            - Base center-of-mass position (additive)
            - Encoder bias (persistent calibration error)
        Optional (disabled by default, enable via config):
            - Base body mass (added mass)
            - Joint damping, armature, frictionloss
            - PD controller gains (Kp, Kd)
            - Motor torque limits (motor strength)
        """
        dr = self.domain_rand_cfg
        if not dr.enable:
            self._startup_domain_rand_config_type = "disabled"
            self._startup_domain_rand_seed = -1
            self._startup_domain_rand_patch = self.export_startup_domain_rand_patch(
                dr_config_type=self._startup_domain_rand_config_type,
                dr_seed=self._startup_domain_rand_seed,
            )
            return

        patch = self.sample_startup_domain_rand_bundle(
            rng=self.np_random,
            dr_config_type=self._startup_domain_rand_config_type,
            dr_seed=self._startup_domain_rand_seed,
        )
        self.apply_startup_domain_rand_bundle(patch)

        log.info(
            "Startup DR applied: friction=%.3f, com_disp=%s, "
            "encoder_bias_range=[%.4f, %.4f]",
            self.mjModel.geom_friction[0, 0],
            self.mjModel.body_ipos[self._base_body_id]
            - self._nominal_body_ipos[self._base_body_id],
            self._encoder_bias.min(),
            self._encoder_bias.max(),
        )

    def _resample_push_interval(self):
        """Sample a new random push interval for this episode."""
        if self._deterministic_push_schedule is not None:
            self._push_interval_steps = 0
            return
        dr = self.domain_rand_cfg
        if dr.enable and dr.push_robots:
            lo, hi = dr.push_interval_range_s
            interval_s = self.np_random.uniform(lo, hi)
            self._push_interval_steps = max(
                1, int(interval_s / self.control_dt)
            )
        else:
            self._push_interval_steps = 0

    def _maybe_push_robot(self):
        """Apply a random 6-DOF velocity perturbation to the base at intervals.

        Simulates unexpected external pushes (bumps, wind, collisions) that
        the policy must recover from. Matches MjLab's push_by_setting_velocity
        with 6-DOF velocity kicks and randomized timing.
        """
        if self._deterministic_push_schedule is not None:
            control_step = self._step_count - 1
            delta = self._deterministic_push_schedule.get(control_step)
            if delta is not None:
                self.mjData.qvel[:6] += delta
            return

        dr = self.domain_rand_cfg
        if not dr.enable or not dr.push_robots:
            return

        self._steps_since_last_push += 1
        if self._steps_since_last_push < self._push_interval_steps:
            return

        self._steps_since_last_push = 0
        # Resample next push interval (random, per MjLab)
        self._resample_push_interval()

        # Apply 6-DOF velocity kicks matching MjLab's push config:
        # qvel[0:3] = linear velocity (x, y, z)
        # qvel[3:6] = angular velocity (roll, pitch, yaw)
        ranges = dr.push_velocity_ranges
        self.mjData.qvel[0] += self.np_random.uniform(*ranges.get("x", (0.0, 0.0)))
        self.mjData.qvel[1] += self.np_random.uniform(*ranges.get("y", (0.0, 0.0)))
        self.mjData.qvel[2] += self.np_random.uniform(*ranges.get("z", (0.0, 0.0)))
        self.mjData.qvel[3] += self.np_random.uniform(*ranges.get("roll", (0.0, 0.0)))
        self.mjData.qvel[4] += self.np_random.uniform(*ranges.get("pitch", (0.0, 0.0)))
        self.mjData.qvel[5] += self.np_random.uniform(*ranges.get("yaw", (0.0, 0.0)))

    def _get_info(self) -> dict:
        """Return info dictionary with useful debugging information."""
        base_lin_vel_body = self._base_lin_vel_body()
        info = {
            "step_count": self._step_count,
            "commands": self._commands.copy(),
            "base_lin_vel_body": base_lin_vel_body.copy(),
            "base_ang_vel_body": self.mjData.qvel[3:6].copy(),
            "base_height": float(self.mjData.qpos[2]),
            "raw_q_target": self._raw_q_target.copy(),
            "filtered_q_target": self._filtered_q_target.copy(),
            "applied_torques": self._applied_torques.copy(),
            "foot_contacts": self._last_foot_contacts.copy(),
            "feet_air_time": self._feet_air_time.copy(),
            "feet_contact_time": self._feet_contact_time.copy(),
            "swing_peak": self._swing_peak.copy(),
            "reward_components": self._reward_components.copy(),
        }
        if self.enable_substep_diagnostics:
            info["substep_diagnostics"] = (
                None
                if self._last_substep_diagnostics is None
                else {
                    key: value.copy()
                    for key, value in self._last_substep_diagnostics.items()
                }
            )
        return info

    @staticmethod
    def _default_reward_cfg() -> dict[str, float]:
        """Default reward configuration weights.

        Ported from the unitree_rl_mjlab velocity tracking task configuration
        for the Unitree Go2 robot. This reward formulation is proven to
        produce stable quadruped locomotion gaits with proper trotting.

        Reward terms and weights:
            Positive rewards (desired behavior):
                - track_lin_vel (4.0):     Exponential xy velocity tracking
                - track_ang_vel (2.5):     Exponential yaw rate tracking
                - lin_vel_forward (6.0):   Linear forward velocity
                - ang_vel_forward (1.0):   Linear angular velocity
                - alive (0.0):             Constant survival bonus
                - pose (0.42):             Speed-dependent default pose tracking
                - track_base_height (1.0): Exponential target base height tracking
                - feet_air_time (0.75):    Trotting gait encouragement
                - foot_gait (1.35):        Diagonal trot phase matching

            Penalties (undesired behavior):
                - flat_orientation (-0.7):  Body tilt
                - pitch_tilt (-2.0):        Forward/back body pitch tilt
                - lateral_vel (-1.0):       Sideways body velocity
                - body_ang_vel (-0.16):     Excessive body angular velocity
                - angular_momentum (-0.014): Whole-body angular momentum
                - is_terminated (-10.0):    Falling over
                - joint_acc (-3.0e-7):      Jerky joint motion
                - joint_pos_limits (-1.0):  Joints near limits
                - action_rate (-0.045):     Rapid action changes
                - feet_clearance (-1.0):    Incorrect swing foot height
                - feet_slip (-0.12):        Foot sliding during contact
                - soft_landing (-2e-4):     High impact forces at landing
                - bad_two_foot_contacts (-0.7): Bounding/pacing support
        """
        return {
            # -- Tracking rewards --
            # Exponential kernel: exp(-error / sigma) where sigma = std^2 = 0.25
            "tracking_sigma": 0.25,
            "w_track_lin_vel": 4.0,
            "w_track_ang_vel": 2.5,
            # -- Forward velocity rewards (linear, constant gradient) --
            # Critical for SAC to escape the standing-still local optimum.
            # Projects velocity onto command direction, clipped at cmd magnitude.
            "w_lin_vel_forward": 6.0,
            "w_ang_vel_forward": 1.0,
            # -- Direction keeping penalties --
            "w_lateral_vel": -1.0,
            # -- Alive bonus (constant per-step survival reward) --
            "w_alive": 0.0,
            # -- Orientation penalty --
            "w_flat_orientation": -0.7,
            # -- Variable posture reward --
            # Speed-dependent default pose tracking with per-joint-type stds
            "w_pose": 0.42,
            "w_track_base_height": 1.0,
            "base_height_target": 0.27,
            "base_height_sigma": 0.01,
            "posture_walking_threshold": 0.05,   # speed below this → standing
            "posture_running_threshold": 1.5,   # speed above this → running
            # -- Body angular velocity penalty (world frame, xy only) --
            "w_body_ang_vel": -0.16,
            "w_pitch_tilt": -2.0,
            # -- Angular momentum penalty (whole-body) --
            "w_angular_momentum": -0.014,
            # -- Termination penalty (large negative on fall) --
            "w_is_terminated": -10.0,
            # -- Joint acceleration L2 penalty --
            "w_joint_acc": -3.0e-7,
            # -- Joint position limits penalty (soft limits at 95% range) --
            "w_joint_pos_limits": -1.0,
            # -- Action rate L2 penalty --
            "w_action_rate": -0.045,
            # -- Feet air time reward (trotting gait) --
            "w_feet_air_time": 0.75,
            "feet_air_time_threshold": 0.245,   # target stance/swing duration (s)
            # -- Scheduled diagonal trot reward --
            # FR+RL stance alternates with FL+RR stance. The 0.52 s period gives
            # slightly longer stance/swing windows for less tip-toeing.
            "w_foot_gait": 1.35,
            "foot_gait_period": 0.52,
            "foot_gait_stance_fraction": 0.52,
            # -- Penalize exact two-foot non-diagonal support (bound/pace) --
            "w_bad_two_foot_contacts": -0.7,
            # -- Feet clearance penalty (target swing foot height) --
            "w_feet_clearance": -1.0,
            "foot_clearance_target": 0.07,    # meters
            # -- Feet slip penalty (no sliding during contact) --
            "w_feet_slip": -0.12,
            # -- Soft landing penalty (minimize impact forces) --
            "w_soft_landing": -2e-4,
            # -- Command threshold for actually walking --
            "command_threshold": 0.1,
            # -- Reward clipping --
            "only_positive_rewards": False,
        }
