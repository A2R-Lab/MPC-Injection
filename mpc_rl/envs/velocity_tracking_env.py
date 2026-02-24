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
    action_applied = default_joint_pos + action_scale * action
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

from mpc_rl.envs.domain_randomization import DomainRandomizationConfig

log = logging.getLogger(__name__)


class QuadrupedVelocityTrackingEnv(gym.Env):
    """Gymnasium environment for training a quadruped to track commanded velocities.

    Uses only real-hardware-available observations (no base linear velocity).
    Actions are joint position residuals around the default pose, converted
    to torques via a PD controller.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        robot: str = "go2",
        scene: str = "flat",
        render_mode: str | None = None,
        # Simulation parameters
        # Physics at 200 Hz (sim_dt=0.005), control at 50 Hz (decimation=4)
        # to match the real Go2 robot's 50 Hz control loop.
        # Reference: Genesis Go2 env uses dt=0.02 control with substeps=2 (100 Hz physics).
        sim_dt: float = 0.005,
        decimation: int = 4,
        # PD controller gains
        kp: float = 40.0,
        kd: float = 0.5,
        action_scale: float = 0.25,
        # Command ranges
        lin_vel_x_range: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y_range: tuple[float, float] = (-0.5, 0.5),
        ang_vel_z_range: tuple[float, float] = (-1.0, 1.0),
        # Reward weights
        reward_cfg: dict[str, float] | None = None,
        # Termination thresholds
        max_pitch: float = 0.6,
        max_roll: float = 0.6,
        min_base_height: float = 0.1,
        # Episode settings
        command_resample_interval: int = 250, # og 500
        # Domain randomization
        domain_rand_cfg: DomainRandomizationConfig | None = None,
    ):
        """Initialize the velocity tracking environment.

        Args:
            robot: Robot model name (go2, go1, mini_cheetah, aliengo, etc.).
            scene: Terrain type (flat, perlin, random_boxes, etc.).
            render_mode: Gymnasium render mode ("human", "rgb_array", or None).
            sim_dt: MuJoCo physics timestep in seconds. Default 0.005 (200 Hz)
                for stable contact dynamics.
            decimation: Number of physics steps per control step.
                control_dt = sim_dt * decimation = 0.005 * 4 = 0.02s (50 Hz),
                matching the real Go2 robot's control frequency.
            kp: Proportional gain for PD controller.
            kd: Derivative gain for PD controller.
            action_scale: Scaling factor for action residuals (radians).
            lin_vel_x_range: Range for commanded x velocity (m/s).
            lin_vel_y_range: Range for commanded y velocity (m/s).
            ang_vel_z_range: Range for commanded yaw rate (rad/s).
            reward_cfg: Dictionary of reward weights. See _default_reward_cfg().
            max_pitch: Maximum pitch angle before termination (radians).
            max_roll: Maximum roll angle before termination (radians).
            min_base_height: Minimum base height before termination (meters).
            command_resample_interval: Resample velocity commands every N control steps.
        """
        super().__init__()

        # Store configuration
        self.robot_name = robot
        self.robot_cfg: RobotConfig = get_robot_config(robot_name=robot)
        self.render_mode = render_mode
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.control_dt = sim_dt * decimation

        # PD controller parameters
        self.kp = np.float64(kp)
        self.kd = np.float64(kd)
        self.action_scale = np.float64(action_scale)

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

        # Reward configuration
        self.reward_cfg = self._default_reward_cfg()
        if reward_cfg is not None:
            self.reward_cfg.update(reward_cfg)

        # ── Load MuJoCo model ────────────────────────────────────────────
        self._load_model(scene)

        # ── Extract default joint positions from keyframe ────────────────
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
        self._foot_geom_id_set = set(self._foot_geom_ids.values())
        self._num_feet = len(self._foot_geom_ids)

        # --- Store nominal model values for domain randomization --------
        # These are the "ground truth" XML values that randomization scales/offsets.
        # Stored once at init so reset() can always start from the nominal model.
        self._nominal_friction = self.mjModel.geom_friction.copy()
        self._nominal_body_mass = self.mjModel.body_mass.copy()
        self._nominal_body_ipos = self.mjModel.body_ipos.copy()
        self._nominal_dof_damping = self.mjModel.dof_damping.copy()
        self._nominal_dof_armature = self.mjModel.dof_armature.copy()
        self._nominal_dof_frictionloss = self.mjModel.dof_frictionloss.copy()
        # Store base body ID for mass/CoM randomization
        # Go2 XML uses "base" as the root body; fall back to body ID 1 (first
        # non-world body) if lookup fails.
        _base_id = mujoco.mj_name2id(
            self.mjModel, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self._base_body_id = _base_id if _base_id >= 0 else 1
        # Nominal PD gains (will be randomized per-episode)
        self._nominal_kp = float(self.kp)
        self._nominal_kd = float(self.kd)
        # Nominal torque limits (for motor strength randomization)
        self._nominal_torque_limits = self.torque_limits.copy()
        # Current motor strength scale (updated at reset)
        self._motor_strength_scale = 1.0

        # Perturbation tracking
        self._push_interval_steps = 0  # set in reset
        self._steps_since_last_push = 0

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

        # ── Define action space ──────────────────────────────────────────
        # Actions are joint position residuals in [-1, 1], scaled by action_scale
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float64,
        )

        # ── Internal state tracking ─────────────────────────────────────
        self._commands = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(self.num_joints, dtype=np.float64)
        self._prev_last_action = np.zeros(self.num_joints, dtype=np.float64)
        self._applied_torques = np.zeros(self.num_joints, dtype=np.float64)
        self._step_count = 0
        self._steps_since_command_resample = 0
        self._fixed_commands = False  # When True, step() will NOT auto-resample commands

        # -- Feet air time tracking (for locomotion reward) --
        # Tracks how long each foot has been in the air. On first ground
        # contact, the air time is used to compute a reward that encourages
        # stepping of appropriate duration (Legged Gym approach).
        self._feet_air_time = np.zeros(self._num_feet, dtype=np.float64)
        self._last_foot_contacts = np.zeros(self._num_feet, dtype=bool)

        # -- Swing peak tracking (for feet height reward) --
        # Tracks the maximum foot height during each swing phase.
        # On first contact, the peak is compared to the target height.
        self._swing_peak = np.zeros(self._num_feet, dtype=np.float64)
        self._first_contact = np.zeros(self._num_feet, dtype=bool)
        self._current_contacts = np.zeros(self._num_feet, dtype=bool)

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

        # ── Rendering ───────────────────────────────────────────────────
        self.viewer = None
        self._renderer = None  # For rgb_array mode

    # ═════════════════════════════════════════════════════════════════════
    # Gymnasium API
    # ═════════════════════════════════════════════════════════════════════

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
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Store previous action for action rate penalty
        self._prev_last_action = self._last_action.copy()
        self._last_action = action.copy()

        # Compute joint position targets
        q_target = self.default_joint_pos + self.action_scale * action

        # Apply PD control for `decimation` simulation steps
        for _ in range(self.decimation):
            q_current = self.mjData.qpos[7:]
            dq_current = self.mjData.qvel[6:]

            # PD controller: tau = Kp * (q_target - q) + Kd * (0 - dq)
            torques = self.kp * (q_target - q_current) + self.kd * (0.0 - dq_current)

            # Clip torques to actuator limits
            torques = np.clip(
                torques,
                self.torque_limits[:, 0],
                self.torque_limits[:, 1],
            )
            self._applied_torques = torques

            self.mjData.ctrl[:] = torques
            mujoco.mj_step(self.mjModel, self.mjData)

        self._step_count += 1
        self._steps_since_command_resample += 1

        # Apply random perturbation (push) periodically
        self._maybe_push_robot()

        # Update feet air time tracking (also updates swing peak & first_contact)
        self._update_feet_air_time()

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
        self._applied_torques = np.zeros(self.num_joints, dtype=np.float64)
        self._feet_air_time = np.zeros(self._num_feet, dtype=np.float64)
        self._last_foot_contacts = np.zeros(self._num_feet, dtype=bool)
        self._swing_peak = np.zeros(self._num_feet, dtype=np.float64)
        self._first_contact = np.zeros(self._num_feet, dtype=bool)
        self._current_contacts = np.zeros(self._num_feet, dtype=bool)
        self._step_count = 0
        self._steps_since_command_resample = 0

        # Apply domain randomization to physics parameters
        self._apply_domain_randomization()

        # Reset perturbation tracking
        if self.domain_rand_cfg.enable and self.domain_rand_cfg.push_robots:
            self._push_interval_steps = max(
                1, int(self.domain_rand_cfg.push_interval_s / self.control_dt)
            )
        self._steps_since_last_push = 0

        # Sample new velocity commands (only if not using externally fixed commands)
        if not self._fixed_commands:
            self._sample_commands()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

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
            self._renderer.update_scene(self.mjData)
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

    # ═════════════════════════════════════════════════════════════════════
    # Public API for external command control
    # ═════════════════════════════════════════════════════════════════════

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

    # ═════════════════════════════════════════════════════════════════════
    # Internal methods
    # ═════════════════════════════════════════════════════════════════════

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
        self.mjData = mujoco.MjData(self.mjModel)

        # Set simulation timestep
        self.mjModel.opt.timestep = self.sim_dt

        # Apply custom zero position if specified
        if self.robot_cfg.qpos0_js is not None:
            self.mjModel.qpos0[7:] = np.array(self.robot_cfg.qpos0_js)

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
        joint_pos_rel = self.mjData.qpos[7:].copy() - self.default_joint_pos

        # Joint velocities
        joint_vel = self.mjData.qvel[6:].copy()

        # Previous actions
        prev_actions = self._last_action.copy()

        # ── Apply observation noise for sim-to-real robustness ──────────
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
        """Update feet air time, swing peak, and contact tracking.

        Follows the MuJoCo Playground / Legged Gym approach:
        1. Detect current foot contacts with ground
        2. Filter contacts (OR with previous to handle noisy contact)
        3. Track first contact (foot was in air and just landed)
        4. Update swing peak (max foot z height during swing)
        5. Increment air time for all feet
        6. Reset air time for grounded feet
        """
        # Current foot contacts
        contacts = self._get_foot_contacts()

        # Filter contacts (OR with last step to smooth noisy contact reporting)
        contact_filt = np.logical_or(contacts, self._last_foot_contacts)

        # Detect first contact: foot was in the air and just landed
        first_contact = (self._feet_air_time > 0.0) & contact_filt

        # Increment air time for ALL feet by one control step
        self._feet_air_time += self.control_dt

        # Update swing peak: track max foot z during swing phase
        foot_positions = self._get_foot_positions()
        foot_z = foot_positions[:, 2]  # z heights
        self._swing_peak = np.maximum(self._swing_peak, foot_z)

        # Store for use in reward computation and swing_peak reset in step()
        self._first_contact = first_contact.copy()
        self._current_contacts = contacts.copy()

        # Reset air time for feet that are on the ground
        self._feet_air_time *= ~contact_filt

        # Store contacts for next step
        self._last_foot_contacts = contacts.copy()

    def _compute_reward(self, action: np.ndarray, terminated: bool) -> float:
        """Compute the reward for the current step.

        Hybrid reward design: combines MuJoCo Playground's foot-shaping costs
        with SAC-specific linear forward rewards and alive bonus.

        Tracking rewards (exponential kernel, peak at target):
            - tracking_lin_vel, tracking_ang_vel: Precise matching

        Linear forward rewards (constant gradient, critical for SAC):
            - lin_vel_forward: Proportional reward for moving toward command
            - ang_vel_forward: Proportional reward for turning toward command
            These provide the gradient SAC needs to break out of standing still.
            PPO with thousands of parallel envs can explore past this with the
            exponential tracking alone; SAC with few envs cannot.

        Behavioral rewards:
            - alive: Constant per-step survival bonus
            - pose: Stay near default joint configuration
            - feet_air_time: Encourage proper step duration

        Costs (Playground-inspired gait shaping):
            - lin_vel_z, ang_vel_xy, orientation: Base stability
            - torques, action_rate, energy: Regularization
            - feet_slip, feet_clearance, feet_height: Foot trajectory quality
            - stand_still, termination, dof_pos_limits: Safety
        """
        cfg = self.reward_cfg
        sigma = cfg["tracking_sigma"]

        # -- Ground truth velocities (simulation only) --
        base_lin_vel_body = self._base_lin_vel_body()
        base_ang_vel_body = self.mjData.qvel[3:6]  # body frame
        base_lin_vel_world = self.mjData.qvel[0:3]  # world frame
        base_ang_vel_world = self.mjData.qvel[3:6]  # Note: MuJoCo qvel[3:6] is body-frame
        qpos_joints = self.mjData.qpos[7:].copy()
        qvel_joints = self.mjData.qvel[6:].copy()
        torques = self._applied_torques
        cmd_norm = np.linalg.norm(self._commands)
        contacts = self._current_contacts
        first_contact = self._first_contact

        foot_positions = self._get_foot_positions()
        foot_velocities = self._get_foot_velocities()

        # ── Tracking rewards ────────────────────────────────────────────
        # Linear velocity tracking in xy plane (exponential kernel)
        lin_vel_error = np.sum((self._commands[:2] - base_lin_vel_body[:2]) ** 2)
        r_tracking_lin_vel = np.exp(-lin_vel_error / sigma)

        # Angular velocity tracking around z axis (exponential kernel)
        ang_vel_error = (self._commands[2] - base_ang_vel_body[2]) ** 2
        r_tracking_ang_vel = np.exp(-ang_vel_error / sigma)

        # ── Linear forward rewards (critical for SAC exploration) ─────────
        # The exponential tracking kernel has near-zero gradient at large errors
        # (standing still with vx=1.0 gives exp(-4)=0.018). SAC with few parallel
        # envs cannot detect this tiny signal through temporal-difference learning.
        # These linear terms provide CONSTANT, proportional gradient: every
        # increment of velocity toward the command is immediately rewarded.
        # Clipped at command magnitude to avoid rewarding overshooting.
        cmd_xy = self._commands[:2]
        cmd_speed = np.linalg.norm(cmd_xy)
        if cmd_speed > 0.1:
            cmd_dir = cmd_xy / cmd_speed
            vel_proj = np.dot(base_lin_vel_body[:2], cmd_dir)
            r_lin_vel_forward = np.clip(vel_proj, 0.0, cmd_speed)
        else:
            r_lin_vel_forward = 0.0

        cmd_wz = self._commands[2]
        if abs(cmd_wz) > 0.1:
            wz_proj = base_ang_vel_body[2] * np.sign(cmd_wz)
            r_ang_vel_forward = np.clip(wz_proj, 0.0, abs(cmd_wz))
        else:
            r_ang_vel_forward = 0.0

        # ── Behavioral rewards ──────────────────────────────────────────
        # Pose reward: stay close to default joint configuration.
        # Hip and thigh joints weighted 1.0, calf joints weighted 0.1.
        # Pattern: [hip, thigh, calf] x 4 legs = 12 joints
        pose_weight = np.array([1.0, 1.0, 0.1] * 4)
        r_pose = np.exp(-np.sum(pose_weight * (qpos_joints - self.default_joint_pos) ** 2))

        # Feet air time reward: encourage appropriate step duration
        threshold = cfg.get("feet_air_time_threshold", 0.1)
        rew_air_time = np.sum((self._feet_air_time - threshold) * first_contact)
        rew_air_time *= (cmd_norm > 0.01)  # No reward for zero commands
        r_feet_air_time = float(rew_air_time)

        # ── Costs (all return positive values, scaled by negative weights) ──
        # Penalize vertical base velocity (discourages bouncing)
        c_lin_vel_z = base_lin_vel_body[2] ** 2

        # Penalize roll and pitch angular velocity (discourages rocking)
        c_ang_vel_xy = np.sum(base_ang_vel_body[:2] ** 2)

        # Penalize non-upright orientation
        gravity_body = self._projected_gravity()
        c_orientation = np.sum(gravity_body[:2] ** 2)

        # Penalize torques: sqrt(sum(τ²)) + sum(|τ|) (MuJoCo Playground formulation)
        c_torques = np.sqrt(np.sum(torques ** 2)) + np.sum(np.abs(torques))

        # Penalize action rate (smoothness)
        c_action_rate = np.sum((action - self._prev_last_action) ** 2)

        # Penalize energy consumption: sum(|dq| * |τ|)
        c_energy = np.sum(np.abs(qvel_joints) * np.abs(torques))

        # Penalize foot slip: xy velocity² of feet in contact with ground
        foot_vel_xy = foot_velocities[:, :2]
        foot_vel_xy_sq = np.sum(foot_vel_xy ** 2, axis=1)
        c_feet_slip = float(np.sum(foot_vel_xy_sq * contacts) * (cmd_norm > 0.01))

        # Penalize foot clearance: deviation from target height during swing
        max_foot_height = cfg.get("max_foot_height", 0.1)
        foot_z = foot_positions[:, 2]
        foot_vel_xy_norm = np.sqrt(np.linalg.norm(foot_vel_xy, axis=1))
        clearance_delta = np.abs(foot_z - max_foot_height)
        c_feet_clearance = float(np.sum(clearance_delta * foot_vel_xy_norm))

        # Penalize swing peak not reaching target height
        peak_error = self._swing_peak / max_foot_height - 1.0
        c_feet_height = float(np.sum(peak_error ** 2 * first_contact) * (cmd_norm > 0.01))

        # Penalize joint deviation from default when commands are near zero
        c_stand_still = float(np.sum(np.abs(qpos_joints - self.default_joint_pos)) * (cmd_norm < 0.01))

        # Penalize early termination
        c_termination = float(terminated)

        # Penalize joints approaching limits (soft limits at 95% of range)
        out_of_limits = -np.clip(qpos_joints - self._soft_joint_lower, None, 0.0)
        out_of_limits += np.clip(qpos_joints - self._soft_joint_upper, 0.0, None)
        c_dof_pos_limits = float(np.sum(out_of_limits))

        # ── Combine: each term scaled by its weight ─────────────────────
        reward = (
            # Exponential tracking (precise matching once locomotion is found)
            cfg["w_tracking_lin_vel"] * r_tracking_lin_vel
            + cfg["w_tracking_ang_vel"] * r_tracking_ang_vel
            # Linear forward rewards (provides gradient at all velocities for SAC)
            + cfg["w_lin_vel_forward"] * r_lin_vel_forward
            + cfg["w_ang_vel_forward"] * r_ang_vel_forward
            # Alive bonus (constant per-step reward for not falling)
            + cfg["w_alive"] * 1.0
            # Behavioral shaping
            + cfg["w_pose"] * r_pose
            + cfg["w_feet_air_time"] * r_feet_air_time
            # Base penalties
            + cfg["w_lin_vel_z"] * c_lin_vel_z
            + cfg["w_ang_vel_xy"] * c_ang_vel_xy
            + cfg["w_orientation"] * c_orientation
            # Regularization penalties
            + cfg["w_torques"] * c_torques
            + cfg["w_action_rate"] * c_action_rate
            + cfg["w_energy"] * c_energy
            # Foot shaping penalties
            + cfg["w_feet_slip"] * c_feet_slip
            + cfg["w_feet_clearance"] * c_feet_clearance
            + cfg["w_feet_height"] * c_feet_height
            # Other penalties
            + cfg["w_stand_still"] * c_stand_still
            + cfg["w_termination"] * c_termination
            + cfg["w_dof_pos_limits"] * c_dof_pos_limits
        )

        # NOTE: MuJoCo Playground scales by dt and clips to [0, inf) — that
        # convention suits PPO with massive parallelism.  For SAC with few envs,
        # negative rewards are an essential learning signal (tells the Q-function
        # that falling / bad posture is worse than standing), and dt-scaling
        # shrinks the reward 50× which makes the entropy coefficient dominate.
        # We therefore skip both dt-scaling and non-negative clipping.

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
        if np.linalg.norm(self._commands[:2]) < 0.2:
            self._commands[:2] = 0.0

        self._steps_since_command_resample = 0

    # ═════════════════════════════════════════════════════════════════════
    # Domain Randomization
    # ═════════════════════════════════════════════════════════════════════

    def _apply_domain_randomization(self):
        """Randomize physics parameters at the start of each episode.

        Modifies mjModel fields in-place, starting from the stored nominal
        (XML) values to prevent drift across episodes. This is the standard
        approach used by Legged Gym, Isaac Lab, and mjlab.

        Randomized parameters:
            - Friction coefficients (tangential/torsional/rolling)
            - Base body mass (added mass simulating payload)
            - Base center-of-mass position (simulating payload offset)
            - Joint damping, armature, frictionloss
            - PD controller gains (Kp, Kd)
            - Motor torque limits (motor strength)
        """
        dr = self.domain_rand_cfg
        if not dr.enable:
            return

        rng = self.np_random

        # ── Friction randomization ──────────────────────────────────────
        # Scale tangential friction (column 0 of geom_friction) for all geoms.
        # Torsional (col 1) and rolling (col 2) friction are also scaled by
        # the same factor to maintain consistent contact behavior.
        friction_scale = rng.uniform(*dr.friction_range)
        self.mjModel.geom_friction[:] = self._nominal_friction * friction_scale

        # ── Base mass randomization (payload variation) ─────────────────
        added_mass = rng.uniform(*dr.added_mass_range)
        self.mjModel.body_mass[:] = self._nominal_body_mass.copy()
        self.mjModel.body_mass[self._base_body_id] += added_mass

        # ── Center-of-mass displacement ─────────────────────────────────
        com_disp = rng.uniform(
            dr.com_displacement_range[0],
            dr.com_displacement_range[1],
            size=3,
        )
        self.mjModel.body_ipos[:] = self._nominal_body_ipos.copy()
        self.mjModel.body_ipos[self._base_body_id] += com_disp

        # ── Joint damping randomization ─────────────────────────────────
        damping_scale = rng.uniform(*dr.joint_damping_scale_range)
        self.mjModel.dof_damping[:] = self._nominal_dof_damping * damping_scale

        # ── Joint armature (rotor inertia) randomization ────────────────
        armature_scale = rng.uniform(*dr.joint_armature_scale_range)
        self.mjModel.dof_armature[:] = self._nominal_dof_armature * armature_scale

        # ── Joint Coulomb friction randomization ────────────────────────
        # Unlike damping (viscous), frictionloss is set to absolute values
        # since the nominal value is often zero.
        joint_friction = rng.uniform(
            *dr.joint_friction_range,
            size=self.mjModel.dof_frictionloss.shape,
        )
        self.mjModel.dof_frictionloss[:] = joint_friction

        # ── PD gain randomization ───────────────────────────────────────
        kp_scale = rng.uniform(*dr.kp_scale_range)
        kd_scale = rng.uniform(*dr.kd_scale_range)
        self.kp = np.float64(self._nominal_kp * kp_scale)
        self.kd = np.float64(self._nominal_kd * kd_scale)

        # ── Motor strength (torque limit) randomization ─────────────────
        self._motor_strength_scale = rng.uniform(*dr.motor_strength_range)
        self.torque_limits = self._nominal_torque_limits * self._motor_strength_scale

    def _maybe_push_robot(self):
        """Apply a random velocity perturbation to the base at intervals.

        Simulates unexpected external pushes (bumps, wind, collisions) that
        the policy must recover from. Following IsaacGymEnvs' push_robots()
        and mjlab's push_by_setting_velocity(), this directly sets the base
        velocity to a random value.
        """
        dr = self.domain_rand_cfg
        if not dr.enable or not dr.push_robots:
            return

        self._steps_since_last_push += 1
        if self._steps_since_last_push < self._push_interval_steps:
            return

        self._steps_since_last_push = 0

        # Apply random linear velocity kick in xy plane
        self.mjData.qvel[0] += self.np_random.uniform(*dr.push_vel_xy_range)
        self.mjData.qvel[1] += self.np_random.uniform(*dr.push_vel_xy_range)

        # Apply random angular velocity kick around z
        self.mjData.qvel[5] += self.np_random.uniform(*dr.push_ang_vel_range)

    def _get_info(self) -> dict:
        """Return info dictionary with useful debugging information."""
        base_lin_vel_body = self._base_lin_vel_body()
        return {
            "step_count": self._step_count,
            "commands": self._commands.copy(),
            "base_lin_vel_body": base_lin_vel_body.copy(),
            "base_ang_vel_body": self.mjData.qvel[3:6].copy(),
            "base_height": float(self.mjData.qpos[2]),
            "applied_torques": self._applied_torques.copy(),
            "foot_contacts": self._last_foot_contacts.copy(),
            "feet_air_time": self._feet_air_time.copy(),
            "swing_peak": self._swing_peak.copy(),
        }

    @staticmethod
    def _default_reward_cfg() -> dict[str, float]:
        """Default reward configuration weights.

        Hybrid of the original SAC-tuned rewards and MuJoCo Playground
        Go1 joystick gait-shaping costs.

        Key design principles:
            - Exponential tracking rewards for precise velocity matching
            - Linear forward rewards provide constant gradient for SAC to
              escape the standing-still local optimum (not needed for PPO)
            - Alive bonus establishes a baseline reward
            - Playground-inspired foot costs (slip, clearance, height) shape
              proper gait quality instead of burst-walking
            - Moderate orientation penalty (-0.5) keeps robot upright without
              preventing natural body pitch during locomotion
        """
        return {
            # ── Tracking rewards (exponential kernel) ────────────────
            "w_tracking_lin_vel": 1.5,   # exp(-error²/sigma)
            "w_tracking_ang_vel": 0.75,  # exp(-error²/sigma)
            "tracking_sigma": 0.25,       # Kernel width
            # ── Linear forward rewards (SAC exploration) ────────────
            # Provides constant gradient unlike exponential which is flat
            # at large errors. Critical for SAC to discover locomotion.
            "w_lin_vel_forward": 2.0,    # Linear, clipped at cmd magnitude
            "w_ang_vel_forward": 0.5,    # Linear, clipped at cmd magnitude
            # ── Alive bonus ─────────────────────────────────────────
            "w_alive": 0.5,              # Constant per-step survival reward
            # ── Behavioral rewards (Playground-inspired) ────────────
            "w_pose": 0.5,               # Stay near default joint config
            "w_feet_air_time": 0.25,     # Reward appropriate step duration
            "feet_air_time_threshold": 0.25,  # seconds; longer = no rapid hopping
            # ── Base costs (negative weights) ───────────────────────
            "w_lin_vel_z": -4.0,         # Penalize vertical base velocity (anti-bounce)
            "w_ang_vel_xy": -0.5,        # Penalize roll/pitch angular vel (anti-rock)
            "w_orientation": -0.5,       # Moderate orientation penalty
            # ── Regularization costs ────────────────────────────────
            "w_torques": -0.0002,        # Penalize torques (Playground form)
            "w_action_rate": -0.01,      # Penalize jerky actions
            "w_energy": -0.001,          # Penalize energy consumption
            # ── Feet costs (shape proper gait) ──────────────────────
            "w_feet_clearance": -1.0,    # Foot height deviation from target
            "w_feet_height": -0.2,       # Swing peak not reaching target
            "w_feet_slip": -0.1,         # Foot sliding during contact
            "max_foot_height": 0.1,      # Target foot clearance (meters)
            # ── Other costs ─────────────────────────────────────────
            "w_stand_still": -0.5,       # Joint deviation at zero command
            "w_termination": -1.0,       # Early termination penalty
            "w_dof_pos_limits": -1.0,    # Joints approaching limits
        }
