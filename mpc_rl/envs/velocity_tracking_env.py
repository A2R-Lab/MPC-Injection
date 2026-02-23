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
        self._feet_air_time_reward = 0.0

        # -- Joint acceleration tracking (for smooth motion penalty) --
        # Penalizing acceleration (d^2q/dt^2) instead of velocity encourages
        # smooth motion without discouraging joint movement itself.
        self._last_joint_vel = np.zeros(self.num_joints, dtype=np.float64)
        self._joint_acc = np.zeros(self.num_joints, dtype=np.float64)

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

        # Update feet air time tracking and compute air time reward
        self._update_feet_air_time()

        # Compute joint acceleration for smooth-motion penalty
        joint_vel_current = self.mjData.qvel[6:].copy()
        self._joint_acc = (joint_vel_current - self._last_joint_vel) / self.control_dt
        self._last_joint_vel = joint_vel_current

        # Resample commands periodically (only during training, not when commands are set externally)
        if not self._fixed_commands and self._steps_since_command_resample >= self.command_resample_interval:
            self._sample_commands()

        # Compute observation, reward, termination
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = False  # Handled by gymnasium's max_episode_steps
        info = self._get_info()

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
        self._feet_air_time_reward = 0.0
        self._last_joint_vel = np.zeros(self.num_joints, dtype=np.float64)
        self._joint_acc = np.zeros(self.num_joints, dtype=np.float64)
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

    def _update_feet_air_time(self):
        """Update feet air time tracking and compute the air time reward.

        Follows the Legged Gym (ETH/NVIDIA) approach:
        1. Detect current foot contacts with ground
        2. Filter contacts (OR with previous to handle noisy reporting)
        3. On first ground contact: reward (air_time - threshold) per foot
        4. Increment air time for all feet
        5. Reset air time for grounded feet

        The reward is only active when velocity commands are non-zero,
        so the robot is not rewarded for stepping in place at zero command.
        """
        threshold = self.reward_cfg.get("feet_air_time_threshold", 0.4)

        # Current foot contacts
        contacts = self._get_foot_contacts()

        # Filter contacts (OR with last step to smooth noisy contact reporting)
        contact_filt = np.logical_or(contacts, self._last_foot_contacts)

        # Detect first contact: foot was in the air and just landed
        first_contact = (self._feet_air_time > 0.0) & contact_filt

        # Increment air time for ALL feet by one control step
        self._feet_air_time += self.control_dt

        # Reward on landing: (air_time - threshold) for feet that just touched down.
        # Positive when step duration > threshold, negative when too short.
        air_time_reward = float(
            np.sum((self._feet_air_time - threshold) * first_contact)
        )

        # Only reward stepping when velocity commands are non-zero
        cmd_xy_norm = np.linalg.norm(self._commands[:2])
        if cmd_xy_norm < 0.1:
            air_time_reward = 0.0

        self._feet_air_time_reward = air_time_reward

        # Reset air time for feet that are on the ground
        self._feet_air_time *= ~contact_filt

        # Store contacts for next step
        self._last_foot_contacts = contacts.copy()

    def _compute_reward(self) -> float:
        """Compute the reward for the current step.

        Reward design calibrated against Legged Gym (ETH/NVIDIA) and Isaac Lab
        Go2 configurations, adapted for off-policy SAC training.

        Rewards:
            - Linear velocity xy tracking (exponential kernel for fine-tuning)
            - Angular velocity z tracking (exponential kernel for fine-tuning)
            - Linear forward velocity (linear, clipped; provides gradient at all velocities)
            - Angular forward velocity (linear, clipped; provides gradient at all ang vels)
            - Feet air time (encourages stepping when velocity is commanded)
            - Alive bonus (constant reward for not falling)

        Penalties:
            - Vertical base velocity (discourages bouncing)
            - Roll/pitch angular velocity (discourages rocking)
            - Orientation deviation from upright (disabled by default)
            - Joint torques (energy efficiency)
            - Action rate (smoothness)
            - Joint acceleration (smooth motion without penalizing motion itself)

        The exponential tracking rewards peak at the target velocity but have
        near-zero gradient far from the target (standing gives exp(-4)=0.018
        for vx=1.0). The linear forward rewards compensate by providing
        constant gradient: every velocity increment toward the command is
        proportionally rewarded, which is critical for SAC to escape the
        standing-still local optimum.
        """
        cfg = self.reward_cfg

        # -- Ground truth velocities (simulation only) --
        base_lin_vel_body = self._base_lin_vel_body()
        base_ang_vel_body = self.mjData.qvel[3:6]  # body frame

        # -- Tracking rewards --
        # Linear velocity tracking in xy plane
        lin_vel_error_sq = np.sum(
            (self._commands[:2] - base_lin_vel_body[:2]) ** 2
        )
        lin_vel_reward = np.exp(-lin_vel_error_sq / cfg["tracking_sigma"])

        # Angular velocity tracking around z axis
        ang_vel_error_sq = (self._commands[2] - base_ang_vel_body[2]) ** 2
        ang_vel_reward = np.exp(-ang_vel_error_sq / cfg["tracking_sigma"])

        # Feet air time reward (computed in _update_feet_air_time)
        feet_air_time_reward = self._feet_air_time_reward

        # -- Forward velocity rewards (linear) --
        # The exponential tracking kernel has near-zero gradient at large errors:
        # standing still with vx=1.0 gives exp(-4)=0.018, gradient=0.15.
        # After propagating through dynamics, SAC's Q-function can't detect this.
        # These linear terms provide CONSTANT gradient: every increment of
        # velocity toward the command is immediately and proportionally rewarded.
        # Clipped at command magnitude to avoid rewarding overshooting.
        cmd_xy = self._commands[:2]
        cmd_speed = np.linalg.norm(cmd_xy)
        if cmd_speed > 0.1:
            cmd_dir = cmd_xy / cmd_speed
            vel_proj = np.dot(base_lin_vel_body[:2], cmd_dir)
            lin_vel_forward_reward = np.clip(vel_proj, 0.0, cmd_speed)
        else:
            lin_vel_forward_reward = 0.0

        cmd_wz = self._commands[2]
        if abs(cmd_wz) > 0.1:
            wz_proj = base_ang_vel_body[2] * np.sign(cmd_wz)
            ang_vel_forward_reward = np.clip(wz_proj, 0.0, abs(cmd_wz))
        else:
            ang_vel_forward_reward = 0.0

        # -- Penalties --
        # Penalize vertical base velocity (discourages bouncing/jumping)
        lin_vel_z_penalty = base_lin_vel_body[2] ** 2

        # Penalize roll and pitch angular velocity (discourages rocking)
        ang_vel_xy_penalty = np.sum(base_ang_vel_body[:2] ** 2)

        # Penalize orientation deviation from upright
        gravity_body = self._projected_gravity()
        orientation_penalty = np.sum(gravity_body[:2] ** 2)

        # Penalize joint torques (energy efficiency)
        torque_penalty = np.sum(self._applied_torques ** 2)

        # Penalize action rate (smoothness)
        action_rate_penalty = np.sum(
            (self._last_action - self._prev_last_action) ** 2
        )

        # Penalize joint acceleration (encourages smooth motion without
        # discouraging joint movement itself, unlike joint velocity penalty)
        joint_acc_penalty = np.sum(self._joint_acc ** 2)

        # -- Combine --
        reward = (
            # Exponential tracking (precise fine-tuning once locomotion is found)
            cfg["w_lin_vel_tracking"] * lin_vel_reward
            + cfg["w_ang_vel_tracking"] * ang_vel_reward
            # Linear forward rewards (provides gradient at all velocities)
            + cfg["w_lin_vel_forward"] * lin_vel_forward_reward
            + cfg["w_ang_vel_forward"] * ang_vel_forward_reward
            # Locomotion shaping
            + cfg["w_feet_air_time"] * feet_air_time_reward
            # Alive bonus (constant per-step reward for not falling)
            + cfg["w_alive"] * 1.0
            # Penalties
            + cfg["w_lin_vel_z"] * lin_vel_z_penalty
            + cfg["w_ang_vel_xy"] * ang_vel_xy_penalty
            + cfg["w_orientation"] * orientation_penalty
            + cfg["w_torques"] * torque_penalty
            + cfg["w_action_rate"] * action_rate_penalty
            + cfg["w_joint_acc"] * joint_acc_penalty
        )

        # Clip reward to be non-negative. Useful for PPO to prevent termination
        # spirals. For SAC, set to False to allow negative rewards which provide
        # stronger signal for the Q-function.
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
            "feet_air_time_reward": self._feet_air_time_reward,
        }

    @staticmethod
    def _default_reward_cfg() -> dict[str, float]:
        """Default reward configuration weights.

        Calibrated against Legged Gym (ETH/NVIDIA) and Isaac Lab Go2 configs,
        adapted for off-policy SAC training.

        Key design principles:
            - Exponential tracking (w_lin/ang_vel_tracking) rewards precise
              velocity matching but has near-zero gradient far from the target.
            - Linear forward rewards (w_lin/ang_vel_forward) provide constant
              gradient at all velocities, critical for SAC to escape the
              standing-still local optimum. Clipped at command magnitude to
              prevent overshooting.
            - Alive bonus provides base reward for not falling.
            - only_positive_rewards=False for SAC (allows negative rewards
              which give the Q-function stronger signal). Set True for PPO.
        """
        return {
            # Tracking rewards (exponential kernel: exp(-error^2 / sigma))
            "w_lin_vel_tracking": 1.5,
            "w_ang_vel_tracking": 0.75,
            "tracking_sigma": 0.25,
            # Forward velocity rewards (linear, clipped at command magnitude)
            # Provides constant gradient unlike exponential which is flat at
            # large errors. Critical for SAC to discover locomotion.
            "w_lin_vel_forward": 2.0,
            "w_ang_vel_forward": 0.5,
            # Alive bonus (constant per-step reward for not falling)
            "w_alive": 0.5,
            # Locomotion shaping
            "w_feet_air_time": 0.25, # 0.25
            "feet_air_time_threshold": 0.6,  # seconds; target step duration # 0.4
            # Penalties (negative weights)
            "w_lin_vel_z": -2.0,
            "w_ang_vel_xy": -0.05,
            "w_orientation": 0.0,  # disabled: walking requires body pitch
            "w_torques": -2e-5,
            "w_action_rate": -0.01,
            "w_joint_acc": -2.5e-7,
            # Reward clipping: False for SAC (negative rewards = useful signal),
            # True for PPO (prevents termination spirals).
            "only_positive_rewards": False,
        }
