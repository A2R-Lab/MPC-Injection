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
        command_resample_interval: int = 500,
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

        # ── Define observation space (Dict: asymmetric actor-critic) ─────
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
        self._step_count = 0
        self._steps_since_command_resample = 0

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

        NOT available on real hardware without estimation. Used only for
        computing the velocity tracking reward during training.
        """
        quat_wxyz = self.mjData.qpos[3:7]
        quat_xyzw = np.roll(quat_wxyz, -1)
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        return R.T @ self.mjData.qvel[0:3]

    def _compute_reward(self) -> float:
        """Compute the reward for the current step.

        Rewards:
            - Linear velocity xy tracking (exponential kernel)
            - Angular velocity z tracking (exponential kernel)

        Penalties:
            - Vertical base velocity (discourages bouncing)
            - Roll/pitch angular velocity (discourages rocking)
            - Orientation deviation from upright (projected gravity error)
            - Joint torques (energy efficiency)
            - Action rate (smoothness)
            - Joint velocities (smoothness)
        """
        cfg = self.reward_cfg

        # ── Ground truth velocities (simulation only) ────────────────────
        base_lin_vel_body = self._base_lin_vel_body()
        base_ang_vel_body = self.mjData.qvel[3:6]  # body frame

        # ── Tracking rewards ─────────────────────────────────────────────
        # Linear velocity tracking in xy plane
        lin_vel_error_sq = np.sum(
            (self._commands[:2] - base_lin_vel_body[:2]) ** 2
        )
        lin_vel_reward = np.exp(-lin_vel_error_sq / cfg["tracking_sigma"])

        # Angular velocity tracking around z axis
        ang_vel_error_sq = (self._commands[2] - base_ang_vel_body[2]) ** 2
        ang_vel_reward = np.exp(-ang_vel_error_sq / cfg["tracking_sigma"])

        # ── Penalties ────────────────────────────────────────────────────
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

        # Penalize joint velocities (smoothness)
        joint_vel = self.mjData.qvel[6:]
        joint_vel_penalty = np.sum(joint_vel ** 2)

        # ── Combine ─────────────────────────────────────────────────────
        reward = (
            cfg["w_lin_vel_tracking"] * lin_vel_reward
            + cfg["w_ang_vel_tracking"] * ang_vel_reward
            + cfg["w_lin_vel_z"] * lin_vel_z_penalty
            + cfg["w_ang_vel_xy"] * ang_vel_xy_penalty
            + cfg["w_orientation"] * orientation_penalty
            + cfg["w_torques"] * torque_penalty
            + cfg["w_action_rate"] * action_rate_penalty
            + cfg["w_joint_vel"] * joint_vel_penalty
        )

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
        """Sample new velocity commands."""
        self._commands[0] = self.np_random.uniform(*self.lin_vel_x_range)
        self._commands[1] = self.np_random.uniform(*self.lin_vel_y_range)
        self._commands[2] = self.np_random.uniform(*self.ang_vel_z_range)
        self._steps_since_command_resample = 0

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
        }

    @staticmethod
    def _default_reward_cfg() -> dict[str, float]:
        """Default reward configuration weights.

        Positive weights for tracking rewards, negative for penalties.
        """
        return {
            # Tracking rewards
            "w_lin_vel_tracking": 1.0,
            "w_ang_vel_tracking": 0.5,
            "tracking_sigma": 0.25,
            # Penalties (negative weights)
            "w_lin_vel_z": -2.0,
            "w_ang_vel_xy": -0.05,
            "w_orientation": -5.0,
            "w_torques": -2e-5,
            "w_action_rate": -0.01,
            "w_joint_vel": -1e-4,
        }
