"""
Script to visualize the trajectories of the cheetah3 environment using viser.

For us the plan is as follows:
1) Convert the cheetah3 MuJoCo model to URDF using cnvrt_mjcf_to_urdf.py
2) Use this URDF in viser to visualize the recorded trajectories (which now includes joint angles)
"""

import numpy as np
import viser
from viser.extras import ViserUrdf
import time
from pathlib import Path
import argparse
from typing import Optional
import yourdfpy

FOOT_NAMES = ["bfoot", "mfoot", "ffoot"]
FOOT_COLORS = {
    "bfoot": (255, 0, 0),
    "mfoot": (0, 180, 0),
    "ffoot": (0, 90, 255),
}
FOOT_GEOM_OFFSETS = {
    "bfoot": np.array([0.03, 0.0, -0.097]),
    "mfoot": np.array([0.03, 0.0, -0.097]),
    "ffoot": np.array([0.045, 0.0, -0.07]),
}


def rotate_vector_by_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a MuJoCo/viser-style wxyz quaternion."""
    quat = np.asarray(quat, dtype=float)
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return vector.copy()

    w, x, y, z = quat / norm
    q_vec = np.array([x, y, z])
    return vector + 2.0 * np.cross(q_vec, np.cross(q_vec, vector) + w * vector)


class MujocoTrajVisualizer:
    """
    Visualize saved MuJoCo trajectories using viser and an MJCF-to-URDF converted model.
    Supports visualizing up to two trajectories simultaneously.
    """

    def __init__(
            self,
            urdf_path: str,
            port: int = 8080,
            dt: float = 0.01,
            show_contacts: bool = True,
            show_body_part_trajectory: bool = True,
            num_ghost_frames: int = 5,
            dual_mode: bool = False,
            robot2_y_offset: float = -3.0,
        ):
        """
        Initializes the visualizer with a urdf model.

        Args:
            urdf_path: Path to the URDF file of the robot.
            port: Port for the viser server.
            dt: Time step, NOTE: This should match the time step used in the xml and urdf.
            show_contacts: Whether to show foot contact indicators.
            show_body_part_trajectory: Whether to show the trajectory path of body parts.
            num_ghost_frames: Maximum number of ghost frames to show (creates this many URDF instances).
            dual_mode: Whether to enable dual trajectory visualization.
            robot2_y_offset: Y-axis offset for the second robot (only used in dual mode).
        """
        self.server = viser.ViserServer(port=port)
        self.dt = dt
        self.show_contacts = show_contacts
        self.show_body_part_trajectory = show_body_part_trajectory
        self.port = port  # Store port for later use
        self.max_ghost_frames = num_ghost_frames
        self.dual_mode = dual_mode
        self.robot2_y_offset = robot2_y_offset

        # Convert to Path for ViserUrdf
        urdf_path = Path(urdf_path)
        self.urdf_path = urdf_path

        # Load URDF to get joint info
        self.urdf = yourdfpy.URDF.load(str(urdf_path), load_collision_meshes=False)

        # Create world frame for main robot (robot1)
        self.world_node = self.server.scene.add_frame(name="/world", show_axes=False)

        # Load URDF visualization for robot1 - ViserUrdf expects a Path object
        self.urdf_handle = ViserUrdf(
            target=self.server,
            urdf_or_path=urdf_path,
            root_node_name="/world",
        )

        # Robot 2 components (only created if dual_mode is enabled)
        self.world_node_2 = None
        self.urdf_handle_2 = None
        self._ghost_frames_2 = []
        self._ghost_urdf_handles_2 = []
        self._ghost_contact_indicators_2 = []
        self._ghost_alphas_2 = []
        self._contact_indicators_2 = {}
        self._trajectory_line_2 = None
        self.body_part_trajectory_2 = []
        self.trajectory_data_2 = None
        
        if self.dual_mode:
            # Create world frame for second robot
            self.world_node_2 = self.server.scene.add_frame(name="/world2", show_axes=False)
            
            # Load URDF visualization for robot2 with different color
            self.urdf_handle_2 = ViserUrdf(
                target=self.server,
                urdf_or_path=urdf_path,
                root_node_name="/world2",
                mesh_color_override=(0.2, 0.5, 1.0),  # Blue color for robot2
            )
            
            # Create ghost frame instances for robot2
            for i in range(self.max_ghost_frames):
                ghost_world = self.server.scene.add_frame(
                    name=f"/ghost2_{i}", 
                    show_axes=False
                )
                ghost_urdf = ViserUrdf(
                    target=self.server,
                    urdf_or_path=urdf_path,
                    root_node_name=f"/ghost2_{i}",
                    mesh_color_override=(0.2, 0.5, 1.0, 1.0),
                )
                self._ghost_frames_2.append(ghost_world)
                self._ghost_urdf_handles_2.append(ghost_urdf)
                self._ghost_contact_indicators_2.append({})
                self._ghost_alphas_2.append(1.0)

        # Create ghost frame instances for robot1 (for showing multiple frames at once)
        # Each ghost has its own world frame and URDF handle
        self._ghost_frames = []
        self._ghost_urdf_handles = []
        self._ghost_contact_indicators = []  # Contact indicators for each ghost
        self._ghost_alphas = []  # Track current alpha for each ghost
        for i in range(self.max_ghost_frames):
            ghost_world = self.server.scene.add_frame(
                name=f"/ghost_{i}", 
                show_axes=False
            )
            # Initially create with default alpha, will be recreated with proper alpha later
            ghost_urdf = ViserUrdf(
                target=self.server,
                urdf_or_path=urdf_path,
                root_node_name=f"/ghost_{i}",
            )
            self._ghost_frames.append(ghost_world)
            self._ghost_urdf_handles.append(ghost_urdf)
            self._ghost_contact_indicators.append({})  # Dict for each ghost's contacts
            self._ghost_alphas.append(1.0)  # Track alpha for recreation
        
        # Initially hide all ghosts
        self._set_ghost_visibility(visible=False)
        if self.dual_mode:
            self._set_ghost_visibility(visible=False, robot_idx=2)

        # Add ground plane
        self.server.scene.add_grid(
            "/grid",
            width=200,
            height=200,
            position=(0, 0, 0),
            plane="xy"
        )

        # Setup GUI controls
        self._setup_gui()

        # Initialize state
        self.body_part_trajectory = []
        self.trajectory_data = None
        self.current_frame = 0

        # Visualization handles
        self._contact_indicators = {}
        self._trajectory_line = None
        
        # Camera follow state
        # These define the camera position relative to the robot when following is enabled
        # The camera will be positioned at: robot_pos + camera_offset
        # and will look at: robot_pos + look_at_offset
        self._camera_follow_enabled = False

    def _setup_gui(self):
        """
        Setup GUI controls for playback.
        """
        # Playback controls
        with self.server.gui.add_folder("Playback Controls"):
            self.play_pause = self.server.gui.add_checkbox(
                "Play", initial_value=False, hint="Toggle trajectory playback"
            )
            self.speed_slider = self.server.gui.add_slider(
                "Speed", min=0.01, max=1.0, step=0.01, initial_value=0.1,
                hint="Playback speed multiplier"
            )
            self.frame_slider = self.server.gui.add_slider(
                "Frame", min=0, max=1000, step=1, initial_value=0,
                hint="Current frame"
            )
            self.loop_checkbox = self.server.gui.add_checkbox(
                "Loop", initial_value=True, hint="Loop playback"
            )
            self.reset_button = self.server.gui.add_button(
                "Reset", hint="Reset to first frame"
            )

            @self.reset_button.on_click
            def _(_):
                self.current_frame = 0
                self.frame_slider.value = 0
                self.update_visualization(0)

        # Visualization options
        with self.server.gui.add_folder("Visualization Options"):
            self.show_contacts_checkbox = self.server.gui.add_checkbox(
                "Show Foot Contacts", initial_value=self.show_contacts,
                hint="Toggle foot contact indicators"
            )
            self.show_body_part_trajectory_checkbox = self.server.gui.add_checkbox(
                "Show Body Part Trajectory", initial_value=self.show_body_part_trajectory,
                hint="Toggle body part trajectory path"
            )
            self.show_stats = self.server.gui.add_checkbox(
                "Show Stats", initial_value=True,
                hint="Show simulation statistics"
            )
        
        # Ghost frames options (for overlaying multiple frames)
        with self.server.gui.add_folder("Ghost Frames"):
            self.show_ghosts_checkbox = self.server.gui.add_checkbox(
                "Show Ghost Frames", initial_value=False,
                hint="Show multiple frames overlaid (onion-skin effect)"
            )
            self.num_ghosts_slider = self.server.gui.add_slider(
                "Number of Ghosts", min=1, max=self.max_ghost_frames, step=1, 
                initial_value=3,
                hint="How many ghost frames to display"
            )
            self.ghost_interval_slider = self.server.gui.add_slider(
                "Frame Interval", min=1, max=100, step=1, initial_value=20,
                hint="Number of frames between each ghost"
            )
            self.ghost_direction = self.server.gui.add_dropdown(
                "Ghost Direction",
                options=["Past", "Future", "Both"],
                initial_value="Past",
                hint="Show ghosts from past frames, future frames, or both"
            )
        
        # Statistics display (NOTE: will be populated after loading the trajectory)
        with self.server.gui.add_folder("Statistics", expand_by_default=False):
            self.stats_text = self.server.gui.add_text(
                "Info", initial_value="No trajectory loaded", disabled=True
            )
        
        # Camera follow controls
        # This section allows the camera to automatically follow the robot during playback
        with self.server.gui.add_folder("Camera Follow", expand_by_default=True):
            self.follow_camera_checkbox = self.server.gui.add_checkbox(
                "Follow Robot", initial_value=False,
                hint="Camera automatically follows the robot during playback"
            )
            
            # Camera offset controls - these define where the camera is positioned relative to the robot
            # ---------------------------------------------------------------------------------
            # CAMERA ANGLE ADJUSTMENT GUIDE:
            # - camera_distance: How far behind/in-front of the robot (X-axis offset)
            #   Positive = behind robot, Negative = in front of robot
            # - camera_height: How high above/below the robot (Z-axis offset)  
            #   Positive = above robot, Negative = below robot
            # - camera_side_offset: How far to the side of the robot (Y-axis offset)
            #   Positive = to the left, Negative = to the right
            # 
            # Common camera angle presets:
            # - Side view: distance=0, height=1.0, side_offset=-5.0
            # - Behind view: distance=-5.0, height=2.0, side_offset=0
            # - Top-down view: distance=0, height=8.0, side_offset=0
            # - 3/4 view: distance=-4.0, height=3.0, side_offset=-3.0
            # ---------------------------------------------------------------------------------
            self.camera_distance_slider = self.server.gui.add_slider(
                "Distance Behind", min=-10.0, max=10.0, step=0.1, initial_value=-5.0,
                hint="Camera distance behind the robot (negative = behind, positive = in front)"
            )
            self.camera_height_slider = self.server.gui.add_slider(
                "Height Above", min=-2.0, max=10.0, step=0.1, initial_value=2.0,
                hint="Camera height above the robot"
            )
            self.camera_side_offset_slider = self.server.gui.add_slider(
                "Side Offset", min=-10.0, max=10.0, step=0.1, initial_value=0.0,
                hint="Camera offset to the side (positive=left, negative=right)"
            )
            
            # Look-at offset - where the camera points relative to the robot
            # Adjust this to look slightly ahead of the robot or at its feet
            self.look_at_height_slider = self.server.gui.add_slider(
                "Look-at Height", min=-2.0, max=5.0, step=0.1, initial_value=0.8,
                hint="Height of the point the camera looks at (relative to ground)"
            )
            self.look_at_forward_slider = self.server.gui.add_slider(
                "Look-at Forward", min=-5.0, max=10.0, step=0.1, initial_value=2.0,
                hint="How far ahead of the robot the camera looks"
            )
            
            # Preset buttons for common camera angles
            self.camera_preset_dropdown = self.server.gui.add_dropdown(
                "Camera Preset",
                options=["Custom", "Side View (Left to Right)", "Side View (Right to Left)", "Behind View", "Top-Down", "3/4 View", "Front View"],
                initial_value="Custom",
                hint="Select a preset camera angle"
            )
            
            @self.camera_preset_dropdown.on_update
            def _(_):
                """Apply camera preset when dropdown selection changes."""
                preset = self.camera_preset_dropdown.value
                if preset == "Side View (Right to Left)":
                    # View from the side (perpendicular to robot's forward direction)
                    self.camera_distance_slider.value = 0.0
                    self.camera_height_slider.value = 1.0
                    self.camera_side_offset_slider.value = 5.0
                    self.look_at_height_slider.value = 1.0
                    self.look_at_forward_slider.value = 0.0
                elif preset == "Side View (Left to Right)":
                    # View from the side (perpendicular to robot's forward direction)
                    self.camera_distance_slider.value = 0.0
                    self.camera_height_slider.value = 1.0
                    self.camera_side_offset_slider.value = -5.0
                    self.look_at_height_slider.value = 1.0
                    self.look_at_forward_slider.value = 0.0
                elif preset == "Behind View":
                    # Classic third-person view from behind
                    self.camera_distance_slider.value = -5.0
                    self.camera_height_slider.value = 2.0
                    self.camera_side_offset_slider.value = 0.0
                    self.look_at_height_slider.value = 0.8
                    self.look_at_forward_slider.value = 2.0
                elif preset == "Top-Down":
                    # Bird's eye view looking straight down
                    self.camera_distance_slider.value = 0.0
                    self.camera_height_slider.value = 8.0
                    self.camera_side_offset_slider.value = 0.0
                    self.look_at_height_slider.value = 0.0
                    self.look_at_forward_slider.value = 0.0
                elif preset == "3/4 View":
                    # Isometric-style view from behind and to the side
                    self.camera_distance_slider.value = -4.0
                    self.camera_height_slider.value = 3.0
                    self.camera_side_offset_slider.value = -3.0
                    self.look_at_height_slider.value = 0.8
                    self.look_at_forward_slider.value = 1.0
                elif preset == "Front View":
                    # View from in front of the robot (facing it)
                    self.camera_distance_slider.value = 5.0
                    self.camera_height_slider.value = 1.5
                    self.camera_side_offset_slider.value = 0.0
                    self.look_at_height_slider.value = 1.0
                    self.look_at_forward_slider.value = 0.0
                # "Custom" preset does nothing - allows manual adjustment
            
            @self.follow_camera_checkbox.on_update
            def _(_):
                """Update camera follow state when checkbox is toggled."""
                self._camera_follow_enabled = self.follow_camera_checkbox.value

    def load_trajectory(self, npz_path: str, robot_idx: int = 1):
        """
        Load trajectory data from .npz file

        Refer to record_body_trajs_by_policy.py for up to date data format.

        - joint_angles: (num_frames, num_qpos) - full qpos including root position/orientation
        - joint_names: list of joint names
        - pos_<body_name>: (num_frames, 3) positions for each body
        - quat_<body_name>: (num_frames, 4) quaternions for each body
        - observations: (num_frames, obs_dim) observations
        - actions: (num_frames, action_dim) actions
        - rewards: (num_frames,) rewards
        - contact_<foot_name>: (num_frames,) boolean contact indicators
        - clearance_<foot_name>: (num_frames,) foot height above ground
        
        Args:
            npz_path: Path to the trajectory .npz file
            robot_idx: Which robot to load trajectory for (1 or 2)
        """
        data = np.load(npz_path, allow_pickle=True)
        
        if robot_idx == 1:
            self.trajectory_data = data
            # Initialize body part trajectory path
            # NOTE: Change this for whatever body part you want to track
            if 'pos_torso' in data.files:
                self.trajectory_path = data['pos_torso']
        else:  # robot_idx == 2
            self.trajectory_data_2 = data
            if 'pos_torso' in data.files:
                self.body_part_trajectory_2 = data['pos_torso']

        # Update frame slider range based on robot1 (primary)
        if robot_idx == 1:
            num_frames = data['timesteps']
            self.frame_slider.max = num_frames - 1
            print(f"Loaded trajectory 1 with {num_frames} frames.")
        else:
            print(f"Loaded trajectory 2 with {data['timesteps']} frames.")
        
        print(f"Available data (robot {robot_idx}): {list(data.keys())}")

        # Display stats (only update for robot1)
        if robot_idx == 1 and self.show_stats.value:
            self._update_statistics()

        # Initialize to first frame if robot1
        if robot_idx == 1:
            self.update_visualization(0)

        return data
    
    def _update_statistics(self):
        """
        Update statistics display based on loaded trajectory data.
        """
        if self.trajectory_data is None:
            return
        
        data = self.trajectory_data
        stats_lines = [
            f"Frames: {data['timesteps']}",
            f"Duration: {data['timesteps'] * self.dt:.2f} s",
            f"Total Reward: {np.sum(data['rewards']):.2f}",
            f"Average Reward: {np.mean(data['rewards']):.2f}",
            f"Terminated Early: {data['done']}",
        ]

        # Add contact stats if available
        for foot_name in FOOT_NAMES:
            contact_key = f"contact_pct_{foot_name}"
            if contact_key in data.files:
                stats_lines.append(f"{foot_name} Contact: {data[contact_key]:.1f}%")
        
        self.stats_text.value = "\n".join(stats_lines)
    
    def _get_actuated_joint_positions(self, frame_idx: int, robot_idx: int = 1) -> np.ndarray:
        """
        Extract actuated joint positions from qpos data

        For cheetah3, the qpos structure is:
        - qpos[0]: rootx (x position - slide joint) - treated as fixed in URDF
        - qpos[1]: rootz (z position - slide joint) - treated as fixed in URDF
        - qpos[2]: rooty (rotation around y-axis - hinge joint) - actuated in URDF
        - qpos[3:12]: actuated joint positions (9 joints for cheetah3)
            - bthigh, bshin, bfoot
            - mthigh, mshin, mfoot
            - fthigh, fshin, ffoot

        The URDF actuated joints are rooty plus the 9 leg joints (10 total).

        Args:
            frame_idx: Frame index to extract positions from
            robot_idx: Which robot to get positions for (1 or 2)

        Returns:
            np.ndarray of actuated joint angles in the expected order for yourdfpy
        """
        traj_data = self.trajectory_data if robot_idx == 1 else self.trajectory_data_2
        if traj_data is None:
            return np.zeros(10)
        
        qpos = traj_data['joint_angles'][frame_idx]

        # Include rooty (qpos[2]) and the 9 leg joints (qpos[3:12]).
        # Total: 10 actuated joints matching the converted URDF structure.
        actuated_joint_angles = qpos[2:]

        return actuated_joint_angles

    def _get_root_state(self, frame_idx: int, robot_idx: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract root position and orientation from qpos data

        For cheetah3:
        - qpos[0]: rootx (x position - horizontal forward/back)
        - qpos[1]: rootz (z position - vertical)

        Note: rooty is handled by the URDF joint system, not as part of root state

        Args:
            frame_idx: Frame index to extract state from
            robot_idx: Which robot to get state for (1 or 2)

        Returns:
            Tuple of (pos, quat) where:
            - pos is (x, y, z) position (y is offset for robot2)
            - quat is identity quaternion (rotation handled by URDF rooty joint)
        """
        # Select the appropriate trajectory data
        traj_data = self.trajectory_data if robot_idx == 1 else self.trajectory_data_2
        if traj_data is None:
            return np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
        
        qpos = traj_data['joint_angles'][frame_idx]

        # Extract root position (x, y, z)
        # qpos[0] is x (forward), qpos[1] is z (height)
        # Apply y-offset for robot2
        y_offset = 0.0 if robot_idx == 1 else self.robot2_y_offset
        root_pos = np.array([qpos[0], y_offset, qpos[1]])

        # Identity quaternion since rotation is handled by rooty joint in URDF
        root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        return root_pos, root_quat

    def _get_foot_marker_position(self, traj_data, foot_name: str, frame_idx: int) -> np.ndarray:
        """
        Return the visible foot geom center instead of the foot body origin.

        The recorded `pos_<foot>` value is the MuJoCo body frame at the ankle
        joint. The cheetah3 foot capsule is offset from that frame in the MJCF,
        so contact spheres should be placed at the rotated geom offset.
        """
        foot_pos = traj_data[f"pos_{foot_name}"][frame_idx].copy()
        quat_key = f"quat_{foot_name}"
        if quat_key not in traj_data.files:
            return foot_pos
        return foot_pos + rotate_vector_by_wxyz(
            traj_data[quat_key][frame_idx],
            FOOT_GEOM_OFFSETS[foot_name],
        )
    
    def _set_ghost_visibility(self, visible: bool, num_visible: int = None, robot_idx: int = 1):
        """
        Set visibility of ghost frame instances.
        
        Args:
            visible: Whether ghosts should be visible
            num_visible: Number of ghosts to show (if None, show/hide all)
            robot_idx: Which robot's ghosts to control (1 or 2)
        """
        # Select the appropriate ghost frames
        ghost_frames = self._ghost_frames if robot_idx == 1 else self._ghost_frames_2
        ghost_urdf_handles = self._ghost_urdf_handles if robot_idx == 1 else self._ghost_urdf_handles_2
        
        for i, (ghost_world, ghost_urdf) in enumerate(zip(ghost_frames, ghost_urdf_handles)):
            if num_visible is not None:
                is_visible = visible and (i < num_visible)
            else:
                is_visible = visible
            
            # Set visibility by moving far away if hidden, or updating position if visible
            # ViserUrdf doesn't have direct visibility control, so we use position
            if not is_visible:
                ghost_world.position = (0, 0, -1000)  # Move far below ground
    
    def _update_ghost_frame(self, ghost_idx: int, frame_idx: int, opacity_factor: float = 1.0, robot_idx: int = 1):
        """
        Update a single ghost frame to show a specific trajectory frame.
        
        Args:
            ghost_idx: Index of the ghost instance to update
            frame_idx: Trajectory frame to display
            opacity_factor: Alpha value for transparency (0.0 = fully transparent, 1.0 = opaque)
            robot_idx: Which robot to update (1 or 2)
        """
        # Select appropriate ghost components
        ghost_frames = self._ghost_frames if robot_idx == 1 else self._ghost_frames_2
        ghost_urdf_handles = self._ghost_urdf_handles if robot_idx == 1 else self._ghost_urdf_handles_2
        ghost_alphas = self._ghost_alphas if robot_idx == 1 else self._ghost_alphas_2
        
        if ghost_idx >= len(ghost_frames):
            return
        
        # Check if we need to recreate the URDF with a different alpha
        # We recreate if alpha changed significantly to update transparency
        alpha_changed = abs(ghost_alphas[ghost_idx] - opacity_factor) > 0.05
        
        if alpha_changed:
            # Remove old URDF handle
            # ViserUrdf doesn't have a built-in remove method, but replacing it works
            # Create new URDF with updated alpha
            base_color = (1.0, 0.5, 0.0) if robot_idx == 1 else (0.2, 0.5, 1.0)
            ghost_name = f"/ghost_{ghost_idx}" if robot_idx == 1 else f"/ghost2_{ghost_idx}"
            
            if robot_idx == 1:
                self._ghost_urdf_handles[ghost_idx] = ViserUrdf(
                    target=self.server,
                    urdf_or_path=self.urdf_path,
                    root_node_name=ghost_name,
                    mesh_color_override=(*base_color, opacity_factor),
                )
                self._ghost_alphas[ghost_idx] = opacity_factor
            else:
                self._ghost_urdf_handles_2[ghost_idx] = ViserUrdf(
                    target=self.server,
                    urdf_or_path=self.urdf_path,
                    root_node_name=ghost_name,
                    mesh_color_override=(*base_color, opacity_factor),
                )
                self._ghost_alphas_2[ghost_idx] = opacity_factor
        
        # Get root position and orientation for this frame
        root_pos, root_quat = self._get_root_state(frame_idx, robot_idx=robot_idx)
        
        # Update ghost world frame position
        ghost_frames[ghost_idx].position = root_pos
        ghost_frames[ghost_idx].wxyz = root_quat
        
        # Get actuated joint positions
        joint_positions = self._get_actuated_joint_positions(frame_idx, robot_idx=robot_idx)
        
        # Update ghost URDF config
        ghost_urdf_handles[ghost_idx].update_cfg(joint_positions)
        
        # Update ghost foot contacts
        self._update_ghost_contacts(ghost_idx, frame_idx, robot_idx=robot_idx)
    
    def _update_ghost_contacts(self, ghost_idx: int, frame_idx: int, robot_idx: int = 1):
        """
        Update foot contact visualization for a specific ghost frame.
        
        Args:
            ghost_idx: Index of the ghost instance
            frame_idx: Trajectory frame to display contacts for
            robot_idx: Which robot to update (1 or 2)
        """
        # Select appropriate components
        ghost_contact_indicators = self._ghost_contact_indicators if robot_idx == 1 else self._ghost_contact_indicators_2
        traj_data = self.trajectory_data if robot_idx == 1 else self.trajectory_data_2
        
        if not self.show_contacts_checkbox.value or traj_data is None:
            # Hide this ghost's contact indicators
            for handle in ghost_contact_indicators[ghost_idx].values():
                handle.visible = False
            return
        
        for foot_name in FOOT_NAMES:
            contact_key = f'contact_{foot_name}'
            pos_key = f'pos_{foot_name}'
            
            if contact_key not in traj_data.files:
                continue
            
            is_in_contact = traj_data[contact_key][frame_idx]
            foot_pos = self._get_foot_marker_position(traj_data, foot_name, frame_idx)
            
            # Apply y-offset for robot2
            if robot_idx == 2:
                foot_pos[1] += self.robot2_y_offset
            
            # Create or update contact indicator for this ghost
            ghost_prefix = "ghost" if robot_idx == 1 else "ghost2"
            indicator_name = f'/{ghost_prefix}_{ghost_idx}_contact_{foot_name}'
            
            if indicator_name not in ghost_contact_indicators[ghost_idx]:
                # Create a sphere for contact indicator
                # Use slightly transparent colors for ghost contacts
                ghost_contact_indicators[ghost_idx][indicator_name] = self.server.scene.add_icosphere(
                    indicator_name,
                    radius=0.08,  # Slightly smaller than main contacts
                    color=FOOT_COLORS[foot_name],
                    position=tuple(foot_pos)
                )
            
            # Update position and visibility
            ghost_contact_indicators[ghost_idx][indicator_name].position = tuple(foot_pos)
            ghost_contact_indicators[ghost_idx][indicator_name].visible = bool(is_in_contact)
    
    def _hide_ghost_contacts(self, ghost_idx: int, robot_idx: int = 1):
        """
        Hide all contact indicators for a specific ghost.
        
        Args:
            ghost_idx: Index of the ghost instance
            robot_idx: Which robot's ghost contacts to hide (1 or 2)
        """
        ghost_contact_indicators = self._ghost_contact_indicators if robot_idx == 1 else self._ghost_contact_indicators_2
        
        if ghost_idx < len(ghost_contact_indicators):
            for handle in ghost_contact_indicators[ghost_idx].values():
                handle.visible = False
    
    def _update_ghost_visualization(self, current_frame: int, robot_idx: int = 1):
        """
        Update all ghost frame visualizations based on current frame and settings.
        
        Args:
            current_frame: The main/current frame being displayed
            robot_idx: Which robot's ghosts to update (1 or 2)
        """
        traj_data = self.trajectory_data if robot_idx == 1 else self.trajectory_data_2
        ghost_frames = self._ghost_frames if robot_idx == 1 else self._ghost_frames_2
        ghost_alphas = self._ghost_alphas if robot_idx == 1 else self._ghost_alphas_2
        
        if not self.show_ghosts_checkbox.value or traj_data is None:
            self._set_ghost_visibility(visible=False, robot_idx=robot_idx)
            # Also hide all ghost contacts
            for ghost_idx in range(self.max_ghost_frames):
                self._hide_ghost_contacts(ghost_idx, robot_idx=robot_idx)
            return
        
        num_ghosts = int(self.num_ghosts_slider.value)
        interval = int(self.ghost_interval_slider.value)
        direction = self.ghost_direction.value
        num_frames = int(traj_data['timesteps'])
        
        # Calculate which frames to show as ghosts
        ghost_frame_indices = []
        
        if direction == "Past":
            # Show frames before current
            for i in range(1, num_ghosts + 1):
                frame_idx = current_frame - i * interval
                if frame_idx >= 0:
                    ghost_frame_indices.append(frame_idx)
                    
        elif direction == "Future":
            # Show frames after current
            for i in range(1, num_ghosts + 1):
                frame_idx = current_frame + i * interval
                if frame_idx < num_frames:
                    ghost_frame_indices.append(frame_idx)
                    
        else:  # "Both"
            # Show frames before and after, alternating
            half_ghosts = num_ghosts // 2
            # Past frames
            for i in range(1, half_ghosts + 1):
                frame_idx = current_frame - i * interval
                if frame_idx >= 0:
                    ghost_frame_indices.append(frame_idx)
            # Future frames
            for i in range(1, num_ghosts - half_ghosts + 1):
                frame_idx = current_frame + i * interval
                if frame_idx < num_frames:
                    ghost_frame_indices.append(frame_idx)
        
        # Update visible ghosts with progressive transparency
        for ghost_idx, frame_idx in enumerate(ghost_frame_indices):
            if ghost_idx < self.max_ghost_frames:
                # Calculate opacity factor based on distance from current frame
                # Farther frames are more transparent
                distance = abs(frame_idx - current_frame)
                max_distance = num_ghosts * interval
                # Range from 0.3 (most transparent) to 0.9 (nearly opaque)
                # The current frame itself isn't shown as ghost, so ghosts are always somewhat transparent
                opacity_factor = 0.9 - (distance / max_distance) * 0.6 if max_distance > 0 else 0.9
                
                self._update_ghost_frame(ghost_idx, frame_idx, opacity_factor, robot_idx=robot_idx)
        
        # Hide remaining ghosts and their contacts
        for ghost_idx in range(len(ghost_frame_indices), self.max_ghost_frames):
            ghost_frames[ghost_idx].position = (0, 0, -1000)
            self._hide_ghost_contacts(ghost_idx, robot_idx=robot_idx)
            # Reset alpha tracking for hidden ghosts
            ghost_alphas[ghost_idx] = 1.0
    
    def _update_contact_visualization(self, frame_idx: int, robot_idx: int = 1):
        """
        Update visualization of foot contacts
        
        Args:
            frame_idx: Frame index to display
            robot_idx: Which robot to update (1 or 2)
        """
        contact_indicators = self._contact_indicators if robot_idx == 1 else self._contact_indicators_2
        traj_data = self.trajectory_data if robot_idx == 1 else self.trajectory_data_2
        
        if not self.show_contacts_checkbox.value or traj_data is None:
            # Hide contact indicators
            for handle in contact_indicators.values():
                handle.visible = False
            return
        
        # Show contact indicators for the feet
        for foot_name in FOOT_NAMES:
            contact_key = f'contact_{foot_name}'
            pos_key = f'pos_{foot_name}'

            if contact_key not in traj_data.files:
                continue

            is_in_contact = traj_data[contact_key][frame_idx]
            foot_pos = self._get_foot_marker_position(traj_data, foot_name, frame_idx)
            
            # Apply y-offset for robot2
            if robot_idx == 2:
                foot_pos[1] += self.robot2_y_offset

            # Create or update contact indicator
            prefix = "" if robot_idx == 1 else "2_"
            indicator_name = f'/contact_{prefix}{foot_name}'

            if indicator_name not in contact_indicators:
                # Create a sphere for contact indicator
                contact_indicators[indicator_name] = self.server.scene.add_icosphere(
                    indicator_name,
                    radius=0.1,
                    color=FOOT_COLORS[foot_name],
                    position=tuple(foot_pos)
                )

            # Update position and visibility
            contact_indicators[indicator_name].position = tuple(foot_pos)
            contact_indicators[indicator_name].visible = bool(is_in_contact)
        
    def _update_trajectory_path(self, frame_idx: int, robot_idx: int = 1):
        """
        Update visualization of body part trajectory path
        
        Args:
            frame_idx: Frame index to display
            robot_idx: Which robot to update (1 or 2)
        """
        # Select appropriate components
        if robot_idx == 1:
            trajectory_path = self.trajectory_path if hasattr(self, 'trajectory_path') else None
            trajectory_line = self._trajectory_line
            line_name = "/trajectory_line"
            color_rgb = [0, 1, 1]  # Cyan for robot1
        else:
            trajectory_path = self.body_part_trajectory_2
            trajectory_line = self._trajectory_line_2
            line_name = "/trajectory_line_2"
            color_rgb = [1, 0.5, 0]  # Orange for robot2
        
        if not self.show_body_part_trajectory_checkbox.value or trajectory_path is None or len(trajectory_path) == 0:
            if trajectory_line is not None:
                trajectory_line.visible = False
            return
        
        # Show trajectory up to current frame
        path_positions = trajectory_path[:frame_idx+1].copy()
        
        # Apply y-offset for robot2
        if robot_idx == 2:
            path_positions[:, 1] += self.robot2_y_offset

        if len(path_positions) < 2:
            return
        
        # Create line segments for the path
        # Convert to pairs of consecutive points
        points = np.stack([path_positions[:-1], path_positions[1:]], axis=1)

        # Create colors (fade older parts of the body part trajectory)
        num_segments = len(points)
        colors = np.zeros((num_segments, 2, 3), dtype=np.uint8)
        for i in range(num_segments):
            alpha = i / num_segments  # Fade from 0 to 1
            color_val = int(50 + alpha * 205)  # From dim to bright
            # Apply the color based on robot
            colors[i] = [[int(c * color_val) for c in color_rgb], 
                        [int(c * color_val) for c in color_rgb]]

        if trajectory_line is None:
            new_line = self.server.scene.add_line_segments(
                line_name,
                points=points,
                colors=colors,
                line_width=2.0,
            )
            if robot_idx == 1:
                self._trajectory_line = new_line
            else:
                self._trajectory_line_2 = new_line
        else:
            trajectory_line.points = points
            trajectory_line.colors = colors
            trajectory_line.visible = True
    
    def _update_follow_camera(self, frame_idx: int):
        """
        Update camera position to follow the robot (robot1).
        
        This method positions the camera relative to the robot's current position
        based on the GUI slider values. The camera follows the robot smoothly
        during playback.
        
        Camera Coordinate System (viser uses +Z up):
        - X-axis: Forward/backward (robot walks in +X direction)
        - Y-axis: Left/right (positive = left)
        - Z-axis: Up/down (positive = up)
        
        To customize the camera angle programmatically, modify the offset values:
        - camera_offset: [x, y, z] position offset from robot
        - look_at_offset: [x, y, z] position offset for where camera looks
        
        Args:
            frame_idx: Current frame index being displayed
        """
        if not self._camera_follow_enabled or self.trajectory_data is None:
            return
        
        # Get robot's current position (for robot1)
        root_pos, _ = self._get_root_state(frame_idx, robot_idx=1)
        
        # Calculate camera position based on GUI sliders
        # ---------------------------------------------------------------------------------
        # HOW TO ADJUST CAMERA ANGLE:
        # 
        # The camera position is calculated as: robot_position + camera_offset
        # where camera_offset = [distance_behind, side_offset, height_above]
        #
        # distance_behind (X-axis):
        #   - Negative values place camera behind the robot (following view)
        #   - Positive values place camera in front of the robot (facing view)
        #
        # side_offset (Y-axis):
        #   - Positive values place camera to the left of the robot
        #   - Negative values place camera to the right of the robot
        #
        # height_above (Z-axis):
        #   - Controls vertical position of camera
        #   - Higher values give more of a bird's eye view
        #
        # For example, to get a classic side-scrolling game view:
        #   distance_behind = 0.0 (beside the robot)
        #   side_offset = -5.0 (to the right side)
        #   height_above = 1.0 (at robot's height)
        # ---------------------------------------------------------------------------------
        camera_offset = np.array([
            self.camera_distance_slider.value,   # X: distance behind robot
            self.camera_side_offset_slider.value, # Y: side offset
            self.camera_height_slider.value       # Z: height above robot
        ])
        
        camera_position = root_pos + camera_offset
        
        # Calculate look-at position (where the camera points)
        # This is typically slightly ahead of and at the robot's center height
        look_at_offset = np.array([
            self.look_at_forward_slider.value,  # X: look ahead of robot
            0.0,                                  # Y: centered on robot
            self.look_at_height_slider.value     # Z: look at robot's torso height
        ])
        
        look_at_position = root_pos + look_at_offset
        
        # Update camera for all connected clients
        # This ensures all viewers see the same following camera
        clients = self.server.get_clients()
        for client_id, client in clients.items():
            # Use atomic update to prevent visual jitter during camera movement
            # This ensures position and look_at are applied together
            with client.atomic():
                client.camera.position = camera_position
                client.camera.look_at = look_at_position
    
    def update_visualization(self, frame_idx: int):
        """
        Update the visualization to show a specific frame
        
        Args:
            frame_idx: Frame index to display
        """
        if self.trajectory_data is None:
            return
        
        # Clamp frame index for robot1
        frame_idx = int(np.clip(frame_idx, 0, self.trajectory_data['timesteps'] - 1))

        # Update robot1
        self._update_single_robot_visualization(frame_idx, robot_idx=1)
        
        # Update robot2 if in dual mode
        if self.dual_mode and self.trajectory_data_2 is not None:
            # Clamp frame index for robot2 (might have different length)
            frame_idx_2 = int(np.clip(frame_idx, 0, self.trajectory_data_2['timesteps'] - 1))
            self._update_single_robot_visualization(frame_idx_2, robot_idx=2)
        
        # Update follow camera if enabled (always follows robot1)
        self._update_follow_camera(frame_idx)
    
    def _update_single_robot_visualization(self, frame_idx: int, robot_idx: int = 1):
        """
        Update visualization for a single robot.
        
        Args:
            frame_idx: Frame index to display
            robot_idx: Which robot to update (1 or 2)
        """
        # Select appropriate components based on robot_idx
        if robot_idx == 1:
            world_node = self.world_node
            urdf_handle = self.urdf_handle
            traj_data = self.trajectory_data
        else:  # robot_idx == 2
            world_node = self.world_node_2
            urdf_handle = self.urdf_handle_2
            traj_data = self.trajectory_data_2
        
        if traj_data is None:
            return

        # Get root position and orientation from qpos
        root_pos, root_quat = self._get_root_state(frame_idx, robot_idx=robot_idx)

        # Update world node position (floating base approach)
        # Why move the world frame instead of keeping it static?
        # - The URDF conversion marks rootx/rootz as type="fixed" (not movable joints)
        # - We can't update their positions via joint angles since they don't exist as actuated joints
        # - Instead, we use the world frame as a "floating base carrier" for the entire robot
        # - The world frame translates to follow the robot's global position
        # - Only the relative joint angles (rooty, legs) are updated via update_cfg()
        # This is a common pattern for floating-base robots (humanoids, quadrupeds) where
        # the base can translate freely but isn't part of the actuated joint chain.
        world_node.position = root_pos
        world_node.wxyz = root_quat

        # Get actuated joint positions
        joint_positions = self._get_actuated_joint_positions(frame_idx, robot_idx=robot_idx)

        # Update URDF config
        with self.server.atomic():
            urdf_handle.update_cfg(joint_positions)

        # Update contact visualization
        self._update_contact_visualization(frame_idx, robot_idx=robot_idx)

        # Update body part trajectory path
        self._update_trajectory_path(frame_idx, robot_idx=robot_idx)
        
        # Update ghost frames (multiple overlaid frames)
        self._update_ghost_visualization(frame_idx, robot_idx=robot_idx)
    
    def play(self):
        """
        Run the playback loop
        """
        print(f"\nVisualizer running on http://localhost:{self.port}")
        print("Connect to the visualizer in your browser.")
        print("\nControls:")
        print("  - Play/Pause: Toggle playback")
        print("  - Speed: Adjust playback speed")
        print("  - Frame slider: Scrub through frames")
        print("  - Loop: Enable/disable looping")
        print("\nCamera Follow:")
        print("  - Enable 'Follow Robot' checkbox to have camera track the robot")
        print("  - Adjust Distance/Height/Side sliders to change camera angle")
        print("  - Use Camera Preset dropdown for quick angle changes")
        print("\nPress Ctrl+C to exit\n")
        
        last_time = time.time()
        
        while True:
            current_time = time.time()
            dt_actual = current_time - last_time
            
            if self.play_pause.value and self.trajectory_data is not None:
                # Update frame based on speed
                speed = self.speed_slider.value
                frame_increment = dt_actual * speed / self.dt
                
                self.current_frame += frame_increment
                
                # Handle looping
                num_frames = self.trajectory_data['timesteps']
                if self.current_frame >= num_frames:
                    if self.loop_checkbox.value:
                        self.current_frame = 0
                    else:
                        self.current_frame = num_frames - 1
                        self.play_pause.value = False
                
                # Update visualization
                self.update_visualization(int(self.current_frame))
                
                # Update slider
                self.frame_slider.value = int(self.current_frame)
                
                last_time = current_time
            else:
                # Paused - allow manual frame control
                if self.trajectory_data is not None:
                    manual_frame = int(self.frame_slider.value)
                    if manual_frame != int(self.current_frame):
                        self.current_frame = manual_frame
                        self.update_visualization(self.current_frame)
                
                last_time = current_time
                time.sleep(0.01)  # Small sleep when paused
            
            # Small sleep to prevent CPU spinning
            time.sleep(max(0.001, self.dt / self.speed_slider.value - dt_actual))


def main():
    # Get default URDF path relative to this script
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    default_urdf_path = workspace_root / "mpc_rl" / "tasks" / "cheetah" / "cheetah_modified.urdf"
    
    parser = argparse.ArgumentParser(
        description="Visualize MuJoCo cheetah3 trajectories using Viser",
        epilog="""Examples:
        Single trajectory: python viser_cheetah3_viz_trajs.py --trajectory path/to/traj.npz
        Dual trajectories: python viser_cheetah3_viz_trajs.py --trajectory1 path/to/traj1.npz --trajectory2 path/to/traj2.npz
        """
    )
    
    # Support both --trajectory (backward compatible) and --trajectory1/--trajectory2
    parser.add_argument(
        "--trajectory",
        type=str,
        help="Path to trajectory .npz file (single robot mode, backward compatible)"
    )
    parser.add_argument(
        "--trajectory1",
        type=str,
        help="Path to first trajectory .npz file (dual robot mode)"
    )
    parser.add_argument(
        "--trajectory2",
        type=str,
        help="Path to second trajectory .npz file (dual robot mode)"
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=str(default_urdf_path),
        help=f"Path to URDF file (default: {default_urdf_path})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Viser server (default: 8080)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="MuJoCo timestep in seconds (default: 0.01)"
    )
    parser.add_argument(
        "--no-contacts",
        action="store_true",
        help="Don't show foot contact indicators"
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Don't show trajectory path"
    )
    parser.add_argument(
        "--ghost-frames",
        type=int,
        default=10,
        help="Maximum number of ghost frames to pre-allocate (default: 10)"
    )
    parser.add_argument(
        "--robot2-y-offset",
        type=float,
        default=-3.0,
        help="Y-axis offset for second robot in dual mode (default: -3.0)"
    )
    
    args = parser.parse_args()
    
    # Determine mode: single or dual trajectory
    dual_mode = False
    traj1_path = None
    traj2_path = None
    
    if args.trajectory1 and args.trajectory2:
        # Dual mode with explicit trajectory1/trajectory2
        dual_mode = True
        traj1_path = args.trajectory1
        traj2_path = args.trajectory2
    elif args.trajectory1 or args.trajectory2:
        # Only one trajectory specified with --trajectory1 or --trajectory2
        parser.error("When using --trajectory1 or --trajectory2, both must be specified")
    elif args.trajectory:
        # Single mode with backward compatible --trajectory
        dual_mode = False
        traj1_path = args.trajectory
    else:
        parser.error("Must specify either --trajectory (single mode) or both --trajectory1 and --trajectory2 (dual mode)")
    
    # Create visualizer
    viz = MujocoTrajVisualizer(
        urdf_path=args.urdf,
        port=args.port,
        dt=args.dt,
        show_contacts=not args.no_contacts,
        show_body_part_trajectory=not args.no_trajectory,
        num_ghost_frames=args.ghost_frames,
        dual_mode=dual_mode,
        robot2_y_offset=args.robot2_y_offset,
    )
    
    # Load trajectory data
    viz.load_trajectory(traj1_path, robot_idx=1)
    if dual_mode and traj2_path:
        viz.load_trajectory(traj2_path, robot_idx=2)
    
    # Start playback loop
    viz.play()

if __name__ == "__main__":
    """
    Usage:

    # Basic usage (single trajectory)
    python viser/viser_cheetah3_viz_trajs.py --trajectory body_trajs/mode_traj_data_cheetah3/your_run/trajectories_step_500000.npz

    # Dual trajectory mode - compare two trajectories side by side
    python viser/viser_cheetah3_viz_trajs.py --trajectory1 body_trajs/mode_traj_data_cheetah3/cheetah3-run-SAC-MPC-20260505-202924-percentage-25pct-seed1/trajectories_step_500000.npz --trajectory2 body_trajs/mode_traj_data_cheetah3/cheetah3-run-SAC-MPC-20260505-202313-percentage-0pct-seed1/trajectories_step_500000.npz

    # Specify custom URDF location
    python viser/viser_cheetah3_viz_trajs.py --trajectory path/to/trajectory.npz --urdf path/to/cheetah3.urdf

    # Use different port
    python viser/viser_cheetah3_viz_trajs.py --trajectory trajectory.npz --port 8090

    # Disable visualizations
    python viser/viser_cheetah3_viz_trajs.py --trajectory trajectory.npz --no-contacts --no-trajectory
    
    # Adjust robot2 y-offset in dual mode (default is -3.0)
    python viser/viser_cheetah3_viz_trajs.py \
        --trajectory1 path/to/traj1.npz \
        --trajectory2 path/to/traj2.npz \
        --robot2-y-offset -5.0
    
    # Using camera follow feature:
    # 1. Run the visualizer with any trajectory
    # 2. In the web UI, expand the "Camera Follow" panel
    # 3. Check "Follow Robot" to enable camera tracking
    # 4. Adjust sliders or use presets to change the camera angle:
    #    - Side View: Classic side-scrolling perspective
    #    - Behind View: Third-person following camera
    #    - Top-Down: Bird's eye view
    #    - 3/4 View: Isometric-style angle
    #    - Front View: Looking at the robot face-on
    
    """
    main()
