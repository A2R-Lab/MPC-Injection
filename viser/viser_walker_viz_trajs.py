"""
Script to visualize the trajectories of the walker environment using viser.

Inspired by Se Hwan Jeon's script used in his Residual MPC paper: https://arxiv.org/abs/2510.12717

For us the plan is as follows:
1) Convert the walker MuJoCo model to URDF using cnvrt_mjcf_to_urdf.py
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

class MujocoTrajVisualizer:
    """
    Visualize saved MuJoCo trajectories using viser and an MJCF-to-URDF converted model.
    """

    def __init__(
            self,
            urdf_path: str,
            port: int = 8080,
            dt: float = 0.0025,
            show_contacts: bool = True,
            show_body_part_trajectory: bool = True,
            num_ghost_frames: int = 5,
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
        """
        self.server = viser.ViserServer(port=port)
        self.dt = dt
        self.show_contacts = show_contacts
        self.show_body_part_trajectory = show_body_part_trajectory
        self.port = port  # Store port for later use
        self.max_ghost_frames = num_ghost_frames

        # Convert to Path for ViserUrdf
        urdf_path = Path(urdf_path)
        self.urdf_path = urdf_path

        # Load URDF to get joint info
        self.urdf = yourdfpy.URDF.load(str(urdf_path), load_collision_meshes=False)

        # Create world frame for main robot
        self.world_node = self.server.scene.add_frame(name="/world", show_axes=False)

        # Load URDF visualization - ViserUrdf expects a Path object
        self.urdf_handle = ViserUrdf(
            target=self.server,
            urdf_or_path=urdf_path,
            root_node_name="/world",
        )

        # Create ghost frame instances (for showing multiple frames at once)
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
                "Speed", min=0.1, max=1.0, step=0.05, initial_value=0.1,
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

    def load_trajectory(self, npz_path: str):
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
        """
        data = np.load(npz_path, allow_pickle=True)
        self.trajectory_data = data

        # Update frame slider range
        num_frames = data['timesteps']
        self.frame_slider.max = num_frames - 1

        print(f"Loaded trajectory with {num_frames} frames.")
        print(f"Available data: {list(data.keys())}")

        # Display stats
        if self.show_stats.value:
            self._update_statistics()
        
        # Initializat body part trajectory path
        # NOTE: Change this for whatever body part you want to track
        if 'pos_torso' in data.files:
            self.trajectory_path = data['pos_torso']

        # Initialize to first frame
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
        if 'contact_pct_right_foot' in data.files:
            stats_lines.append(f"Right Foot Contact: {data['contact_pct_right_foot']:.1f}%")
        if 'contact_pct_left_foot' in data.files:
            stats_lines.append(f"Left Foot Contact: {data['contact_pct_left_foot']:.1f}%")
        
        self.stats_text.value = "\n".join(stats_lines)
    
    def _get_actuated_joint_positions(self, frame_idx: int) -> np.ndarray:
        """
        Extract actuated joint positions from qpos data

        For the walker, the qpos structure is:
        - qpos[0]: rootz (z position - slide joint) - treated as fixed in URDF
        - qpos[1]: rootx (x position - slide joint) - treated as fixed in URDF
        - qpos[2]: rooty (rotation around y-axis - hinge joint) - actuated in URDF
        - qpos[3:9]: actuated joint positions (6 joints for the walker)
            - right_hip, right_knee, right_ankle
            - left_hip, left_knee, left_ankle

        The URDF actuated joints are: rooty, right_hip, right_knee, right_ankle, 
        left_hip, left_knee, left_ankle (7 total)

        Returns:
            np.ndarray of actuated joint angles in the expected order for yourdfpy
        """
        qpos = self.trajectory_data['joint_angles'][frame_idx]

        # Include rooty (qpos[2]) and the 6 leg joints (qpos[3:9])
        # Total: 7 actuated joints matching URDF structure
        actuated_joint_angles = qpos[2:]

        return actuated_joint_angles

    def _get_root_state(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract root position and orientation from qpos data

        For the walker:
        - qpos[0]: rootz (z position - vertical)
        - qpos[1]: rootx (x position - horizontal forward/back)

        Note: rooty is handled by the URDF joint system, not as part of root state

        Returns:
            Tuple of (pos, quat) where:
            - pos is (x, y, z) position (y=0 for 2D walker)
            - quat is identity quaternion (rotation handled by URDF rooty joint)
        """
        qpos = self.trajectory_data['joint_angles'][frame_idx]

        # Extract root position (x, y, z) - note: y is always 0 for 2D walker
        # qpos[0] is z (height), qpos[1] is x (forward)
        root_pos = np.array([qpos[1], 0.0, qpos[0]])

        # Identity quaternion since rotation is handled by rooty joint in URDF
        root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        return root_pos, root_quat
    
    def _set_ghost_visibility(self, visible: bool, num_visible: int = None):
        """
        Set visibility of ghost frame instances.
        
        Args:
            visible: Whether ghosts should be visible
            num_visible: Number of ghosts to show (if None, show/hide all)
        """
        for i, (ghost_world, ghost_urdf) in enumerate(zip(self._ghost_frames, self._ghost_urdf_handles)):
            if num_visible is not None:
                is_visible = visible and (i < num_visible)
            else:
                is_visible = visible
            
            # Set visibility by moving far away if hidden, or updating position if visible
            # ViserUrdf doesn't have direct visibility control, so we use position
            if not is_visible:
                ghost_world.position = (0, 0, -1000)  # Move far below ground
    
    def _update_ghost_frame(self, ghost_idx: int, frame_idx: int, opacity_factor: float = 1.0):
        """
        Update a single ghost frame to show a specific trajectory frame.
        
        Args:
            ghost_idx: Index of the ghost instance to update
            frame_idx: Trajectory frame to display
            opacity_factor: Alpha value for transparency (0.0 = fully transparent, 1.0 = opaque)
        """
        if ghost_idx >= len(self._ghost_frames):
            return
        
        # Check if we need to recreate the URDF with a different alpha
        # We recreate if alpha changed significantly to update transparency
        alpha_changed = abs(self._ghost_alphas[ghost_idx] - opacity_factor) > 0.05
        
        if alpha_changed:
            # Remove old URDF handle
            # ViserUrdf doesn't have a built-in remove method, but replacing it works
            # Create new URDF with updated alpha
            self._ghost_urdf_handles[ghost_idx] = ViserUrdf(
                target=self.server,
                urdf_or_path=self.urdf_path,
                root_node_name=f"/ghost_{ghost_idx}",
                mesh_color_override=(1.0, 0.5, 0.0, opacity_factor),  # Orange with custom alpha
            )
            self._ghost_alphas[ghost_idx] = opacity_factor
        
        # Get root position and orientation for this frame
        root_pos, root_quat = self._get_root_state(frame_idx)
        
        # Update ghost world frame position
        self._ghost_frames[ghost_idx].position = root_pos
        self._ghost_frames[ghost_idx].wxyz = root_quat
        
        # Get actuated joint positions
        joint_positions = self._get_actuated_joint_positions(frame_idx)
        
        # Update ghost URDF config
        self._ghost_urdf_handles[ghost_idx].update_cfg(joint_positions)
        
        # Update ghost foot contacts
        self._update_ghost_contacts(ghost_idx, frame_idx)
    
    def _update_ghost_contacts(self, ghost_idx: int, frame_idx: int):
        """
        Update foot contact visualization for a specific ghost frame.
        
        Args:
            ghost_idx: Index of the ghost instance
            frame_idx: Trajectory frame to display contacts for
        """
        if not self.show_contacts_checkbox.value:
            # Hide this ghost's contact indicators
            for handle in self._ghost_contact_indicators[ghost_idx].values():
                handle.visible = False
            return
        
        foot_names = ['right_foot', 'left_foot']
        
        for foot_name in foot_names:
            contact_key = f'contact_{foot_name}'
            pos_key = f'pos_{foot_name}'
            
            if contact_key not in self.trajectory_data.files:
                continue
            
            is_in_contact = self.trajectory_data[contact_key][frame_idx]
            foot_pos = self.trajectory_data[pos_key][frame_idx]
            
            # Create or update contact indicator for this ghost
            indicator_name = f'/ghost_{ghost_idx}_contact_{foot_name}'
            
            if indicator_name not in self._ghost_contact_indicators[ghost_idx]:
                # Create a sphere for contact indicator
                # Use slightly transparent colors for ghost contacts
                self._ghost_contact_indicators[ghost_idx][indicator_name] = self.server.scene.add_icosphere(
                    indicator_name,
                    radius=0.08,  # Slightly smaller than main contacts
                    color=(0, 200, 0) if foot_name == 'left_foot' else (200, 0, 0),
                    position=foot_pos
                )
            
            # Update position and visibility
            self._ghost_contact_indicators[ghost_idx][indicator_name].position = foot_pos
            self._ghost_contact_indicators[ghost_idx][indicator_name].visible = bool(is_in_contact)
    
    def _hide_ghost_contacts(self, ghost_idx: int):
        """
        Hide all contact indicators for a specific ghost.
        
        Args:
            ghost_idx: Index of the ghost instance
        """
        if ghost_idx < len(self._ghost_contact_indicators):
            for handle in self._ghost_contact_indicators[ghost_idx].values():
                handle.visible = False
    
    def _update_ghost_visualization(self, current_frame: int):
        """
        Update all ghost frame visualizations based on current frame and settings.
        
        Args:
            current_frame: The main/current frame being displayed
        """
        if not self.show_ghosts_checkbox.value or self.trajectory_data is None:
            self._set_ghost_visibility(visible=False)
            # Also hide all ghost contacts
            for ghost_idx in range(self.max_ghost_frames):
                self._hide_ghost_contacts(ghost_idx)
            return
        
        num_ghosts = int(self.num_ghosts_slider.value)
        interval = int(self.ghost_interval_slider.value)
        direction = self.ghost_direction.value
        num_frames = int(self.trajectory_data['timesteps'])
        
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
                
                self._update_ghost_frame(ghost_idx, frame_idx, opacity_factor)
        
        # Hide remaining ghosts and their contacts
        for ghost_idx in range(len(ghost_frame_indices), self.max_ghost_frames):
            self._ghost_frames[ghost_idx].position = (0, 0, -1000)
            self._hide_ghost_contacts(ghost_idx)
            # Reset alpha tracking for hidden ghosts
            self._ghost_alphas[ghost_idx] = 1.0
    
    def _update_contact_visualization(self, frame_idx: int):
        """
        Update visualization of foot contacts
        """
        if not self.show_contacts_checkbox.value:
            # Hide contact indicators
            for handle in self._contact_indicators.values():
                handle.visible = False
            return
        
        # Show contact indicators for the feet
        foot_names = ['right_foot', 'left_foot']

        for foot_name in foot_names:
            contact_key = f'contact_{foot_name}'
            pos_key = f'pos_{foot_name}'

            if contact_key not in self.trajectory_data.files:
                continue

            is_in_contact = self.trajectory_data[contact_key][frame_idx]
            foot_pos = self.trajectory_data[pos_key][frame_idx]

            # Create or update contact indicator
            indicator_name = f'/contact_{foot_name}'

            if indicator_name not in self._contact_indicators:
                # Create a sphere for contact indicator
                self._contact_indicators[indicator_name] = self.server.scene.add_icosphere(
                    indicator_name,
                    radius=0.1,
                    color=(0, 255, 0) if foot_name == 'left_foot' else (255, 0, 0),
                    position=foot_pos
                )

            # Update position and visibility
            self._contact_indicators[indicator_name].position = foot_pos
            self._contact_indicators[indicator_name].visible = bool(is_in_contact)
        
    def _update_trajectory_path(self, frame_idx: int):
        """
        Update visualization of body part trajectory path
        """
        if not self.show_body_part_trajectory_checkbox.value or self.trajectory_path is None:
            if self._trajectory_line is not None:
                self._trajectory_line.visible = False
            return
        
        # Show trajectory up to current frame
        path_positions = self.trajectory_path[:frame_idx+1]

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
            colors[i] = [[0, color_val, color_val], [0, color_val, color_val]]  # Cyan

        if self._trajectory_line is None:
            self._trajectory_line = self.server.scene.add_line_segments(
                "/trajectory_line",
                points=points,
                colors=colors,
                line_width=2.0,
            )
        else:
            self._trajectory_line.points = points
            self._trajectory_line.colors = colors
            self._trajectory_line.visible = True
    
    def update_visualization(self, frame_idx: int):
        """
        Update the visualization to show a specific frame
        
        Args:
            frame_idx: Frame index to display
        """
        if self.trajectory_data is None:
            return
        
        # Clamp frame index
        frame_idx = int(np.clip(frame_idx, 0, self.trajectory_data['timesteps'] - 1))

        # Get root position and orientation from qpos
        root_pos, root_quat = self._get_root_state(frame_idx)

        # Update world node position (floating base approach)
        # Why move the world frame instead of keeping it static?
        # - The URDF conversion marks rootx/rootz as type="fixed" (not movable joints)
        # - We can't update their positions via joint angles since they don't exist as actuated joints
        # - Instead, we use the world frame as a "floating base carrier" for the entire robot
        # - The world frame translates to follow the robot's global position
        # - Only the relative joint angles (rooty, legs) are updated via update_cfg()
        # This is a common pattern for floating-base robots (humanoids, quadrupeds) where
        # the base can translate freely but isn't part of the actuated joint chain.
        self.world_node.position = root_pos
        self.world_node.wxyz = root_quat

        # Get actuated joint positions
        joint_positions = self._get_actuated_joint_positions(frame_idx)

        # Update URDF config
        with self.server.atomic():
            self.urdf_handle.update_cfg(joint_positions)

        # Update contact visualization
        self._update_contact_visualization(frame_idx)

        # Update body part trajectory path
        self._update_trajectory_path(frame_idx)
        
        # Update ghost frames (multiple overlaid frames)
        self._update_ghost_visualization(frame_idx)
    
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
    default_urdf_path = workspace_root / "mpc_rl" / "tasks" / "walker" / "walker_modified.urdf"
    
    parser = argparse.ArgumentParser(
        description="Visualize MuJoCo walker trajectories using Viser"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="Path to trajectory .npz file"
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
        default=0.0025,
        help="MuJoCo timestep in seconds (default: 0.0025)"
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
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = MujocoTrajVisualizer(
        urdf_path=args.urdf,
        port=args.port,
        dt=args.dt,
        show_contacts=not args.no_contacts,
        show_body_part_trajectory=not args.no_trajectory,
        num_ghost_frames=args.ghost_frames,
    )
    
    # Load trajectory data
    viz.load_trajectory(args.trajectory)
    
    # Start playback loop
    viz.play()

if __name__ == "__main__":
    """
    Usage:

    # Basic usage
    python mujoco_trajectory_visualizer.py --trajectory model_traj_data/your_run/trajectories_step_500000.npz

    # Specify custom URDF location
    python mujoco_trajectory_visualizer.py --trajectory path/to/trajectory.npz --urdf path/to/walker.urdf

    # Use different port
    python mujoco_trajectory_visualizer.py --trajectory trajectory.npz --port 8090

    # Disable visualizations
    python mujoco_trajectory_visualizer.py --trajectory trajectory.npz --no-contacts --no-trajectory
    
    """
    main()