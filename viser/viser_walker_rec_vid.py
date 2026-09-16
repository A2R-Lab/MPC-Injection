"""
Record a high-quality viser video for a saved MuJoCo walker trajectory.

Open the viser URL in a browser when prompted. The script renders each selected
trajectory frame through that browser client, writes PNG frames, then encodes an
MP4 with ffmpeg.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import viser
from viser.extras import ViserUrdf


FOOT_NAMES = ("right_foot", "left_foot")
FOOT_COLORS = {
    "right_foot": (235, 54, 54),
    "left_foot": (40, 190, 90),
}
TRAIL_COLORS = {
    "torso": (0, 220, 255),
    "right_thigh": (255, 150, 0),
    "right_leg": (255, 210, 0),
    "right_foot": (235, 54, 54),
    "left_thigh": (180, 110, 255),
    "left_leg": (70, 130, 255),
    "left_foot": (40, 190, 90),
}


class WalkerTrajectoryVideoRecorder:
    """Viser scene and frame-capture helper for walker trajectory videos."""

    def __init__(
        self,
        urdf_path: Path,
        port: int,
        dt: float,
        show_contacts: bool,
        show_trails: bool,
        trail_bodies: list[str],
        trail_window: int,
        trail_line_width: float,
        contact_radius: float,
    ) -> None:
        self.server = viser.ViserServer(port=port)
        self.port = port
        self.dt = dt
        self.show_contacts = show_contacts
        self.show_trails = show_trails
        self.trail_bodies = trail_bodies
        self.trail_window = trail_window
        self.trail_line_width = trail_line_width
        self.contact_radius = contact_radius

        self.urdf_path = Path(urdf_path)
        self.trajectory_data: np.lib.npyio.NpzFile | None = None
        self.num_frames = 0

        self.world_node = self.server.scene.add_frame("/world", show_axes=False)
        self.urdf_handle = ViserUrdf(
            target=self.server,
            urdf_or_path=self.urdf_path,
            root_node_name="/world",
        )
        self.server.scene.add_grid(
            "/grid",
            width=200,
            height=200,
            position=(0.0, 0.0, 0.0),
            plane="xy",
        )

        self.contact_handles: dict[str, Any] = {}
        self.trail_handles: dict[str, Any] = {}

    def load_trajectory(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=True)
        required = ["timesteps", "joint_angles"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{npz_path} is missing required keys: {missing}")

        self.trajectory_data = data
        self.num_frames = int(data["timesteps"])

        if self.trail_bodies == ["all"]:
            self.trail_bodies = sorted(
                key.removeprefix("pos_")
                for key in data.files
                if key.startswith("pos_")
            )
        else:
            valid_bodies = []
            for body_name in self.trail_bodies:
                if f"pos_{body_name}" in data.files:
                    valid_bodies.append(body_name)
                else:
                    print(f"Warning: skipping missing body trail pos_{body_name}")
            self.trail_bodies = valid_bodies

        print(f"Loaded {npz_path}")
        print(f"  frames: {self.num_frames}")
        print(f"  duration at dt={self.dt:g}: {self.num_frames * self.dt:.3f} s")
        print(f"  trail bodies: {', '.join(self.trail_bodies) or 'none'}")
        if "contact_pct_right_foot" in data.files:
            print(f"  right foot contact: {float(data['contact_pct_right_foot']):.1f}%")
        if "contact_pct_left_foot" in data.files:
            print(f"  left foot contact: {float(data['contact_pct_left_foot']):.1f}%")

        self.update_visualization(0)

    def _get_actuated_joint_positions(self, frame_idx: int) -> np.ndarray:
        assert self.trajectory_data is not None
        qpos = self.trajectory_data["joint_angles"][frame_idx]
        return np.asarray(qpos[2:], dtype=float)

    def _get_root_state(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.trajectory_data is not None
        qpos = self.trajectory_data["joint_angles"][frame_idx]

        # Walker qpos layout follows the existing visualizer:
        # qpos[0] = rootz, qpos[1] = rootx, qpos[2] = rooty.
        root_pos = np.array([qpos[1], 0.0, qpos[0]], dtype=float)
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return root_pos, root_quat

    def _update_contacts(self, frame_idx: int) -> None:
        assert self.trajectory_data is not None
        if not self.show_contacts:
            for handle in self.contact_handles.values():
                handle.visible = False
            return

        for foot_name in FOOT_NAMES:
            contact_key = f"contact_{foot_name}"
            pos_key = f"pos_{foot_name}"
            if contact_key not in self.trajectory_data.files or pos_key not in self.trajectory_data.files:
                continue

            is_in_contact = bool(self.trajectory_data[contact_key][frame_idx])
            foot_pos = np.asarray(self.trajectory_data[pos_key][frame_idx], dtype=float)
            handle_name = f"/contact_{foot_name}"

            if handle_name not in self.contact_handles:
                self.contact_handles[handle_name] = self.server.scene.add_icosphere(
                    handle_name,
                    radius=self.contact_radius,
                    color=FOOT_COLORS[foot_name],
                    position=tuple(foot_pos),
                )

            handle = self.contact_handles[handle_name]
            handle.position = tuple(foot_pos)
            handle.visible = is_in_contact

    def _fade_colors(self, base_color: tuple[int, int, int], num_segments: int) -> np.ndarray:
        colors = np.zeros((num_segments, 2, 3), dtype=np.uint8)
        base = np.asarray(base_color, dtype=float)
        for idx in range(num_segments):
            fade = 0.25 + 0.75 * ((idx + 1) / max(1, num_segments))
            colors[idx] = np.asarray(base * fade, dtype=np.uint8)
        return colors

    def _update_body_trails(self, frame_idx: int) -> None:
        assert self.trajectory_data is not None
        if not self.show_trails:
            for handle in self.trail_handles.values():
                handle.visible = False
            return

        for body_name in self.trail_bodies:
            pos_key = f"pos_{body_name}"
            if pos_key not in self.trajectory_data.files:
                continue

            start_idx = 0
            if self.trail_window > 0:
                start_idx = max(0, frame_idx - self.trail_window + 1)
            positions = np.asarray(
                self.trajectory_data[pos_key][start_idx : frame_idx + 1],
                dtype=float,
            )

            line_name = f"/trail_{body_name}"
            handle = self.trail_handles.get(line_name)
            if len(positions) < 2:
                if handle is not None:
                    handle.visible = False
                continue

            points = np.stack([positions[:-1], positions[1:]], axis=1)
            colors = self._fade_colors(
                TRAIL_COLORS.get(body_name, (255, 255, 255)),
                len(points),
            )

            if handle is None:
                self.trail_handles[line_name] = self.server.scene.add_line_segments(
                    line_name,
                    points=points,
                    colors=colors,
                    line_width=self.trail_line_width,
                )
            else:
                handle.points = points
                handle.colors = colors
                handle.line_width = self.trail_line_width
                handle.visible = True

    def update_visualization(self, frame_idx: int) -> None:
        if self.trajectory_data is None:
            return

        frame_idx = int(np.clip(frame_idx, 0, self.num_frames - 1))
        root_pos, root_quat = self._get_root_state(frame_idx)
        joint_positions = self._get_actuated_joint_positions(frame_idx)

        with self.server.atomic():
            self.world_node.position = tuple(root_pos)
            self.world_node.wxyz = tuple(root_quat)
            self.urdf_handle.update_cfg(joint_positions)
            self._update_contacts(frame_idx)
            self._update_body_trails(frame_idx)

    def camera_pose(
        self,
        frame_idx: int,
        camera_distance: float,
        camera_side_offset: float,
        camera_height: float,
        look_at_forward: float,
        look_at_height: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        root_pos, _ = self._get_root_state(frame_idx)
        camera_position = root_pos + np.array(
            [camera_distance, camera_side_offset, camera_height],
            dtype=float,
        )
        look_at_position = root_pos + np.array(
            [look_at_forward, 0.0, look_at_height],
            dtype=float,
        )
        return camera_position, look_at_position


def parse_trail_bodies(value: str) -> list[str]:
    bodies = [item.strip() for item in value.split(",") if item.strip()]
    return bodies or ["torso", "right_foot", "left_foot"]


def apply_camera_preset(args: argparse.Namespace) -> None:
    if args.camera_preset == "custom":
        return
    presets = {
        "side_left": {
            "camera_distance": 0.0,
            "camera_side_offset": -5.0,
            "camera_height": 1.15,
            "look_at_forward": 0.8,
            "look_at_height": 0.9,
            "fov_deg": 45.0,
        },
        "side_right": { # Default
            "camera_distance": 0.0,
            "camera_side_offset": 5.0,
            "camera_height": 1.3,
            "look_at_forward": 0.0,
            "look_at_height": 1.0,
            "fov_deg": 45.0,
        },
        "behind": {
            "camera_distance": -5.0,
            "camera_side_offset": 0.0,
            "camera_height": 2.0,
            "look_at_forward": 2.0,
            "look_at_height": 0.8,
            "fov_deg": 55.0,
        },
        "three_quarter": {
            "camera_distance": -4.0,
            "camera_side_offset": -3.0,
            "camera_height": 2.8,
            "look_at_forward": 1.2,
            "look_at_height": 0.85,
            "fov_deg": 50.0,
        },
    }
    for name, value in presets[args.camera_preset].items():
        setattr(args, name, value)


def wait_for_client(server: viser.ViserServer, port: int) -> viser.ClientHandle:
    print(f"\nOpen http://localhost:{port}/?fixedDpr=1 in your browser.")
    print("Waiting for a viser web client...")
    while True:
        clients = server.get_clients()
        if clients:
            client = next(iter(clients.values()))
            break
        time.sleep(0.1)

    while True:
        try:
            _ = client.camera.update_timestamp
            break
        except AssertionError:
            time.sleep(0.05)

    print(f"Connected client {client.client_id}.")
    return client


def rgba_to_rgb(image: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected rendered image to have 3 dimensions, got {image.shape}")
    if image.shape[2] == 3:
        return np.asarray(image, dtype=np.uint8)
    if image.shape[2] != 4:
        raise ValueError(f"Expected RGB or RGBA render, got shape {image.shape}")

    rgb = image[..., :3].astype(np.float32)
    alpha = image[..., 3:4].astype(np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    return np.asarray(rgb * alpha + bg * (1.0 - alpha), dtype=np.uint8)


def encode_video(
    ffmpeg: str,
    frames_dir: Path,
    output_path: Path,
    fps: int,
    crf: int,
    preset: str,
    num_frames: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-frames:v",
        str(num_frames),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    print("\nEncoding video:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def default_output_path(trajectory_path: Path) -> Path:
    return trajectory_path.with_name(f"{trajectory_path.stem}_viser.mp4")


def frame_indices(start_frame: int, end_frame: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if end_frame <= start_frame:
        raise ValueError("--end-frame must be greater than --start-frame")
    return list(range(start_frame, end_frame, stride))


def record_frames(
    recorder: WalkerTrajectoryVideoRecorder,
    client: viser.ClientHandle,
    indices: list[int],
    frames_dir: Path,
    args: argparse.Namespace,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    fov_rad = np.deg2rad(args.fov_deg)
    background = tuple(int(x) for x in args.background_rgb)

    if not args.manual_camera:
        client.camera.up_direction = (0.0, 0.0, 1.0)

    progress_every = max(1, len(indices) // 20)
    for output_idx, traj_frame_idx in enumerate(indices):
        recorder.update_visualization(traj_frame_idx)

        if args.manual_camera:
            recorder.server.flush()
            client.flush()
            time.sleep(args.render_pause)
            rendered = client.get_render(
                height=args.render_height,
                width=args.render_width,
                transport_format="png",
            )
        else:
            camera_position, look_at_position = recorder.camera_pose(
                traj_frame_idx,
                args.camera_distance,
                args.camera_side_offset,
                args.camera_height,
                args.look_at_forward,
                args.look_at_height,
            )
            with client.atomic():
                client.camera.position = camera_position
                client.camera.look_at = look_at_position
                client.camera.fov = fov_rad
            recorder.server.flush()
            client.flush()
            time.sleep(args.render_pause)
            rendered = client.get_render(
                height=args.render_height,
                width=args.render_width,
                position=camera_position,
                wxyz=np.asarray(client.camera.wxyz, dtype=float),
                fov=fov_rad,
                transport_format="png",
            )

        rgb = rgba_to_rgb(rendered, background)
        frame_path = frames_dir / f"{output_idx:06d}.png"
        iio.imwrite(frame_path, rgb)

        if output_idx == 0 or (output_idx + 1) % progress_every == 0 or output_idx + 1 == len(indices):
            print(
                f"Rendered {output_idx + 1}/{len(indices)} frames "
                f"(trajectory frame {traj_frame_idx})"
            )


def main() -> None:
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    default_urdf_path = workspace_root / "mpc_rl" / "tasks" / "walker" / "walker_modified.urdf"

    parser = argparse.ArgumentParser(
        description="Record a viser MP4 of a saved MuJoCo walker trajectory.",
    )
    parser.add_argument("--trajectory", type=Path, required=True, help="Path to trajectory .npz file")
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument("--urdf", type=Path, default=default_urdf_path, help=f"Walker URDF path (default: {default_urdf_path})")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    parser.add_argument("--dt", type=float, default=0.0025, help="Trajectory timestep in seconds (default: 0.0025)")

    parser.add_argument("--start-frame", type=int, default=0, help="First trajectory frame to render (default: 0)")
    parser.add_argument("--end-frame", type=int, help="Exclusive end frame; defaults to trajectory length")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth trajectory frame (default: 1)")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate (default: 30)")
    parser.add_argument("--render-width", type=int, default=1920, help="Render width in pixels (default: 1920)")
    parser.add_argument("--render-height", type=int, default=1080, help="Render height in pixels (default: 1080)")
    parser.add_argument("--render-pause", type=float, default=0.02, help="Seconds to wait after scene updates before capture (default: 0.02)")

    parser.add_argument("--no-contacts", action="store_true", help="Hide foot contact indicators")
    parser.add_argument("--no-trails", action="store_true", help="Hide body-part trajectory trails")
    parser.add_argument(
        "--trail-bodies",
        type=parse_trail_bodies,
        default=parse_trail_bodies("torso,right_foot,left_foot"),
        help="Comma-separated body names to trail, or 'all' (default: torso,right_foot,left_foot)",
    )
    parser.add_argument("--trail-window", type=int, default=0, help="Trail length in frames; 0 means full history (default: 0)")
    parser.add_argument("--trail-line-width", type=float, default=3.0, help="Trail line width (default: 3.0)")
    parser.add_argument("--contact-radius", type=float, default=0.1, help="Foot contact sphere radius (default: 0.1)")

    parser.add_argument(
        "--camera-preset",
        choices=["side_left", "side_right", "behind", "three_quarter", "custom"],
        default="side_right",
        help="Scripted camera preset (default: side_right)",
    )
    parser.add_argument("--manual-camera", action="store_true", help="Use the browser camera as-is instead of scripted follow camera")
    parser.add_argument("--camera-distance", type=float, default=0.0, help="Camera X offset from walker root for custom camera")
    parser.add_argument("--camera-side-offset", type=float, default=-5.0, help="Camera Y offset from walker root for custom camera")
    parser.add_argument("--camera-height", type=float, default=1.15, help="Camera Z offset from walker root for custom camera")
    parser.add_argument("--look-at-forward", type=float, default=0.8, help="Look-at X offset from walker root")
    parser.add_argument("--look-at-height", type=float, default=0.9, help="Look-at Z offset from walker root")
    parser.add_argument("--fov-deg", type=float, default=45.0, help="Vertical FOV for scripted camera (default: 45)")

    parser.add_argument("--frames-dir", type=Path, help="Directory for rendered PNG frames")
    parser.add_argument("--keep-frames", action="store_true", help="Keep rendered PNG frames")
    parser.add_argument("--skip-encode", action="store_true", help="Only render PNG frames; do not run ffmpeg")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: ffmpeg)")
    parser.add_argument("--crf", type=int, default=16, help="libx264 CRF quality, lower is higher quality (default: 16)")
    parser.add_argument("--encode-preset", default="slow", help="libx264 preset (default: slow)")
    parser.add_argument(
        "--background-rgb",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=(255, 255, 255),
        help="RGB background for compositing PNG renders, e.g. 255,255,255",
    )
    parser.add_argument("--no-wait-for-enter", action="store_true", help="Start recording as soon as a client connects")

    args = parser.parse_args()
    apply_camera_preset(args)

    if len(args.background_rgb) != 3 or any(channel < 0 or channel > 255 for channel in args.background_rgb):
        parser.error("--background-rgb must contain three values in [0, 255]")

    output_path = args.output or default_output_path(args.trajectory)

    recorder = WalkerTrajectoryVideoRecorder(
        urdf_path=args.urdf,
        port=args.port,
        dt=args.dt,
        show_contacts=not args.no_contacts,
        show_trails=not args.no_trails,
        trail_bodies=args.trail_bodies,
        trail_window=args.trail_window,
        trail_line_width=args.trail_line_width,
        contact_radius=args.contact_radius,
    )
    recorder.load_trajectory(args.trajectory)

    end_frame = args.end_frame if args.end_frame is not None else recorder.num_frames
    start_frame = int(np.clip(args.start_frame, 0, recorder.num_frames - 1))
    end_frame = int(np.clip(end_frame, start_frame + 1, recorder.num_frames))
    indices = frame_indices(start_frame, end_frame, args.stride)
    video_duration = len(indices) / args.fps
    sim_duration = (indices[-1] - indices[0] + 1) * args.dt

    if not args.manual_camera:
        camera_position, look_at_position = recorder.camera_pose(
            indices[0],
            args.camera_distance,
            args.camera_side_offset,
            args.camera_height,
            args.look_at_forward,
            args.look_at_height,
        )
        recorder.server.initial_camera.position = tuple(camera_position)
        recorder.server.initial_camera.look_at = tuple(look_at_position)
        recorder.server.initial_camera.fov = float(np.deg2rad(args.fov_deg))

    client = wait_for_client(recorder.server, args.port)
    print(
        f"Prepared {len(indices)} frames from trajectory frames "
        f"[{indices[0]}, {indices[-1]}] with stride {args.stride}."
    )
    print(f"Output video duration: {video_duration:.2f} s at {args.fps} fps")
    print(f"Trajectory time covered: {sim_duration:.2f} s")
    print(
        "Requested render size is "
        f"{args.render_width}x{args.render_height}; keep the browser viewport at least that large."
    )

    if not args.no_wait_for_enter:
        try:
            input("Press Enter when the browser view is loaded and ready to record...")
        except EOFError:
            print("No interactive stdin available; starting recording now.")

    cleanup_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is not None:
        frames_dir = args.frames_dir
    elif args.keep_frames or args.skip_encode:
        frames_dir = output_path.with_suffix("").with_name(f"{output_path.stem}_frames")
    else:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="viser_walker_frames_")
        frames_dir = Path(cleanup_dir.name)

    try:
        record_frames(recorder, client, indices, frames_dir, args)
        if args.skip_encode:
            print(f"\nRendered PNG frames to {frames_dir}")
        else:
            encode_video(
                ffmpeg=args.ffmpeg,
                frames_dir=frames_dir,
                output_path=output_path,
                fps=args.fps,
                crf=args.crf,
                preset=args.encode_preset,
                num_frames=len(indices),
            )
            print(f"\nWrote video: {output_path}")
            if args.keep_frames or args.frames_dir is not None:
                print(f"Kept PNG frames: {frames_dir}")
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()


if __name__ == "__main__":
    main()
