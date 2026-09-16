"""
Record a high-quality viser video for a saved MuJoCo quadruped trajectory.

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
import mujoco
import numpy as np
import viser
from viser import transforms as tf

from viser_quadruped_viz_trajs import (
    COM_COLOR,
    DEFAULT_SOURCE_MODEL_PATH,
    FOOT_COLORS,
    RobotMeshInstance,
    _build_visual_geom_specs,
    _xmat_to_wxyz,
)


DEFAULT_TRAIL_BODIES = "com,feet"


class QuadrupedTrajectoryVideoRecorder:
    """Viser scene and frame-capture helper for quadruped trajectory videos."""

    def __init__(
        self,
        source_model_path: Path,
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
        self.server.scene.set_up_direction("+z")
        self.server.scene.configure_default_lights(enabled=False, cast_shadow=False)
        self.server.scene.add_light_ambient(
            "/lights/ambient",
            color=(255, 255, 255),
            intensity=1.65,
        )
        self.server.scene.add_light_hemisphere(
            "/lights/hemi",
            sky_color=(255, 255, 255),
            ground_color=(188, 195, 205),
            intensity=1.15,
        )
        self.server.scene.add_light_directional(
            "/lights/key",
            color=(255, 245, 235),
            intensity=1.8,
            cast_shadow=False,
            wxyz=tf.SO3.from_rpy_radians(-0.9, 0.2, -0.7).wxyz,
        )
        self.server.scene.add_light_directional(
            "/lights/fill",
            color=(220, 235, 255),
            intensity=0.9,
            cast_shadow=False,
            wxyz=tf.SO3.from_rpy_radians(0.65, -0.3, 2.4).wxyz,
        )

        self.port = port
        self.dt = dt
        self.show_contacts = show_contacts
        self.show_trails = show_trails
        self.trail_bodies = trail_bodies
        self.trail_window = trail_window
        self.trail_line_width = trail_line_width
        self.contact_radius = contact_radius

        self.source_model_path = Path(source_model_path).resolve()
        self.render_model, self.visual_geom_specs = _build_visual_geom_specs(self.source_model_path)
        self.render_data = mujoco.MjData(self.render_model)
        self.robot_instance = self._create_robot_mesh_instance("/robot")

        self.server.scene.add_grid(
            "/grid",
            width=200,
            height=200,
            position=(0.0, 0.0, 0.0),
            plane="xy",
            shadow_opacity=0.0,
        )

        self.trajectory_data: np.lib.npyio.NpzFile | None = None
        self.num_frames = 0
        self.foot_names: list[str] = []
        self.body_names: list[str] = []
        self.body_name_to_index: dict[str, int] = {}
        self.contact_handles: dict[str, Any] = {}
        self.trail_handles: dict[str, Any] = {}

    def _create_robot_mesh_instance(self, root_name: str) -> RobotMeshInstance:
        root = self.server.scene.add_frame(root_name, show_axes=False)
        geom_frames: dict[int, Any] = {}
        mesh_handles: dict[int, Any] = {}

        for spec in self.visual_geom_specs:
            frame_name = f"{root_name}/geom_{spec.geom_id}"
            mesh_name = f"{frame_name}/mesh"
            geom_frames[spec.geom_id] = self.server.scene.add_frame(frame_name, show_axes=False)
            mesh_handles[spec.geom_id] = self.server.scene.add_mesh_simple(
                mesh_name,
                vertices=spec.vertices,
                faces=spec.faces,
                color=spec.color,
                opacity=spec.opacity,
                material="standard",
                flat_shading=False,
                cast_shadow=False,
                receive_shadow=False,
            )

        return RobotMeshInstance(
            root=root,
            geom_frames=geom_frames,
            mesh_handles=mesh_handles,
            contact_handles={},
        )

    def load_trajectory(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=True)
        if "qpos_ctrl" not in data.files and "qpos" not in data.files:
            raise ValueError(f"{npz_path} is missing qpos_ctrl or qpos")

        self.trajectory_data = data
        self.num_frames = self._num_frames()
        if "frame_dt" in data.files:
            self.dt = float(data["frame_dt"])
        elif "control_dt" in data.files:
            self.dt = float(data["control_dt"])

        self.foot_names = self._load_name_list("foot_names", ["FL", "FR", "RL", "RR"])
        self.body_names = self._load_name_list("body_names", [])
        self.body_name_to_index = {name: idx for idx, name in enumerate(self.body_names)}
        self.trail_bodies = self._expand_and_validate_trail_bodies(self.trail_bodies)

        print(f"Loaded {npz_path}")
        print(f"  frames: {self.num_frames}")
        print(f"  duration at dt={self.dt:g}: {self.num_frames * self.dt:.3f} s")
        print(f"  model xml: {self.source_model_path}")
        print(f"  trail bodies: {', '.join(self.trail_bodies) or 'none'}")
        self._print_contact_summary()

        self.update_visualization(0)

    def _load_name_list(self, key: str, fallback: list[str]) -> list[str]:
        assert self.trajectory_data is not None
        if key not in self.trajectory_data.files:
            return list(fallback)
        return [str(name) for name in self.trajectory_data[key]]

    def _num_frames(self) -> int:
        assert self.trajectory_data is not None
        if "qpos_ctrl" in self.trajectory_data.files:
            return int(self.trajectory_data["qpos_ctrl"].shape[0])
        if "timesteps" in self.trajectory_data.files:
            return int(self.trajectory_data["timesteps"])
        return int(self._qpos_ctrl().shape[0])

    def _qpos_ctrl(self) -> np.ndarray:
        assert self.trajectory_data is not None
        if "qpos_ctrl" in self.trajectory_data.files:
            return np.asarray(self.trajectory_data["qpos_ctrl"], dtype=float)
        qpos = np.asarray(self.trajectory_data["qpos"], dtype=float)
        decimation = int(self.trajectory_data["decimation"]) if "decimation" in self.trajectory_data.files else 4
        return qpos[:, ::decimation].T

    def _get_root_state(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        qpos = self._qpos_ctrl()[frame_idx]
        return np.asarray(qpos[:3], dtype=float), np.asarray(qpos[3:7], dtype=float)

    def _apply_qpos_to_robot(self, qpos: np.ndarray) -> None:
        self.render_data.qpos[:] = qpos
        mujoco.mj_forward(self.render_model, self.render_data)

        for spec in self.visual_geom_specs:
            frame = self.robot_instance.geom_frames[spec.geom_id]
            frame.position = np.array(self.render_data.geom_xpos[spec.geom_id], copy=True)
            frame.wxyz = _xmat_to_wxyz(self.render_data.geom_xmat[spec.geom_id])

    def _print_contact_summary(self) -> None:
        assert self.trajectory_data is not None
        if "foot_contacts" not in self.trajectory_data.files:
            return

        contacts = np.asarray(self.trajectory_data["foot_contacts"], dtype=bool)
        if contacts.ndim != 2:
            return

        for foot_idx, foot_name in enumerate(self.foot_names):
            if foot_idx < contacts.shape[1]:
                print(f"  {foot_name} contact: {100.0 * float(np.mean(contacts[:, foot_idx])):.1f}%")

    def _canonical_trail_name(self, name: str) -> str | None:
        assert self.trajectory_data is not None
        lower_name = name.lower()

        if lower_name in {"com", "center_of_mass", "center-of-mass"}:
            return "com" if "com_positions" in self.trajectory_data.files else None
        if lower_name == "base" and "base_positions" in self.trajectory_data.files:
            return "base"

        for foot_name in self.foot_names:
            if lower_name == foot_name.lower():
                return foot_name if "foot_positions" in self.trajectory_data.files else None
        for body_name in self.body_names:
            if lower_name == body_name.lower():
                return body_name if "body_positions" in self.trajectory_data.files else None
        return None

    def _expand_and_validate_trail_bodies(self, requested_bodies: list[str]) -> list[str]:
        assert self.trajectory_data is not None
        expanded: list[str] = []

        def add_name(name: str) -> None:
            if name not in expanded:
                expanded.append(name)

        for body_name in requested_bodies:
            lower_name = body_name.lower()
            if lower_name == "all":
                if "com_positions" in self.trajectory_data.files:
                    add_name("com")
                if "foot_positions" in self.trajectory_data.files:
                    for foot_name in self.foot_names:
                        add_name(foot_name)
                if "body_positions" in self.trajectory_data.files:
                    for saved_body_name in self.body_names:
                        add_name(saved_body_name)
                continue
            if lower_name == "feet":
                if "foot_positions" in self.trajectory_data.files:
                    for foot_name in self.foot_names:
                        add_name(foot_name)
                else:
                    print("Warning: skipping foot trails because foot_positions is missing")
                continue

            canonical_name = self._canonical_trail_name(body_name)
            if canonical_name is None:
                print(f"Warning: skipping missing trail body {body_name}")
                continue
            add_name(canonical_name)

        return expanded

    def _trail_color(self, trail_name: str) -> tuple[int, int, int]:
        if trail_name == "com":
            return tuple(int(x) for x in COM_COLOR)
        if trail_name in FOOT_COLORS:
            return tuple(int(x) for x in FOOT_COLORS[trail_name])

        palette = (
            (0, 220, 255),
            (255, 150, 0),
            (255, 210, 0),
            (180, 110, 255),
            (70, 130, 255),
            (40, 190, 90),
        )
        return palette[sum(ord(char) for char in trail_name) % len(palette)]

    def _trail_positions(self, trail_name: str, window: slice) -> np.ndarray | None:
        assert self.trajectory_data is not None

        if trail_name == "com" and "com_positions" in self.trajectory_data.files:
            return np.asarray(self.trajectory_data["com_positions"][window], dtype=float)
        if trail_name == "base" and "base_positions" in self.trajectory_data.files:
            return np.asarray(self.trajectory_data["base_positions"][window], dtype=float)
        if trail_name in self.foot_names and "foot_positions" in self.trajectory_data.files:
            return np.asarray(
                self.trajectory_data["foot_positions"][window, self.foot_names.index(trail_name), :],
                dtype=float,
            )
        if trail_name in self.body_name_to_index and "body_positions" in self.trajectory_data.files:
            return np.asarray(
                self.trajectory_data["body_positions"][window, self.body_name_to_index[trail_name], :],
                dtype=float,
            )
        return None

    def _update_contacts(self, frame_idx: int) -> None:
        assert self.trajectory_data is not None
        if (
            not self.show_contacts
            or "foot_positions" not in self.trajectory_data.files
            or "foot_contacts" not in self.trajectory_data.files
        ):
            for handle in self.contact_handles.values():
                handle.visible = False
            return

        foot_positions = np.asarray(self.trajectory_data["foot_positions"][frame_idx], dtype=float)
        foot_contacts = np.asarray(self.trajectory_data["foot_contacts"][frame_idx], dtype=bool)

        for foot_idx, foot_name in enumerate(self.foot_names):
            if foot_idx >= len(foot_positions) or foot_idx >= len(foot_contacts):
                continue

            handle_name = f"/contact_{foot_name}"
            if handle_name not in self.contact_handles:
                self.contact_handles[handle_name] = self.server.scene.add_icosphere(
                    handle_name,
                    radius=self.contact_radius,
                    color=self._trail_color(foot_name),
                    position=tuple(foot_positions[foot_idx]),
                )

            handle = self.contact_handles[handle_name]
            handle.position = tuple(foot_positions[foot_idx])
            handle.visible = bool(foot_contacts[foot_idx])

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

        start_idx = 0
        if self.trail_window > 0:
            start_idx = max(0, frame_idx - self.trail_window + 1)
        window = slice(start_idx, frame_idx + 1)

        for body_name in self.trail_bodies:
            positions = self._trail_positions(body_name, window)
            if positions is None:
                continue

            line_name = f"/trail_{body_name}"
            handle = self.trail_handles.get(line_name)
            if len(positions) < 2:
                if handle is not None:
                    handle.visible = False
                continue

            points = np.stack([positions[:-1], positions[1:]], axis=1)
            colors = self._fade_colors(self._trail_color(body_name), len(points))

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
        qpos = self._qpos_ctrl()[frame_idx]

        with self.server.atomic():
            self._apply_qpos_to_robot(qpos)
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
    return bodies or parse_trail_bodies(DEFAULT_TRAIL_BODIES)


def apply_camera_preset(args: argparse.Namespace) -> None:
    if args.camera_preset == "custom":
        return
    presets = {
        "side_left": {
            "camera_distance": 0.0,
            "camera_side_offset": -2.1,
            "camera_height": 0.85,
            "look_at_forward": 0.0,
            "look_at_height": 0.0,
            "fov_deg": 50.0,
        },
        "side_right": { # default
            "camera_distance": 0.0,
            "camera_side_offset": 2.1,
            "camera_height": 0.85,
            "look_at_forward": 0.0,
            "look_at_height": 0.0,
            "fov_deg": 50.0,
        },
        "behind": {
            "camera_distance": -2.0,
            "camera_side_offset": 0.0,
            "camera_height": 0.75,
            "look_at_forward": 0.8,
            "look_at_height": 0.35,
            "fov_deg": 55.0,
        },
        "three_quarter": {
            "camera_distance": -1.5,
            "camera_side_offset": 1.2,
            "camera_height": 0.9,
            "look_at_forward": 0.4,
            "look_at_height": 0.4,
            "fov_deg": 50.0,
        },
        "front": {
            "camera_distance": 2.0,
            "camera_side_offset": 0.0,
            "camera_height": 0.65,
            "look_at_forward": 0.0,
            "look_at_height": 0.45,
            "fov_deg": 55.0,
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
    fps: float,
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
    recorder: QuadrupedTrajectoryVideoRecorder,
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

    static_camera_pose: tuple[np.ndarray, np.ndarray] | None = None
    if args.static_camera and not args.manual_camera:
        static_camera_pose = recorder.camera_pose(
            indices[0],
            args.camera_distance,
            args.camera_side_offset,
            args.camera_height,
            args.look_at_forward,
            args.look_at_height,
        )

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
                transport_format=args.transport_format,
            )
        else:
            if static_camera_pose is None:
                camera_position, look_at_position = recorder.camera_pose(
                    traj_frame_idx,
                    args.camera_distance,
                    args.camera_side_offset,
                    args.camera_height,
                    args.look_at_forward,
                    args.look_at_height,
                )
            else:
                camera_position, look_at_position = static_camera_pose
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
                transport_format=args.transport_format,
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
    parser = argparse.ArgumentParser(
        description="Record a viser MP4 of a saved MuJoCo quadruped trajectory.",
    )
    parser.add_argument("--trajectory", type=Path, required=True, help="Path to trajectory .npz file")
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument(
        "--model-xml",
        "--urdf",
        dest="model_xml",
        type=Path,
        default=DEFAULT_SOURCE_MODEL_PATH,
        help=f"Path to the Go2 MuJoCo XML visual model (default: {DEFAULT_SOURCE_MODEL_PATH})",
    )
    parser.add_argument("--port", type=int, default=8081, help="Viser server port (default: 8081)")
    parser.add_argument("--dt", type=float, default=0.02, help="Fallback trajectory timestep in seconds (default: 0.02)")

    parser.add_argument("--start-frame", type=int, default=0, help="First trajectory frame to render (default: 0)")
    parser.add_argument("--end-frame", type=int, help="Exclusive end frame; defaults to trajectory length")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth trajectory frame (default: 1)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video frame rate (default: 30)")
    parser.add_argument("--render-width", type=int, default=1920, help="Render width in pixels (default: 1920)")
    parser.add_argument("--render-height", type=int, default=1080, help="Render height in pixels (default: 1080)")
    parser.add_argument("--render-pause", type=float, default=0.02, help="Seconds to wait after scene updates before capture (default: 0.02)")
    parser.add_argument(
        "--transport-format",
        choices=["png", "jpeg"],
        default="png",
        help="Browser-to-recorder image format; JPEG is faster but lossy (default: png)",
    )

    parser.add_argument("--no-contacts", action="store_true", help="Hide foot contact indicators")
    parser.add_argument("--no-trails", action="store_true", help="Hide trajectory trails")
    parser.add_argument(
        "--trail-bodies",
        type=parse_trail_bodies,
        default=parse_trail_bodies(DEFAULT_TRAIL_BODIES),
        help=(
            "Comma-separated trails: com, feet, base, saved body names, or all "
            f"(default: {DEFAULT_TRAIL_BODIES})"
        ),
    )
    parser.add_argument("--trail-window", type=int, default=0, help="Trail length in frames; 0 means full history (default: 0)")
    parser.add_argument("--trail-line-width", type=float, default=3.0, help="Trail line width (default: 3.0)")
    parser.add_argument("--contact-radius", type=float, default=0.03, help="Foot contact sphere radius (default: 0.03)")

    parser.add_argument(
        "--camera-preset",
        choices=["side_left", "side_right", "behind", "three_quarter", "front", "custom"],
        default="side_right",
        help="Scripted camera preset (default: side_right)",
    )
    parser.add_argument("--manual-camera", action="store_true", help="Use the browser camera as-is instead of scripted follow camera")
    parser.add_argument(
        "--static-camera",
        action="store_true",
        help="Lock the scripted camera at the first frame instead of following the robot",
    )
    parser.add_argument("--camera-distance", type=float, default=0.0, help="Camera X offset from robot root for custom camera")
    parser.add_argument("--camera-side-offset", type=float, default=2.1, help="Camera Y offset from robot root for custom camera")
    parser.add_argument("--camera-height", type=float, default=0.85, help="Camera Z offset from robot root for custom camera")
    parser.add_argument("--look-at-forward", type=float, default=0.0, help="Look-at X offset from robot root")
    parser.add_argument("--look-at-height", type=float, default=0.0, help="Look-at Z offset from robot root")
    parser.add_argument("--fov-deg", type=float, default=50.0, help="Vertical FOV for scripted camera (default: 50)")

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

    recorder = QuadrupedTrajectoryVideoRecorder(
        source_model_path=args.model_xml,
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
    sim_duration = (indices[-1] - indices[0] + 1) * recorder.dt

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
        cleanup_dir = tempfile.TemporaryDirectory(prefix="viser_quadruped_frames_")
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
