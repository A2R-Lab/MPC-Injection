"""Visualize recorded quadruped trajectories with Viser.

This viewer renders the Go2 directly from the MuJoCo XML visual meshes so the
shape and colors match the training model more closely than a collision-URDF
fallback. It also mirrors the walker viewer's playback and ghost-frame tooling.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import viser
from viser import transforms as tf


DEFAULT_SOURCE_MODEL_PATH = (
    Path(__file__).parent.parent
    / "deps"
    / "gym-quadruped"
    / "gym_quadruped"
    / "robot_model"
    / "go2"
    / "go2.xml"
)

COM_COLOR = np.array([0, 220, 255], dtype=np.uint8)
FOOT_COLORS = {
    "FL": np.array([255, 90, 90], dtype=np.uint8),
    "FR": np.array([255, 175, 60], dtype=np.uint8),
    "RL": np.array([60, 215, 165], dtype=np.uint8),
    "RR": np.array([95, 150, 255], dtype=np.uint8),
}
GHOST_COLOR = np.array([255, 145, 40], dtype=np.uint8)
GHOST_COLOR_2 = np.array([95, 150, 255], dtype=np.uint8)

DEFAULT_CAMERA_FOLLOW_ENABLED = True
DEFAULT_CAMERA_DISTANCE = -0.4
DEFAULT_CAMERA_HEIGHT = 0.4
DEFAULT_CAMERA_SIDE_OFFSET = 1.6
DEFAULT_CAMERA_LOOK_AT_HEIGHT = 0.45
DEFAULT_CAMERA_LOOK_AT_FORWARD = -0.5
DEFAULT_CAMERA_PRESET = "Side View (Right to Left)"


@dataclass
class XmlVisualGeom:
    body_name: str
    mesh_name: str
    rgba: tuple[float, float, float, float]


@dataclass
class VisualGeomSpec:
    geom_id: int
    body_name: str
    mesh_name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[int, int, int]
    opacity: float | None


@dataclass
class RobotMeshInstance:
    root: Any
    geom_frames: dict[int, Any]
    mesh_handles: dict[int, Any]
    contact_handles: dict[str, Any]


def _parse_rgba(text: str | None) -> tuple[float, float, float, float] | None:
    if text is None:
        return None
    values = [float(v) for v in text.split()]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4:
        raise ValueError(f"Expected 3 or 4 rgba values, got: {text}")
    return tuple(values)  # type: ignore[return-value]


def _rgba_to_color_and_opacity(
    rgba: tuple[float, float, float, float],
) -> tuple[tuple[int, int, int], float | None]:
    rgb = tuple(int(np.clip(round(channel * 255.0), 0, 255)) for channel in rgba[:3])
    alpha = float(np.clip(rgba[3], 0.0, 1.0))
    return rgb, None if alpha >= 0.999 else alpha


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat_wxyz.astype(np.float64))
    return mat.reshape(3, 3)


def _xmat_to_wxyz(xmat: np.ndarray) -> np.ndarray:
    return tf.SO3.from_matrix(np.asarray(xmat, dtype=np.float64).reshape(3, 3)).wxyz


def _extract_mesh_geometry(
    model: mujoco.MjModel,
    mesh_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = np.array(
        model.mesh_vert[vert_start : vert_start + vert_count],
        dtype=np.float32,
        copy=True,
    )
    faces = np.array(
        model.mesh_face[face_start : face_start + face_count],
        dtype=np.uint32,
        copy=True,
    )

    mesh_scale = np.array(model.mesh_scale[mesh_id], dtype=np.float32, copy=True)
    vertices *= mesh_scale[None, :]

    if faces.size > 0 and int(faces.max()) >= len(vertices):
        raise ValueError(f"Invalid mesh face indices for mesh id {mesh_id}")

    return vertices, faces


def _collect_xml_visual_mesh_geoms(xml_path: Path) -> list[XmlVisualGeom]:
    root = ET.parse(xml_path).getroot()

    material_rgba: dict[str, tuple[float, float, float, float]] = {}
    asset = root.find("asset")
    if asset is not None:
        for material in asset.findall("material"):
            name = material.get("name")
            rgba = _parse_rgba(material.get("rgba"))
            if name is not None and rgba is not None:
                material_rgba[name] = rgba

    visual_geoms: list[XmlVisualGeom] = []

    def visit_body(body_elem: ET.Element) -> None:
        body_name = body_elem.get("name", "unnamed_body")
        for child in body_elem:
            if child.tag == "geom":
                mesh_name = child.get("mesh")
                if mesh_name is None:
                    continue
                if child.get("class") != "visual" and child.get("group") != "2":
                    continue

                rgba = _parse_rgba(child.get("rgba"))
                if rgba is None:
                    material_name = child.get("material")
                    rgba = material_rgba.get(material_name or "", (0.5, 0.5, 0.5, 1.0))

                visual_geoms.append(
                    XmlVisualGeom(
                        body_name=body_name,
                        mesh_name=mesh_name,
                        rgba=rgba,
                    )
                )
            elif child.tag == "body":
                visit_body(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> found in {xml_path}")

    for child in worldbody:
        if child.tag == "body":
            visit_body(child)

    return visual_geoms


def _build_visual_geom_specs(model_xml_path: Path) -> tuple[mujoco.MjModel, list[VisualGeomSpec]]:
    model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    xml_visual_geoms = _collect_xml_visual_mesh_geoms(model_xml_path)

    compiled_visual_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        and int(model.geom_group[geom_id]) == 2
    ]

    if len(xml_visual_geoms) != len(compiled_visual_geom_ids):
        raise ValueError(
            "Mismatch between XML visual mesh geoms "
            f"({len(xml_visual_geoms)}) and compiled MuJoCo visual mesh geoms "
            f"({len(compiled_visual_geom_ids)})"
        )

    specs: list[VisualGeomSpec] = []
    for xml_geom, geom_id in zip(xml_visual_geoms, compiled_visual_geom_ids):
        mesh_id = int(model.geom_dataid[geom_id])
        vertices, faces = _extract_mesh_geometry(model, mesh_id)
        color, opacity = _rgba_to_color_and_opacity(xml_geom.rgba)
        specs.append(
            VisualGeomSpec(
                geom_id=geom_id,
                body_name=xml_geom.body_name,
                mesh_name=xml_geom.mesh_name,
                vertices=vertices,
                faces=faces,
                color=color,
                opacity=opacity,
            )
        )

    return model, specs


class QuadrupedTrajVisualizer:
    """Viser visualizer for saved quadruped trajectories."""

    def __init__(
        self,
        *,
        source_model_path: Path,
        port: int = 8081,
        dt: float = 0.02,
        show_contacts: bool = True,
        show_com_trail: bool = True,
        show_foot_trails: bool = True,
        num_ghost_frames: int = 8,
        dual_mode: bool = False,
        robot2_x_offset: float = 0.0,
        robot2_y_offset: float = -3.0,
    ):
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
        self.max_ghost_frames = num_ghost_frames
        self.dual_mode = dual_mode
        self.robot2_x_offset = robot2_x_offset
        self.robot2_y_offset = robot2_y_offset
        self.source_model_path = source_model_path.resolve()
        self.render_model, self.visual_geom_specs = _build_visual_geom_specs(self.source_model_path)
        self.render_data = mujoco.MjData(self.render_model)
        self.ghost_render_data = [mujoco.MjData(self.render_model) for _ in range(self.max_ghost_frames)]
        self.render_data_2 = mujoco.MjData(self.render_model) if self.dual_mode else None
        self.ghost_render_data_2 = [
            mujoco.MjData(self.render_model) for _ in range(self.max_ghost_frames)
        ] if self.dual_mode else []

        self.robot_instance = self._create_robot_mesh_instance("/robot")
        self.robot_instance_2 = self._create_robot_mesh_instance(
            "/robot2",
            root_position=(self.robot2_x_offset, self.robot2_y_offset, 0.0),
        ) if self.dual_mode else None
        self._ghost_instances: list[RobotMeshInstance] = [
            self._create_robot_mesh_instance(
                f"/ghost_{ghost_idx}",
                color_override=tuple(int(x) for x in GHOST_COLOR),
                opacity_override=0.45,
                cast_shadow=False,
                receive_shadow=False,
            )
            for ghost_idx in range(self.max_ghost_frames)
        ]
        self._ghost_instances_2: list[RobotMeshInstance] = [
            self._create_robot_mesh_instance(
                f"/ghost2_{ghost_idx}",
                root_position=(self.robot2_x_offset, self.robot2_y_offset, 0.0),
                color_override=tuple(int(x) for x in GHOST_COLOR_2),
                opacity_override=0.45,
                cast_shadow=False,
                receive_shadow=False,
            )
            for ghost_idx in range(self.max_ghost_frames)
        ] if self.dual_mode else []
        self._set_ghost_visibility(False)
        if self.dual_mode:
            self._set_ghost_visibility(False, robot_idx=2)

        self.server.scene.add_grid(
            "/grid",
            width=200,
            height=200,
            position=(0.0, 0.0, 0.0),
            plane="xy",
            shadow_opacity=0.0,
        )

        self.trajectory_data = None
        self.trajectory_data_2 = None
        self.current_frame = 0.0
        self._camera_follow_enabled = DEFAULT_CAMERA_FOLLOW_ENABLED
        self._trail_handles: dict[str, Any] = {}
        self._last_time = time.perf_counter()

        self._setup_gui(
            show_contacts=show_contacts,
            show_com_trail=show_com_trail,
            show_foot_trails=show_foot_trails,
        )

    def _create_robot_mesh_instance(
        self,
        root_name: str,
        *,
        root_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color_override: tuple[int, int, int] | None = None,
        opacity_override: float | None = None,
        cast_shadow: bool = True,
        receive_shadow: bool | float = True,
    ) -> RobotMeshInstance:
        root = self.server.scene.add_frame(root_name, show_axes=False)
        root.position = root_position
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
                color=color_override if color_override is not None else spec.color,
                opacity=opacity_override if color_override is not None else spec.opacity,
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

    def _setup_gui(
        self,
        *,
        show_contacts: bool,
        show_com_trail: bool,
        show_foot_trails: bool,
    ):
        with self.server.gui.add_folder("Playback Controls"):
            self.play_pause = self.server.gui.add_checkbox("Play", initial_value=False)
            self.speed_slider = self.server.gui.add_slider(
                "Speed",
                min=0.05,
                max=2.0,
                step=0.05,
                initial_value=1.0,
            )
            self.frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=1000,
                step=1,
                initial_value=0,
            )
            self.loop_checkbox = self.server.gui.add_checkbox("Loop", initial_value=True)
            self.reset_button = self.server.gui.add_button("Reset")

            @self.reset_button.on_click
            def _(_):
                self.current_frame = 0.0
                self.frame_slider.value = 0
                self._reset_playback_clock()
                self.update_visualization(0)

            @self.play_pause.on_update
            def _(_):
                self._reset_playback_clock()
                if self.trajectory_data is not None and self.play_pause.value:
                    if int(self.current_frame) >= self._num_frames() - 1 and self.loop_checkbox.value:
                        self.current_frame = 0.0
                        self.frame_slider.value = 0
                        self.update_visualization(0)

            @self.frame_slider.on_update
            def _(_):
                if self.trajectory_data is not None and not self.play_pause.value:
                    self.current_frame = float(self.frame_slider.value)
                    self.update_visualization(int(self.current_frame))

        with self.server.gui.add_folder("Visualization Options"):
            self.show_contacts_checkbox = self.server.gui.add_checkbox(
                "Show Foot Contacts",
                initial_value=show_contacts,
            )
            self.show_com_trail_checkbox = self.server.gui.add_checkbox(
                "Show CoM Trail",
                initial_value=show_com_trail,
            )
            self.show_foot_trails_checkbox = self.server.gui.add_checkbox(
                "Show Foot Trails",
                initial_value=show_foot_trails,
            )
            self.show_stats_checkbox = self.server.gui.add_checkbox(
                "Show Stats",
                initial_value=True,
            )
            self.trail_length_slider = self.server.gui.add_slider(
                "Trail Length",
                min=10,
                max=1000,
                step=10,
                initial_value=250,
            )
            self.line_width_slider = self.server.gui.add_slider(
                "Trail Width",
                min=1.0,
                max=6.0,
                step=0.5,
                initial_value=2.0,
            )

        with self.server.gui.add_folder("Ghost Frames"):
            self.show_ghosts_checkbox = self.server.gui.add_checkbox(
                "Show Ghost Frames",
                initial_value=False,
            )
            self.num_ghosts_slider = self.server.gui.add_slider(
                "Number of Ghosts",
                min=1,
                max=self.max_ghost_frames,
                step=1,
                initial_value=min(4, self.max_ghost_frames),
            )
            self.ghost_interval_slider = self.server.gui.add_slider(
                "Frame Interval",
                min=1,
                max=100,
                step=1,
                initial_value=20,
            )
            self.ghost_direction = self.server.gui.add_dropdown(
                "Ghost Direction",
                options=["Past", "Future", "Both"],
                initial_value="Past",
            )

        with self.server.gui.add_folder("Statistics", expand_by_default=False):
            self.stats_text = self.server.gui.add_text(
                "Info",
                initial_value="No trajectory loaded",
                disabled=True,
            )

        with self.server.gui.add_folder("Camera Follow", expand_by_default=True):
            self.follow_camera_checkbox = self.server.gui.add_checkbox(
                "Follow Robot",
                initial_value=DEFAULT_CAMERA_FOLLOW_ENABLED,
            )
            self.camera_distance_slider = self.server.gui.add_slider(
                "Distance Behind",
                min=-10.0,
                max=10.0,
                step=0.1,
                initial_value=DEFAULT_CAMERA_DISTANCE,
            )
            self.camera_height_slider = self.server.gui.add_slider(
                "Height Above",
                min=-2.0,
                max=10.0,
                step=0.1,
                initial_value=DEFAULT_CAMERA_HEIGHT,
            )
            self.camera_side_offset_slider = self.server.gui.add_slider(
                "Side Offset",
                min=-10.0,
                max=10.0,
                step=0.1,
                initial_value=DEFAULT_CAMERA_SIDE_OFFSET,
            )
            self.look_at_height_slider = self.server.gui.add_slider(
                "Look-at Height",
                min=-2.0,
                max=5.0,
                step=0.05,
                initial_value=DEFAULT_CAMERA_LOOK_AT_HEIGHT,
            )
            self.look_at_forward_slider = self.server.gui.add_slider(
                "Look-at Forward",
                min=-5.0,
                max=10.0,
                step=0.1,
                initial_value=DEFAULT_CAMERA_LOOK_AT_FORWARD,
            )
            self.camera_preset_dropdown = self.server.gui.add_dropdown(
                "Camera Preset",
                options=[
                    "Custom",
                    "Side View (Left to Right)",
                    "Side View (Right to Left)",
                    "Behind View",
                    "Top-Down",
                    "3/4 View",
                    "Front View",
                ],
                initial_value=DEFAULT_CAMERA_PRESET,
            )

            @self.follow_camera_checkbox.on_update
            def _(_):
                self._camera_follow_enabled = self.follow_camera_checkbox.value
                self.update_visualization(int(self.current_frame))

            @self.camera_preset_dropdown.on_update
            def _(_):
                preset = self.camera_preset_dropdown.value
                if preset == "Side View (Right to Left)":
                    self.camera_distance_slider.value = DEFAULT_CAMERA_DISTANCE
                    self.camera_height_slider.value = DEFAULT_CAMERA_HEIGHT
                    self.camera_side_offset_slider.value = DEFAULT_CAMERA_SIDE_OFFSET
                    self.look_at_height_slider.value = DEFAULT_CAMERA_LOOK_AT_HEIGHT
                    self.look_at_forward_slider.value = DEFAULT_CAMERA_LOOK_AT_FORWARD
                elif preset == "Side View (Left to Right)":
                    self.camera_distance_slider.value = 0.0
                    self.camera_height_slider.value = 1.0
                    self.camera_side_offset_slider.value = -5.0
                    self.look_at_height_slider.value = 0.8
                    self.look_at_forward_slider.value = 0.0
                elif preset == "Behind View":
                    self.camera_distance_slider.value = -5.0
                    self.camera_height_slider.value = 2.0
                    self.camera_side_offset_slider.value = 0.0
                    self.look_at_height_slider.value = 0.4
                    self.look_at_forward_slider.value = 1.5
                elif preset == "Top-Down":
                    self.camera_distance_slider.value = 0.0
                    self.camera_height_slider.value = 8.0
                    self.camera_side_offset_slider.value = 0.0
                    self.look_at_height_slider.value = 0.0
                    self.look_at_forward_slider.value = 0.0
                elif preset == "3/4 View":
                    self.camera_distance_slider.value = -4.0
                    self.camera_height_slider.value = 3.0
                    self.camera_side_offset_slider.value = -3.0
                    self.look_at_height_slider.value = 0.4
                    self.look_at_forward_slider.value = 1.0
                elif preset == "Front View":
                    self.camera_distance_slider.value = 5.0
                    self.camera_height_slider.value = 1.5
                    self.camera_side_offset_slider.value = 0.0
                    self.look_at_height_slider.value = 0.8
                    self.look_at_forward_slider.value = 0.0
                self.update_visualization(int(self.current_frame))

        def bind_refresh(handle):
            @handle.on_update
            def _(_):
                if self.trajectory_data is not None:
                    self.update_visualization(int(self.current_frame))

        for handle in (
            self.show_contacts_checkbox,
            self.show_com_trail_checkbox,
            self.show_foot_trails_checkbox,
            self.show_stats_checkbox,
            self.trail_length_slider,
            self.line_width_slider,
            self.show_ghosts_checkbox,
            self.num_ghosts_slider,
            self.ghost_interval_slider,
            self.ghost_direction,
            self.camera_distance_slider,
            self.camera_height_slider,
            self.camera_side_offset_slider,
            self.look_at_height_slider,
            self.look_at_forward_slider,
        ):
            bind_refresh(handle)

    def _reset_playback_clock(self) -> None:
        self._last_time = time.perf_counter()

    def _trajectory_data_for_robot(self, robot_idx: int):
        return self.trajectory_data if robot_idx == 1 else self.trajectory_data_2

    def _robot_instance_for_robot(self, robot_idx: int) -> RobotMeshInstance | None:
        return self.robot_instance if robot_idx == 1 else self.robot_instance_2

    def _ghost_instances_for_robot(self, robot_idx: int) -> list[RobotMeshInstance]:
        return self._ghost_instances if robot_idx == 1 else self._ghost_instances_2

    def _ghost_render_data_for_robot(self, robot_idx: int) -> list[mujoco.MjData]:
        return self.ghost_render_data if robot_idx == 1 else self.ghost_render_data_2

    def _render_data_for_robot(self, robot_idx: int) -> mujoco.MjData | None:
        return self.render_data if robot_idx == 1 else self.render_data_2

    def _robot_root_name(self, robot_idx: int) -> str:
        return "/robot" if robot_idx == 1 else "/robot2"

    def _ghost_root_name(self, robot_idx: int, ghost_idx: int) -> str:
        return f"/ghost_{ghost_idx}" if robot_idx == 1 else f"/ghost2_{ghost_idx}"

    def load_trajectory(self, npz_path: str, robot_idx: int = 1):
        data = np.load(npz_path, allow_pickle=True)
        if robot_idx == 1:
            self.trajectory_data = data
            if "frame_dt" in data.files:
                self.dt = float(data["frame_dt"])
            elif "control_dt" in data.files:
                self.dt = float(data["control_dt"])

            num_frames = self._num_frames(robot_idx=1)
            self.frame_slider.max = max(0, num_frames - 1)
            self.trail_length_slider.max = max(10, num_frames)
            self.current_frame = 0.0
            self.frame_slider.value = 0
            self._reset_playback_clock()
            self._update_statistics()
            self.update_visualization(0)
        else:
            self.trajectory_data_2 = data
            if self.trajectory_data is not None:
                self.update_visualization(int(self.current_frame))
        return data

    def _update_statistics(self):
        if self.trajectory_data is None:
            return

        data = self.trajectory_data
        stats_lines = [
            f"Frames: {self._num_frames()}",
            f"Duration: {(self._num_frames() - 1) * self.dt:.2f} s",
        ]
        if "rewards" in data.files and len(data["rewards"]) > 0:
            stats_lines.append(f"Total Reward: {np.sum(data['rewards']):.2f}")
            stats_lines.append(f"Average Reward: {np.mean(data['rewards']):.2f}")
        if "done" in data.files:
            stats_lines.append(f"Terminated Early: {bool(data['done'])}")
        self.stats_text.value = "\n".join(stats_lines) if self.show_stats_checkbox.value else ""

    def _num_frames(self, robot_idx: int = 1) -> int:
        data = self._trajectory_data_for_robot(robot_idx)
        if data is None:
            return 0
        if "qpos_ctrl" in data.files:
            return int(data["qpos_ctrl"].shape[0])
        if "timesteps" in data.files:
            return int(data["timesteps"])
        return int(self._qpos_ctrl(robot_idx=robot_idx).shape[0])

    def _qpos_ctrl(self, robot_idx: int = 1) -> np.ndarray:
        data = self._trajectory_data_for_robot(robot_idx)
        if data is None:
            return np.zeros((0, 19))
        if "qpos_ctrl" in data.files:
            return data["qpos_ctrl"]
        qpos = data["qpos"]
        decimation = int(data["decimation"]) if "decimation" in data.files else 4
        return qpos[:, ::decimation].T

    def _foot_names(self, robot_idx: int = 1) -> list[str]:
        data = self._trajectory_data_for_robot(robot_idx)
        if data is None:
            return []
        if "foot_names" in data.files:
            return [str(name) for name in data["foot_names"]]
        return ["FL", "FR", "RL", "RR"]

    def _get_root_state(self, frame_idx: int, robot_idx: int = 1) -> tuple[np.ndarray, np.ndarray]:
        qpos = self._qpos_ctrl(robot_idx=robot_idx)[frame_idx]
        return qpos[:3], qpos[3:7]

    def _set_ghost_visibility(
        self,
        visible: bool,
        num_visible: int | None = None,
        robot_idx: int = 1,
    ):
        for ghost_idx, ghost_instance in enumerate(self._ghost_instances_for_robot(robot_idx)):
            ghost_instance.root.visible = visible if num_visible is None else visible and ghost_idx < num_visible

    def _apply_qpos_to_mesh_instance(
        self,
        instance: RobotMeshInstance,
        qpos: np.ndarray,
        render_data: mujoco.MjData,
        *,
        opacity_override: float | None = None,
    ) -> None:
        render_data.qpos[:] = qpos
        mujoco.mj_forward(self.render_model, render_data)

        for spec in self.visual_geom_specs:
            frame = instance.geom_frames[spec.geom_id]
            frame.position = np.array(render_data.geom_xpos[spec.geom_id], copy=True)
            frame.wxyz = _xmat_to_wxyz(render_data.geom_xmat[spec.geom_id])
            if opacity_override is not None:
                instance.mesh_handles[spec.geom_id].opacity = opacity_override

    def _ensure_contact_handle(
        self,
        instance: RobotMeshInstance,
        root_name: str,
        foot_name: str,
        radius: float,
    ):
        handle_name = f"{root_name}/contact_{foot_name}"
        if handle_name not in instance.contact_handles:
            instance.contact_handles[handle_name] = self.server.scene.add_icosphere(
                handle_name,
                radius=radius,
                color=tuple(int(x) for x in FOOT_COLORS[foot_name]),
                position=(0.0, 0.0, 0.0),
            )
        return instance.contact_handles[handle_name]

    def _hide_instance_contacts(self, instance: RobotMeshInstance):
        for handle in instance.contact_handles.values():
            handle.visible = False

    def _update_contact_visualization(self, frame_idx: int, robot_idx: int = 1):
        data = self._trajectory_data_for_robot(robot_idx)
        instance = self._robot_instance_for_robot(robot_idx)
        if (
            data is None
            or instance is None
            or not self.show_contacts_checkbox.value
            or "foot_positions" not in data.files
            or "foot_contacts" not in data.files
        ):
            if instance is not None:
                self._hide_instance_contacts(instance)
            return

        foot_positions = data["foot_positions"][frame_idx]
        foot_contacts = data["foot_contacts"][frame_idx]
        root_name = self._robot_root_name(robot_idx)
        for foot_idx, foot_name in enumerate(self._foot_names(robot_idx=robot_idx)):
            handle = self._ensure_contact_handle(
                instance,
                root_name,
                foot_name,
                radius=0.03,
            )
            handle.position = tuple(foot_positions[foot_idx])
            handle.visible = bool(foot_contacts[foot_idx])

    def _update_ghost_contacts(self, ghost_idx: int, frame_idx: int, robot_idx: int = 1):
        ghost_instances = self._ghost_instances_for_robot(robot_idx)
        if ghost_idx >= len(ghost_instances):
            return

        ghost_instance = ghost_instances[ghost_idx]
        data = self._trajectory_data_for_robot(robot_idx)
        if (
            data is None
            or not self.show_contacts_checkbox.value
            or "foot_positions" not in data.files
            or "foot_contacts" not in data.files
        ):
            self._hide_instance_contacts(ghost_instance)
            return

        foot_positions = data["foot_positions"][frame_idx]
        foot_contacts = data["foot_contacts"][frame_idx]
        ghost_root_name = self._ghost_root_name(robot_idx, ghost_idx)
        for foot_idx, foot_name in enumerate(self._foot_names(robot_idx=robot_idx)):
            handle = self._ensure_contact_handle(
                ghost_instance,
                ghost_root_name,
                foot_name,
                radius=0.02,
            )
            handle.position = tuple(foot_positions[foot_idx])
            handle.visible = bool(foot_contacts[foot_idx])

    def _update_ghost_frame(
        self,
        ghost_idx: int,
        frame_idx: int,
        opacity_factor: float = 1.0,
        robot_idx: int = 1,
    ):
        ghost_instances = self._ghost_instances_for_robot(robot_idx)
        ghost_render_data = self._ghost_render_data_for_robot(robot_idx)
        if ghost_idx >= len(ghost_instances) or ghost_idx >= len(ghost_render_data):
            return

        ghost_instances[ghost_idx].root.visible = True
        self._apply_qpos_to_mesh_instance(
            ghost_instances[ghost_idx],
            self._qpos_ctrl(robot_idx=robot_idx)[frame_idx],
            ghost_render_data[ghost_idx],
            opacity_override=float(opacity_factor),
        )
        self._update_ghost_contacts(ghost_idx, frame_idx, robot_idx=robot_idx)

    def _update_ghost_visualization(self, current_frame: int, robot_idx: int = 1):
        ghost_instances = self._ghost_instances_for_robot(robot_idx)
        data = self._trajectory_data_for_robot(robot_idx)
        if data is None or not self.show_ghosts_checkbox.value:
            self._set_ghost_visibility(False, robot_idx=robot_idx)
            for ghost_instance in ghost_instances:
                self._hide_instance_contacts(ghost_instance)
            return

        num_ghosts = int(self.num_ghosts_slider.value)
        interval = int(self.ghost_interval_slider.value)
        direction = self.ghost_direction.value
        num_frames = self._num_frames(robot_idx=robot_idx)

        ghost_frame_indices: list[int] = []
        if direction == "Past":
            for offset_idx in range(1, num_ghosts + 1):
                candidate = current_frame - offset_idx * interval
                if candidate >= 0:
                    ghost_frame_indices.append(candidate)
        elif direction == "Future":
            for offset_idx in range(1, num_ghosts + 1):
                candidate = current_frame + offset_idx * interval
                if candidate < num_frames:
                    ghost_frame_indices.append(candidate)
        else:
            half = num_ghosts // 2
            for offset_idx in range(1, half + 1):
                candidate = current_frame - offset_idx * interval
                if candidate >= 0:
                    ghost_frame_indices.append(candidate)
            for offset_idx in range(1, num_ghosts - half + 1):
                candidate = current_frame + offset_idx * interval
                if candidate < num_frames:
                    ghost_frame_indices.append(candidate)

        for ghost_idx, frame_idx in enumerate(ghost_frame_indices):
            if ghost_idx >= self.max_ghost_frames:
                break
            distance = abs(frame_idx - current_frame)
            max_distance = max(1, num_ghosts * interval)
            opacity_factor = 0.9 - (distance / max_distance) * 0.6
            self._update_ghost_frame(
                ghost_idx,
                frame_idx,
                opacity_factor,
                robot_idx=robot_idx,
            )

        for ghost_idx in range(len(ghost_frame_indices), self.max_ghost_frames):
            if ghost_idx >= len(ghost_instances):
                break
            ghost_instances[ghost_idx].root.visible = False
            self._hide_instance_contacts(ghost_instances[ghost_idx])

    def _fade_colors(self, base_color: np.ndarray, num_segments: int) -> np.ndarray:
        colors = np.zeros((num_segments, 2, 3), dtype=np.uint8)
        for segment_idx in range(num_segments):
            alpha = (segment_idx + 1) / max(1, num_segments)
            color_val = int(45 + alpha * 210)
            colors[segment_idx] = np.array(
                [[int(channel / 255.0 * color_val) for channel in base_color]] * 2,
                dtype=np.uint8,
            )
        return colors

    def _update_line_handle(self, line_name: str, positions: np.ndarray, base_color: np.ndarray):
        handle = self._trail_handles.get(line_name)
        if len(positions) < 2:
            if handle is not None:
                handle.visible = False
            return

        points = np.stack([positions[:-1], positions[1:]], axis=1)
        colors = self._fade_colors(base_color, len(points))
        if handle is None:
            self._trail_handles[line_name] = self.server.scene.add_line_segments(
                line_name,
                points=points,
                colors=colors,
                line_width=float(self.line_width_slider.value),
            )
        else:
            handle.points = points
            handle.colors = colors
            handle.line_width = float(self.line_width_slider.value)
            handle.visible = True

    def _update_trails(self, frame_idx: int, robot_idx: int = 1):
        data = self._trajectory_data_for_robot(robot_idx)
        if data is None:
            return

        trail_length = int(self.trail_length_slider.value)
        start_idx = max(0, frame_idx - trail_length + 1)
        window = slice(start_idx, frame_idx + 1)
        root_name = self._robot_root_name(robot_idx)

        if self.show_com_trail_checkbox.value and "com_positions" in data.files:
            self._update_line_handle(
                f"{root_name}/com_trail",
                data["com_positions"][window],
                COM_COLOR,
            )
        elif f"{root_name}/com_trail" in self._trail_handles:
            self._trail_handles[f"{root_name}/com_trail"].visible = False

        if "foot_positions" not in data.files:
            return

        for foot_idx, foot_name in enumerate(self._foot_names(robot_idx=robot_idx)):
            line_name = f"{root_name}/trail_{foot_name}"
            if self.show_foot_trails_checkbox.value:
                self._update_line_handle(
                    line_name,
                    data["foot_positions"][window, foot_idx, :],
                    FOOT_COLORS[foot_name],
                )
            elif line_name in self._trail_handles:
                self._trail_handles[line_name].visible = False

    def _update_follow_camera(self, frame_idx: int):
        if not self._camera_follow_enabled or self.trajectory_data is None:
            return

        root_pos, _ = self._get_root_state(frame_idx, robot_idx=1)
        camera_offset = np.array(
            [
                self.camera_distance_slider.value,
                self.camera_side_offset_slider.value,
                self.camera_height_slider.value,
            ]
        )
        look_at_offset = np.array(
            [
                self.look_at_forward_slider.value,
                0.0,
                self.look_at_height_slider.value,
            ]
        )
        camera_position = root_pos + camera_offset
        look_at_position = root_pos + look_at_offset

        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = camera_position
                client.camera.look_at = look_at_position

    def _update_single_robot_visualization(self, frame_idx: int, robot_idx: int = 1):
        data = self._trajectory_data_for_robot(robot_idx)
        instance = self._robot_instance_for_robot(robot_idx)
        render_data = self._render_data_for_robot(robot_idx)
        if data is None or instance is None or render_data is None:
            return

        frame_idx = int(np.clip(frame_idx, 0, self._num_frames(robot_idx=robot_idx) - 1))
        self._apply_qpos_to_mesh_instance(
            instance,
            self._qpos_ctrl(robot_idx=robot_idx)[frame_idx],
            render_data,
        )
        self._update_contact_visualization(frame_idx, robot_idx=robot_idx)
        self._update_trails(frame_idx, robot_idx=robot_idx)
        self._update_ghost_visualization(frame_idx, robot_idx=robot_idx)

    def update_visualization(self, frame_idx: int):
        if self.trajectory_data is None:
            return

        frame_idx = int(np.clip(frame_idx, 0, self._num_frames(robot_idx=1) - 1))
        with self.server.atomic():
            self._update_single_robot_visualization(frame_idx, robot_idx=1)
            if self.dual_mode and self.trajectory_data_2 is not None:
                frame_idx_2 = int(
                    np.clip(frame_idx, 0, self._num_frames(robot_idx=2) - 1)
                )
                self._update_single_robot_visualization(frame_idx_2, robot_idx=2)
            self._update_statistics()
        self._update_follow_camera(frame_idx)

    def _tick_playback(self, now: float | None = None) -> None:
        if self.trajectory_data is None:
            self._reset_playback_clock()
            return

        current_time = time.perf_counter() if now is None else now
        dt_actual = max(0.0, current_time - self._last_time)
        self._last_time = current_time

        if self.play_pause.value:
            num_frames = self._num_frames(robot_idx=1)
            if num_frames <= 0:
                return

            frame_increment = dt_actual * float(self.speed_slider.value) / max(self.dt, 1e-8)
            if frame_increment <= 0.0:
                return

            self.current_frame += frame_increment
            if self.current_frame >= num_frames:
                if self.loop_checkbox.value:
                    self.current_frame = float(self.current_frame % num_frames)
                else:
                    self.current_frame = float(num_frames - 1)
                    self.play_pause.value = False

            frame_idx = int(np.clip(np.floor(self.current_frame), 0, num_frames - 1))
            self.frame_slider.value = frame_idx
            self.update_visualization(frame_idx)
        else:
            manual_frame = int(self.frame_slider.value)
            if manual_frame != int(self.current_frame):
                self.current_frame = float(manual_frame)
                self.update_visualization(manual_frame)

    def play(self):
        print(f"\nVisualizer running on http://localhost:{self.port}")
        print("Connect in your browser. Press Ctrl+C to exit.\n")

        while True:
            self._tick_playback()
            time.sleep(0.005 if self.play_pause.value else 0.01)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saved quadruped trajectories with Viser.",
        epilog=(
            "Examples:\n"
            "  Single trajectory: python viser_quadruped_viz_trajs.py --trajectory path/to/traj.npz\n"
            "  Dual trajectories: python viser_quadruped_viz_trajs.py --trajectory1 path/to/traj1.npz --trajectory2 path/to/traj2.npz"
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        help="Path to a quadruped trajectory .npz file (single-robot mode)",
    )
    parser.add_argument(
        "--trajectory1",
        type=str,
        help="Path to the first quadruped trajectory .npz file (dual-robot mode)",
    )
    parser.add_argument(
        "--trajectory2",
        type=str,
        help="Path to the second quadruped trajectory .npz file (dual-robot mode)",
    )
    parser.add_argument(
        "--model-xml",
        "--urdf",
        dest="model_xml",
        type=Path,
        default=DEFAULT_SOURCE_MODEL_PATH,
        help=f"Path to the Go2 MuJoCo XML visual model. Default: {DEFAULT_SOURCE_MODEL_PATH}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port for the Viser server (default: 8081)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Fallback frame dt in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--ghost-frames",
        type=int,
        default=8,
        help="Maximum number of ghost frames to pre-allocate (default: 8)",
    )
    parser.add_argument(
        "--robot2-x-offset",
        type=float,
        default=0.0,
        help="X-axis offset for the second robot in dual mode (default: 0.0)",
    )
    parser.add_argument(
        "--robot2-y-offset",
        type=float,
        default=-3.0,
        help="Y-axis offset for the second robot in dual mode (default: -3.0)",
    )
    parser.add_argument(
        "--no-contacts",
        action="store_true",
        help="Hide foot contact indicators",
    )
    parser.add_argument(
        "--no-com-trail",
        action="store_true",
        help="Hide the CoM trail",
    )
    parser.add_argument(
        "--no-foot-trails",
        action="store_true",
        help="Hide foot trails",
    )
    args = parser.parse_args()

    dual_mode = False
    traj1_path: str | None = None
    traj2_path: str | None = None

    if args.trajectory1 and args.trajectory2:
        dual_mode = True
        traj1_path = args.trajectory1
        traj2_path = args.trajectory2
    elif args.trajectory1 or args.trajectory2:
        parser.error("When using --trajectory1 or --trajectory2, both must be specified")
    elif args.trajectory:
        traj1_path = args.trajectory
    else:
        parser.error(
            "Must specify either --trajectory (single mode) or both --trajectory1 and --trajectory2 (dual mode)"
        )

    viz = QuadrupedTrajVisualizer(
        source_model_path=args.model_xml,
        port=args.port,
        dt=args.dt,
        show_contacts=not args.no_contacts,
        show_com_trail=not args.no_com_trail,
        show_foot_trails=not args.no_foot_trails,
        num_ghost_frames=args.ghost_frames,
        dual_mode=dual_mode,
        robot2_x_offset=args.robot2_x_offset,
        robot2_y_offset=args.robot2_y_offset,
    )
    viz.load_trajectory(traj1_path, robot_idx=1)
    if dual_mode and traj2_path is not None:
        viz.load_trajectory(traj2_path, robot_idx=2)
    viz.play()


if __name__ == "__main__":
    main()
