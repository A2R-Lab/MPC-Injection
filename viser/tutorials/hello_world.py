"""
Collection of the main functions needed for the tutorials of viser.

Mainly written here so I can learn how the package works. Each function in
main is written in the order I learned them from. To run each just uncomment
the one you want to run in the main function.
"""
from __future__ import annotations

import time
import random
import viser

from typing import Literal

import numpy as np
import tyro
from robot_descriptions.loaders.yourdfpy import load_robot_description

from viser.extras import ViserUrdf


def main_hello_world():
    """
    Creates a server and adds a red sphere

    Two essential steps:
    1. Create a viser.ViserServer() instance, which starts a webserver at localhost:8080
    2. Add 3D content using the scene API
    """
    server = viser.ViserServer()
    server.scene.add_icosphere(
        name="/hello_sphere",
        radius=0.5, # NOTE: just about covers the world axes
        color=(255, 0, 0), # RED
        position=(0.0, 0.0, 0.0),
    )

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to stop the server")

    while True:
        time.sleep(10.0)

def main_core_concepts():
    """
    Create a server, add 3D objects to scene, build interactive GUI controls

    - Scene API (viser.ViserServer.scene): add 3D objs like meshes, point clouds, primitive shapes
    - GUI API (viser.ViserServer.gui): create interactive controls like sliders, buttons, and
      color pickers
    """
    server = viser.ViserServer()

    # Add 3D objs to the scene
    sphere = server.scene.add_icosphere(
        name="/sphere",
        radius=0.3,
        color=(255, 100, 100),
        position=(0.0, 0.0, 0.0),
    )
    box = server.scene.add_box(
        name="/box",
        dimensions=(0.4, 0.4, 0.4),
        color=(100, 255, 100),
        position=(1.0, 0.0, 0.0),
    )

    # Create GU controls
    sphere_visible = server.gui.add_checkbox("Show sphere", initial_value=True)
    sphere_color = server.gui.add_rgb("Sphere color", initial_value=(255, 100, 100))
    box_height = server.gui.add_slider(
        "Box height", min=-1.0, max=1.0, step=0.1, initial_value=0.0
    )

    # Connect GU controls to scene objects
    @sphere_visible.on_update
    def _(_):
        sphere.visible = sphere_visible.value
    
    @sphere_color.on_update
    def _(_):
        sphere.color = sphere_color.value

    @box_height.on_update
    def _(_):
        box.position = (1.0, 0.0, box_height.value)

    print("Open your browser to http://localhost:8080 while server running...")
    while True:
        time.sleep(10.0)

def main_coordinate_frames():
    """
    Visualize 3D coordinate systems and hierarhical transformations

    Create and organize coordinate frames using `viser.SceneApi.add_frame()`. Key concepts:
    - Hierarchical naming: scene nodes follow a filesystem-like struct `/tree/branch/leaf`
    - Relative positioning: child frames are positioned relative to their parent frames
    - Dynamic scene management: frames can be added and removed during runtime with 
      `viser.SceneNodeHandle.remove()`
    """
    server = viser.ViserServer()

    while True:
        # Add some coordinate frames to the scene. These will be visualized in the viewer
        server.scene.add_frame(
            "/tree",
            wxyz=(2.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )
        server.scene.add_frame(
            "/tree/branch",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )
        leaf = server.scene.add_frame(
            "/tree/branch/leaf",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )

        # Move the leaf randomly. Assigned properties are automatically updated in the visualizer
        for i in range(10):
            leaf.position = (random.random() * 2.0, 2.0, 0.2)
            time.sleep(0.5)

        # Remove the leaf node from the scene
        leaf.remove()
        time.sleep(0.5)


def create_robot_control_sliders(
        server: viser.ViserServer,
        viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """
    Create GUI sliders for controlling robot joints
    """
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []

    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )

        slider.on_update( # When sliders move, update the URDF configuration
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )

        slider_handles.append(slider)
        initial_config.append(initial_pos)
    
    return slider_handles, initial_config


def main_urdf_robo_viz(
        robot_type: Literal[
        "panda",
        "ur10",
        "cassie",
        "allegro_hand",
        "barrett_hand",
        "robotiq_2f85",
        "atlas_drc",
        "g1",
        "h1",
        "anymal_c",
        "go2",
    ] = "go2",
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> None:
    """
    Visualize robot models from URDF files with interative joint controls

    Features:
    - `viser.extras.ViserUrdf` for URDF file parsing and visualization
    - Interactive joint sliders for robot articulation
    - Real-time robot pose updates
    - Support for local URDF files and robot_descriptions library
    """
    # Start viser server
    server = viser.ViserServer()

    # Load URDF
    # This takes either a yourdfpy.URDF obj or a path to a .urdf file
    urdf = load_robot_description(
        robot_type + "_description",
        load_meshes=load_meshes,
        build_scene_graph=load_meshes,
        load_collision_meshes=load_collision_meshes,
        build_collision_scene_graph=load_collision_meshes,
    )
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5), # RGB or RGBA tuple
    )

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene

    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")


    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        time.sleep(10.0)



if __name__ == "__main__":
    #main_hello_world()
    #main_core_concepts()
    #main_coordinate_frames()
    tyro.cli(main_urdf_robo_viz) # This must be run with tyro.cli(main)