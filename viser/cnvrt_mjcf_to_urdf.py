"""
Quick script to convert a MuJoCo MJCF model to a proper URDF file.

Uses mjcf-urdf-simple-converter for proper MJCF to URDF conversion.
https://github.com/Yasu31/mjcf_urdf_simple_converter

NOTE: The xmls come from ~/MPC-RL/mpc_rl/tasks/walker/walker_modified.xml

This script first cleans the MJCF (removes floor, cameras, lights, material refs)
then converts to URDF format, and finally adds visual geometry since the converter
doesn't include visual elements.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from mjcf_urdf_simple_converter import convert

# Get paths relative to this script's location
script_dir = Path(__file__).parent
workspace_root = script_dir.parent
walker_dir = workspace_root / "mpc_rl" / "tasks" / "walker"

# Define input and output paths
mjcf_path = walker_dir / "walker_modified.xml"
urdf_path = walker_dir / "walker_modified.urdf"
temp_mjcf_path = walker_dir / "walker_temp_for_urdf.xml"

print(f"Loading and cleaning MJCF from: {mjcf_path}")

# Load and parse the XML
tree = ET.parse(mjcf_path)
root = tree.getroot()

# Remove material references from geoms in default section
for default in root.findall('.//default'):
    for geom in default.findall('.//geom'):
        if 'material' in geom.attrib:
            del geom.attrib['material']

# Find worldbody and remove floor, cameras, and lights that are direct children
worldbody = root.find('.//worldbody')
if worldbody is not None:
    # Remove floor geom
    for geom in worldbody.findall('./geom[@name="floor"]'):
        worldbody.remove(geom)
    
    # Remove cameras that are direct children of worldbody
    for camera in worldbody.findall('./camera'):
        worldbody.remove(camera)
    
    # Remove lights that are direct children of worldbody
    for light in worldbody.findall('./light'):
        worldbody.remove(light)

# Remove material references from all geoms in worldbody
for geom in root.findall('.//geom'):
    if 'material' in geom.attrib:
        del geom.attrib['material']

# Remove all cameras and lights from nested bodies too
for camera in root.findall('.//camera'):
    parent = root.find(f'.//{camera.tag}/..') or worldbody
    if parent is not None:
        try:
            parent.remove(camera)
        except ValueError:
            pass

for light in root.findall('.//light'):
    parent = root.find(f'.//{light.tag}/..') or worldbody
    if parent is not None:
        try:
            parent.remove(light)
        except ValueError:
            pass

# Remove cameras and lights from torso body specifically
torso_body = root.find('.//body[@name="torso"]')
if torso_body is not None:
    for camera in torso_body.findall('./camera'):
        torso_body.remove(camera)
    for light in torso_body.findall('./light'):
        torso_body.remove(light)

# Save cleaned XML to a temporary file
tree.write(str(temp_mjcf_path), encoding='utf-8', xml_declaration=True)
print(f"Created cleaned temporary MJCF at: {temp_mjcf_path}")

print(f"Converting to URDF...")
print(f"  Output: {urdf_path}")

# Convert MJCF to URDF using mjcf-urdf-simple-converter
convert(str(temp_mjcf_path), str(urdf_path))

# Clean up temporary file
temp_mjcf_path.unlink()
print(f"Removed temporary file: {temp_mjcf_path}")

# ============================================================================
# Add visual geometry to the URDF since mjcf-urdf-simple-converter doesn't
# ============================================================================
print("Adding visual geometry to URDF...")

# ============================================================================
# COLOR CONFIGURATION
# ============================================================================
# Colors are specified as RGBA tuples: (Red, Green, Blue, Alpha)
# Each value ranges from 0.0 to 1.0
#
# Example colors:
#   - Orange:     (1.0, 0.5, 0.0, 1.0)
#   - Blue:       (0.4, 0.6, 0.8, 1.0)
#   - Red:        (0.8, 0.4, 0.4, 1.0)
#   - Green:      (0.4, 0.8, 0.4, 1.0)
#   - Yellow:     (1.0, 0.9, 0.0, 1.0)
#   - Purple:     (0.6, 0.3, 0.8, 1.0)
#   - Cyan:       (0.0, 0.8, 0.8, 1.0)
#   - White:      (1.0, 1.0, 1.0, 1.0)
#   - Gray:       (0.5, 0.5, 0.5, 1.0)
#
# To customize per-body-part colors, modify the 'color' field in each spec below.
# Current setup: All body parts are orange to match MuJoCo's default appearance.
# ============================================================================

# Shared color for all body parts (modify this to change all at once)
WALKER_COLOR = (1.0, 0.5, 0.0, 1.0)  # Orange - matches MuJoCo default

# Visual geometry specifications from the original MJCF
# Format: link_name -> specs dict with geometry and color info
#
# For capsules (MuJoCo's default): We create a cylinder + 2 sphere end caps
# MuJoCo capsule: size="radius half_length", so full cylinder length = 2 * half_length
# The spheres are placed at +/- half_length along the capsule axis
#
# To use different colors per body part, replace WALKER_COLOR with a custom tuple:
#   'color': (0.8, 0.4, 0.4, 1.0),  # Custom red for this part
visual_specs = {
    'torso': {
        'type': 'capsule',
        'radius': 0.07,
        'half_length': 0.3,  # Full length = 0.6
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 0.0, 0.0),
        'color': WALKER_COLOR,
    },
    'right_thigh': {
        'type': 'capsule',
        'radius': 0.05,
        'half_length': 0.225,  # Full length = 0.45
        'origin': (0.0, 0.0, -0.225),
        'rpy': (0.0, 0.0, 0.0),
        'color': WALKER_COLOR,
    },
    'right_leg': {
        'type': 'capsule',
        'radius': 0.04,
        'half_length': 0.25,  # Full length = 0.5
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 0.0, 0.0),
        'color': WALKER_COLOR,
    },
    'right_foot': {
        'type': 'capsule',
        'radius': 0.05,
        'half_length': 0.1,  # Full length = 0.2
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 1.5708, 0.0),  # Rotated 90 deg around Y (zaxis="1 0 0")
        'color': WALKER_COLOR,
    },
    'left_thigh': {
        'type': 'capsule',
        'radius': 0.05,
        'half_length': 0.225,
        'origin': (0.0, 0.0, -0.225),
        'rpy': (0.0, 0.0, 0.0),
        'color': WALKER_COLOR,
    },
    'left_leg': {
        'type': 'capsule',
        'radius': 0.04,
        'half_length': 0.25,
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 0.0, 0.0),
        'color': WALKER_COLOR,
    },
    'left_foot': {
        'type': 'capsule',
        'radius': 0.05,
        'half_length': 0.1,
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 1.5708, 0.0),
        'color': WALKER_COLOR,
    },
}


def add_capsule_visual(link_element, spec, link_name):
    """
    Add capsule visual geometry to a URDF link.
    
    URDF doesn't have a native capsule primitive, so we approximate it with:
    - 1 cylinder for the body
    - 2 spheres for the rounded end caps
    
    This creates the same visual appearance as MuJoCo's capsule geoms.
    
    Args:
        link_element: The XML link element to add visuals to
        spec: Dictionary with capsule specifications (radius, half_length, origin, rpy, color)
        link_name: Name of the link (for material naming)
    """
    radius = spec['radius']
    half_length = spec['half_length']
    origin = spec['origin']
    rpy = spec['rpy']
    color = spec['color']
    
    # Cylinder length is the distance between sphere centers
    cylinder_length = 2 * half_length
    
    # Calculate sphere positions in local frame (before rotation)
    # Spheres are at +/- half_length along the Z axis (cylinder axis)
    import math
    
    # For rotated capsules (like feet), we need to transform sphere positions
    # rpy = (roll, pitch, yaw) = rotation around (x, y, z)
    # For pitch rotation (around Y), the Z axis rotates into X-Z plane
    pitch = rpy[1]
    
    # Sphere offsets in local capsule frame (along cylinder axis = local Z)
    if abs(pitch) > 0.01:  # Capsule is rotated (like the feet)
        # After pitch rotation, local Z becomes: (sin(pitch), 0, cos(pitch))
        sphere1_offset = (
            half_length * math.sin(pitch),
            0.0,
            half_length * math.cos(pitch)
        )
        sphere2_offset = (
            -half_length * math.sin(pitch),
            0.0,
            -half_length * math.cos(pitch)
        )
    else:  # No rotation, spheres are along Z axis
        sphere1_offset = (0.0, 0.0, half_length)
        sphere2_offset = (0.0, 0.0, -half_length)
    
    # --- Cylinder (main body) ---
    visual_cyl = ET.SubElement(link_element, 'visual')
    origin_cyl = ET.SubElement(visual_cyl, 'origin')
    origin_cyl.set('xyz', f"{origin[0]} {origin[1]} {origin[2]}")
    origin_cyl.set('rpy', f"{rpy[0]} {rpy[1]} {rpy[2]}")
    
    geometry_cyl = ET.SubElement(visual_cyl, 'geometry')
    cylinder = ET.SubElement(geometry_cyl, 'cylinder')
    cylinder.set('radius', str(radius))
    cylinder.set('length', str(cylinder_length))
    
    material_cyl = ET.SubElement(visual_cyl, 'material')
    material_cyl.set('name', f"{link_name}_material")
    color_cyl = ET.SubElement(material_cyl, 'color')
    color_cyl.set('rgba', f"{color[0]} {color[1]} {color[2]} {color[3]}")
    
    # --- Sphere 1 (top cap) ---
    visual_s1 = ET.SubElement(link_element, 'visual')
    origin_s1 = ET.SubElement(visual_s1, 'origin')
    s1_pos = (origin[0] + sphere1_offset[0], origin[1] + sphere1_offset[1], origin[2] + sphere1_offset[2])
    origin_s1.set('xyz', f"{s1_pos[0]} {s1_pos[1]} {s1_pos[2]}")
    origin_s1.set('rpy', "0 0 0")
    
    geometry_s1 = ET.SubElement(visual_s1, 'geometry')
    sphere1 = ET.SubElement(geometry_s1, 'sphere')
    sphere1.set('radius', str(radius))
    
    material_s1 = ET.SubElement(visual_s1, 'material')
    material_s1.set('name', f"{link_name}_material")  # Reuse same material
    color_s1 = ET.SubElement(material_s1, 'color')
    color_s1.set('rgba', f"{color[0]} {color[1]} {color[2]} {color[3]}")
    
    # --- Sphere 2 (bottom cap) ---
    visual_s2 = ET.SubElement(link_element, 'visual')
    origin_s2 = ET.SubElement(visual_s2, 'origin')
    s2_pos = (origin[0] + sphere2_offset[0], origin[1] + sphere2_offset[1], origin[2] + sphere2_offset[2])
    origin_s2.set('xyz', f"{s2_pos[0]} {s2_pos[1]} {s2_pos[2]}")
    origin_s2.set('rpy', "0 0 0")
    
    geometry_s2 = ET.SubElement(visual_s2, 'geometry')
    sphere2 = ET.SubElement(geometry_s2, 'sphere')
    sphere2.set('radius', str(radius))
    
    material_s2 = ET.SubElement(visual_s2, 'material')
    material_s2.set('name', f"{link_name}_material")
    color_s2 = ET.SubElement(material_s2, 'color')
    color_s2.set('rgba', f"{color[0]} {color[1]} {color[2]} {color[3]}")


# Parse the generated URDF
urdf_tree = ET.parse(urdf_path)
urdf_root = urdf_tree.getroot()

# Add visual elements to each link
for link in urdf_root.findall('.//link'):
    link_name = link.get('name')
    
    if link_name in visual_specs:
        spec = visual_specs[link_name]
        
        if spec['type'] == 'capsule':
            add_capsule_visual(link, spec, link_name)
        else:
            # Fallback: simple cylinder (no rounded caps)
            visual = ET.SubElement(link, 'visual')
            origin = ET.SubElement(visual, 'origin')
            origin.set('xyz', f"{spec['origin'][0]} {spec['origin'][1]} {spec['origin'][2]}")
            origin.set('rpy', f"{spec['rpy'][0]} {spec['rpy'][1]} {spec['rpy'][2]}")
            
            geometry = ET.SubElement(visual, 'geometry')
            cylinder = ET.SubElement(geometry, 'cylinder')
            cylinder.set('radius', str(spec['radius']))
            cylinder.set('length', str(2 * spec['half_length']))
            
            material = ET.SubElement(visual, 'material')
            material.set('name', f"{link_name}_material")
            color = ET.SubElement(material, 'color')
            color.set('rgba', f"{spec['color'][0]} {spec['color'][1]} {spec['color'][2]} {spec['color'][3]}")
        
        print(f"  Added capsule visual to: {link_name}")

# Write the updated URDF
urdf_tree.write(str(urdf_path), encoding='utf-8', xml_declaration=True)

print("Conversion complete!")
print(f"\nURDF saved to: {urdf_path}")