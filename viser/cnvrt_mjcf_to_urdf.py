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

# Visual geometry specifications from the original MJCF
# Format: link_name -> (radius, length, origin_xyz, origin_rpy, color_rgba)
# For capsules in URDF, we use cylinder + 2 spheres, but viser supports cylinders well
# MuJoCo capsule: size="radius half_length", so full length = 2 * half_length
visual_specs = {
    'torso': {
        'type': 'cylinder',
        'radius': 0.07,
        'length': 0.6,  # 2 * 0.3
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 0.0, 0.0),
        'color': (0.4, 0.6, 0.8, 1.0),  # Blue-ish
    },
    'right_thigh': {
        'type': 'cylinder',
        'radius': 0.05,
        'length': 0.45,  # 2 * 0.225
        'origin': (0.0, 0.0, -0.225),
        'rpy': (0.0, 0.0, 0.0),
        'color': (0.8, 0.4, 0.4, 1.0),  # Red-ish
    },
    'right_leg': {
        'type': 'cylinder',
        'radius': 0.04,
        'length': 0.5,  # 2 * 0.25
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 0.0, 0.0),
        'color': (0.8, 0.4, 0.4, 1.0),
    },
    'right_foot': {
        'type': 'cylinder',
        'radius': 0.05,
        'length': 0.2,  # 2 * 0.1
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 1.5708, 0.0),  # Rotated 90 deg around Y (zaxis="1 0 0")
        'color': (0.8, 0.4, 0.4, 1.0),
    },
    'left_thigh': {
        'type': 'cylinder',
        'radius': 0.05,
        'length': 0.45,
        'origin': (0.0, 0.0, -0.225),
        'rpy': (0.0, 0.0, 0.0),
        'color': (0.4, 0.8, 0.4, 1.0),  # Green-ish
    },
    'left_leg': {
        'type': 'cylinder',
        'radius': 0.04,
        'length': 0.5,
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 0.0, 0.0),
        'color': (0.4, 0.8, 0.4, 1.0),
    },
    'left_foot': {
        'type': 'cylinder',
        'radius': 0.05,
        'length': 0.2,
        'origin': (0.0, 0.0, 0.0),
        'rpy': (0.0, 1.5708, 0.0),
        'color': (0.4, 0.8, 0.4, 1.0),
    },
}

# Parse the generated URDF
urdf_tree = ET.parse(urdf_path)
urdf_root = urdf_tree.getroot()

# Add visual elements to each link
for link in urdf_root.findall('.//link'):
    link_name = link.get('name')
    
    if link_name in visual_specs:
        spec = visual_specs[link_name]
        
        # Create visual element
        visual = ET.SubElement(link, 'visual')
        
        # Origin
        origin = ET.SubElement(visual, 'origin')
        origin.set('xyz', f"{spec['origin'][0]} {spec['origin'][1]} {spec['origin'][2]}")
        origin.set('rpy', f"{spec['rpy'][0]} {spec['rpy'][1]} {spec['rpy'][2]}")
        
        # Geometry
        geometry = ET.SubElement(visual, 'geometry')
        if spec['type'] == 'cylinder':
            cylinder = ET.SubElement(geometry, 'cylinder')
            cylinder.set('radius', str(spec['radius']))
            cylinder.set('length', str(spec['length']))
        
        # Material with color
        material = ET.SubElement(visual, 'material')
        material.set('name', f"{link_name}_material")
        color = ET.SubElement(material, 'color')
        color.set('rgba', f"{spec['color'][0]} {spec['color'][1]} {spec['color'][2]} {spec['color'][3]}")
        
        print(f"  Added visual geometry to: {link_name}")

# Write the updated URDF
urdf_tree.write(str(urdf_path), encoding='utf-8', xml_declaration=True)

print("Conversion complete!")
print(f"\nURDF saved to: {urdf_path}")