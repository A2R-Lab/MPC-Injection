"""
Quick script to convert a MuJoCo MJCF model to a proper URDF file.

Uses mjcf-urdf-simple-converter for proper MJCF to URDF conversion.
https://github.com/Yasu31/mjcf_urdf_simple_converter

NOTE: The xmls come from ~/MPC-RL/mpc_rl/tasks/walker/walker_modified.xml

This script first cleans the MJCF (removes floor, cameras, lights, material refs)
then converts to URDF format.
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

print("Conversion complete!")
print(f"\nURDF saved to: {urdf_path}")