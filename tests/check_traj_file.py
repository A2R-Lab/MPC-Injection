#!/usr/bin/env python3
"""
Check the actual trajectory file to see if perturbation was recorded
"""

import numpy as np
from pathlib import Path

# Load a trajectory file that should have perturbation
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trajectory", type=Path)
traj_file = parser.parse_args().trajectory

if traj_file.exists():
    data = np.load(traj_file, allow_pickle=True)
    
    print("Trajectory file contents:")
    print("="*80)
    print(f"Keys in file: {list(data.keys())}")
    
    if 'perturbation_enabled' in data:
        print(f"\nPerturbation enabled: {data['perturbation_enabled']}")
        print(f"Perturbation applied: {data['perturbation_applied']}")
        print(f"Perturbation force (x): {data['perturbation_force_x']} N")
        print(f"Perturbation timestep: {data['perturbation_timestep']}")
    else:
        print("\nNo perturbation data found in file!")
        print("This trajectory was recorded WITHOUT the perturbation feature.")
    
    # Check torso positions around the perturbation timestep
    print(f"\nTotal timesteps: {data['timesteps']}")
    
    if 'pos_torso' in data:
        torso_pos = data['pos_torso']
        print(f"\nTorso positions (X-axis) around expected perturbation time:")
        
        # If perturbation was at timestep 500
        if 'perturbation_timestep' in data:
            pert_step = int(data['perturbation_timestep'])
        else:
            pert_step = 500  # default
            
        start = max(0, pert_step - 5)
        end = min(len(torso_pos), pert_step + 10)
        
        for i in range(start, end):
            marker = " <-- PERTURBATION" if i == pert_step else ""
            print(f"  Step {i:3d}: X = {torso_pos[i][0]:8.5f}{marker}")
        
        # Calculate velocity changes
        if pert_step + 2 < len(torso_pos):
            vel_before = torso_pos[pert_step] - torso_pos[pert_step - 1]
            vel_during = torso_pos[pert_step + 1] - torso_pos[pert_step]
            vel_after = torso_pos[pert_step + 2] - torso_pos[pert_step + 1]
            
            print(f"\nVelocity changes (X-axis):")
            print(f"  Before perturbation: {vel_before[0]:8.6f} m/step")
            print(f"  During perturbation: {vel_during[0]:8.6f} m/step")
            print(f"  After perturbation:  {vel_after[0]:8.6f} m/step")
            print(f"  Change: {vel_during[0] - vel_before[0]:8.6f} m/step")
else:
    print(f"File not found: {traj_file}")
