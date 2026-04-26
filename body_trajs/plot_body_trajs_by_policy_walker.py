#!/usr/bin/env python3
"""
Script to 3D plot the trajectory of a body/limb of a trained SAC-MPC walker model.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

FONT_SIZE = 18

plt.rcParams.update(
    {
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'figure.titlesize': FONT_SIZE + 2,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
        'legend.fontsize': FONT_SIZE,
    }
)


def apply_axis_font_sizes(ax):
    """Apply paper-friendly tick label sizes to 2D and 3D axes."""
    ax.tick_params(axis='x', labelsize=FONT_SIZE - 2)
    ax.tick_params(axis='y', labelsize=FONT_SIZE - 2)
    if hasattr(ax, 'zaxis'):
        ax.tick_params(axis='z', labelsize=FONT_SIZE - 2)

def load_trajectory_data(npz_file: Path, data_type: str = 'observations'):
    """
    Load trajectory data from a file.
    
    Args:
        npz_file: Path to the .npz file
        data_type: Type of data to load - 'observations' or 'body_physics'
    
    Returns:
        Numpy array of data (timesteps, feature_dim)
    """
    data = np.load(npz_file)
    
    if data_type == 'observations':
        return data['observations']
    
    elif data_type == 'body_physics':
        # Concatenate all body positions and orientations into a single feature vector
        body_names = ['torso', 'right_thigh', 'right_leg', 'right_foot', 
                      'left_thigh', 'left_leg', 'left_foot']
        
        features = []
        for body_name in body_names:
            # Add positions (3D)
            features.append(data[f'pos_{body_name}'])
            # Add orientations (4D quaternions)
            features.append(data[f'quat_{body_name}'])
        
        # Concatenate along feature dimension
        # Shape: (timesteps, 7 bodies * (3 pos + 4 quat)) = (timesteps, 49)
        combined_data = np.concatenate(features, axis=1)
        return combined_data
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'observations' or 'body_physics'.")


def extract_checkpoint_number(filename: str):
    """
    Extract the checkpoint number from a filename like 'trajectories_step_100000.npz'.
    
    Args:
        filename: Name of the file
    
    Returns:
        Checkpoint number as integer, or None if pattern doesn't match
    """
    match = re.search(r'trajectories_step_(\d+)\.npz', filename)
    if match:
        return int(match.group(1))
    return None


def plot_torso_height(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Test function to plot 3D torso trajectories across all checkpoints.
    Y-axis represents checkpoint number, showing evolution over training.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    # Find all trajectory files
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    # Get common checkpoints and filter to desired range
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    # Filter to only include 25k, 100k, and 200k
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    if common_checkpoints:
        print(f"Checkpoint range: {common_checkpoints[0]} to {common_checkpoints[-1]}")
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories for each checkpoint
    for checkpoint in common_checkpoints:
        # Load data
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        # Extract torso Z height
        torso_z_pct_1 = data1['pos_torso'][:, 2]  # Z height
        torso_z_pct_2 = data2['pos_torso'][:, 2]  # Z height
        
        # Create time arrays (timesteps)
        time_pct_1 = np.arange(len(torso_z_pct_1))
        time_pct_2 = np.arange(len(torso_z_pct_2))
        
        # Create Y values as checkpoint number for each timestep
        y_pct_1 = np.full_like(torso_z_pct_1, checkpoint)
        y_pct_2 = np.full_like(torso_z_pct_2, checkpoint)
        
        # Plot trajectories at this checkpoint level
        ax.plot(time_pct_1, y_pct_1, torso_z_pct_1,
                c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(time_pct_2, y_pct_2, torso_z_pct_2,
                c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        # Plot starting points
        ax.scatter(time_pct_1[0], checkpoint, torso_z_pct_1[0],
                  c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(time_pct_2[0], checkpoint, torso_z_pct_2[0],
                  c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    # Add legend with manual entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    # Labels and title
    ax.set_xlabel('Time (timesteps)', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE)
    ax.set_zlabel('Torso Height (m)', fontsize=FONT_SIZE)
    ax.set_title('Torso Trajectory Evolution: Checkpoints 25k, 100k, 200k', fontsize=FONT_SIZE + 2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    
    # Set view angle for better perspective
    # elev: vertical angle (higher = more from above), azim: horizontal rotation
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "torso_height_across_checkpoints.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    
    # Show the plot
    plt.show()


def plot_torso_position(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Plot 3D torso position trajectory evolution.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    # Find all trajectory files
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    # Get common checkpoints and filter to desired range
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    # Filter to only include 25k, 100k, and 200k
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    if common_checkpoints:
        print(f"Checkpoint range: {common_checkpoints[0]} to {common_checkpoints[-1]}")
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories for each checkpoint
    for checkpoint in common_checkpoints:
        # Load data
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        # Extract torso position
        torso_pos_pct_1 = data1['pos_torso']  # XYZ position
        torso_pos_pct_2 = data2['pos_torso']  # XYZ position
        
        # Extract X (forward) and Z (height) positions
        torso_x_pct_1 = torso_pos_pct_1[:, 0]  # Forward position
        torso_z_pct_1 = torso_pos_pct_1[:, 2]  # Height
        torso_x_pct_2 = torso_pos_pct_2[:, 0]  # Forward position
        torso_z_pct_2 = torso_pos_pct_2[:, 2]  # Height
        
        # Create Y values as checkpoint number for each timestep
        y_pct_1 = np.full_like(torso_z_pct_1, checkpoint)
        y_pct_2 = np.full_like(torso_z_pct_2, checkpoint)
        
        # Plot 2D trajectories (X-Z) at this checkpoint level
        ax.plot(torso_x_pct_1, y_pct_1, torso_z_pct_1,
                c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(torso_x_pct_2, y_pct_2, torso_z_pct_2,
                c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        # Plot starting points
        ax.scatter(torso_x_pct_1[0], checkpoint, torso_z_pct_1[0],
                  c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(torso_x_pct_2[0], checkpoint, torso_z_pct_2[0],
                  c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    # Add legend with manual entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    # Labels and title
    ax.set_xlabel('Torso X Position (m)', fontsize=FONT_SIZE, labelpad=10)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE, labelpad=20)
    ax.set_zlabel('Torso Height (m)', fontsize=FONT_SIZE, labelpad=10)
    ax.set_title('Torso Trajectory: Checkpoints 25k, 100k, 150k, 200k', fontsize=FONT_SIZE+2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    
    # Adjust margins to reduce empty space
    ax.margins(x=0.01, y=0.01, z=0.01)
    
    # Set view angle for better perspective
    # elev: vertical angle (higher = more from above), azim: horizontal rotation
    # dist: zoom level (lower = zoom in, higher = zoom out), default is 10
    ax.view_init(elev=45, azim=75)
    ax.dist = 8
    
    #plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "torso_xz_trajectory_evolution.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.15)
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    
    # Show the plot
    plt.show()


def plot_left_thigh_position(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Plot 3D trajectory of left thigh across checkpoints.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for checkpoint in common_checkpoints:
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        pos_pct_1 = data1['pos_left_thigh']
        pos_pct_2 = data2['pos_left_thigh']
        
        x_pct_1 = pos_pct_1[:, 0]
        z_pct_1 = pos_pct_1[:, 2]
        x_pct_2 = pos_pct_2[:, 0]
        z_pct_2 = pos_pct_2[:, 2]
        
        y_pct_1 = np.full_like(z_pct_1, checkpoint)
        y_pct_2 = np.full_like(z_pct_2, checkpoint)
        
        ax.plot(x_pct_1, y_pct_1, z_pct_1, c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(x_pct_2, y_pct_2, z_pct_2, c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        ax.scatter(x_pct_1[0], checkpoint, z_pct_1[0], c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(x_pct_2[0], checkpoint, z_pct_2[0], c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    ax.set_xlabel('Left Thigh X Position (m)', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE)
    ax.set_zlabel('Left Thigh Height (m)', fontsize=FONT_SIZE)
    ax.set_title('Left Thigh 2D Trajectory Evolution (X-Z plane)', fontsize=FONT_SIZE + 2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "left_thigh_xz_trajectory_evolution.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    plt.show()


def plot_left_leg_position(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Plot 3D trajectory of left leg across checkpoints.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for checkpoint in common_checkpoints:
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        pos_pct_1 = data1['pos_left_leg']
        pos_pct_2 = data2['pos_left_leg']
        
        x_pct_1 = pos_pct_1[:, 0]
        z_pct_1 = pos_pct_1[:, 2]
        x_pct_2 = pos_pct_2[:, 0]
        z_pct_2 = pos_pct_2[:, 2]
        
        y_pct_1 = np.full_like(z_pct_1, checkpoint)
        y_pct_2 = np.full_like(z_pct_2, checkpoint)
        
        ax.plot(x_pct_1, y_pct_1, z_pct_1, c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(x_pct_2, y_pct_2, z_pct_2, c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        ax.scatter(x_pct_1[0], checkpoint, z_pct_1[0], c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(x_pct_2[0], checkpoint, z_pct_2[0], c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    ax.set_xlabel('Left Leg X Position (m)', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE)
    ax.set_zlabel('Left Leg Height (m)', fontsize=FONT_SIZE)
    ax.set_title('Left Leg 2D Trajectory Evolution (X-Z plane)', fontsize=FONT_SIZE + 2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "left_leg_xz_trajectory_evolution.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    plt.show()


def plot_left_foot_position(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Plot 3D trajectory of left foot across checkpoints.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for checkpoint in common_checkpoints:
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        pos_pct_1 = data1['pos_left_foot']
        pos_pct_2 = data2['pos_left_foot']
        
        x_pct_1 = pos_pct_1[:, 0]
        z_pct_1 = pos_pct_1[:, 2]
        x_pct_2 = pos_pct_2[:, 0]
        z_pct_2 = pos_pct_2[:, 2]
        
        y_pct_1 = np.full_like(z_pct_1, checkpoint)
        y_pct_2 = np.full_like(z_pct_2, checkpoint)
        
        ax.plot(x_pct_1, y_pct_1, z_pct_1, c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(x_pct_2, y_pct_2, z_pct_2, c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        ax.scatter(x_pct_1[0], checkpoint, z_pct_1[0], c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(x_pct_2[0], checkpoint, z_pct_2[0], c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    ax.set_xlabel('Left Foot X Position (m)', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE)
    ax.set_zlabel('Left Foot Height (m)', fontsize=FONT_SIZE)
    ax.set_title('Left Foot 2D Trajectory Evolution (X-Z plane)', fontsize=FONT_SIZE + 2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "left_foot_xz_trajectory_evolution.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    plt.show()


def plot_right_thigh_position(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Plot 3D trajectory of right thigh across checkpoints.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for checkpoint in common_checkpoints:
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        pos_pct_1 = data1['pos_right_thigh']
        pos_pct_2 = data2['pos_right_thigh']
        
        x_pct_1 = pos_pct_1[:, 0]
        z_pct_1 = pos_pct_1[:, 2]
        x_pct_2 = pos_pct_2[:, 0]
        z_pct_2 = pos_pct_2[:, 2]
        
        y_pct_1 = np.full_like(z_pct_1, checkpoint)
        y_pct_2 = np.full_like(z_pct_2, checkpoint)
        
        ax.plot(x_pct_1, y_pct_1, z_pct_1, c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(x_pct_2, y_pct_2, z_pct_2, c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        ax.scatter(x_pct_1[0], checkpoint, z_pct_1[0], c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(x_pct_2[0], checkpoint, z_pct_2[0], c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    ax.set_xlabel('Right Thigh X Position (m)', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE)
    ax.set_zlabel('Right Thigh Height (m)', fontsize=FONT_SIZE)
    ax.set_title('Right Thigh 2D Trajectory Evolution (X-Z plane)', fontsize=FONT_SIZE + 2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "right_thigh_xz_trajectory_evolution.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    plt.show()


def plot_right_leg_position(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Plot 3D trajectory of right leg across checkpoints.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for checkpoint in common_checkpoints:
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        pos_pct_1 = data1['pos_right_leg']
        pos_pct_2 = data2['pos_right_leg']
        
        x_pct_1 = pos_pct_1[:, 0]
        z_pct_1 = pos_pct_1[:, 2]
        x_pct_2 = pos_pct_2[:, 0]
        z_pct_2 = pos_pct_2[:, 2]
        
        y_pct_1 = np.full_like(z_pct_1, checkpoint)
        y_pct_2 = np.full_like(z_pct_2, checkpoint)
        
        ax.plot(x_pct_1, y_pct_1, z_pct_1, c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(x_pct_2, y_pct_2, z_pct_2, c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        ax.scatter(x_pct_1[0], checkpoint, z_pct_1[0], c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(x_pct_2[0], checkpoint, z_pct_2[0], c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    ax.set_xlabel('Right Leg X Position (m)', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE)
    ax.set_zlabel('Right Leg Height (m)', fontsize=FONT_SIZE)
    ax.set_title('Right Leg 2D Trajectory Evolution (X-Z plane)', fontsize=FONT_SIZE + 2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "right_leg_xz_trajectory_evolution.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    plt.show()


def plot_right_foot_position(dir1: Path, dir2: Path, checkpoints=[25000, 100000, 150000, 200000]):
    """
    Plot 3D trajectory of right foot across checkpoints.
    
    Args:
        dir1: Path to first trajectory data directory
        dir2: Path to second trajectory data directory
        checkpoints: List of checkpoint steps to plot
    """
    
    files1 = {extract_checkpoint_number(f.name): f for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f for f in dir2.glob('trajectories_step_*.npz')}
    
    common_checkpoints = sorted(set(files1.keys()) & set(files2.keys()))
    common_checkpoints = [cp for cp in common_checkpoints if cp in checkpoints]
    
    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for checkpoint in common_checkpoints:
        data1 = np.load(files1[checkpoint])
        data2 = np.load(files2[checkpoint])
        
        pos_pct_1 = data1['pos_right_foot']
        pos_pct_2 = data2['pos_right_foot']
        
        x_pct_1 = pos_pct_1[:, 0]
        z_pct_1 = pos_pct_1[:, 2]
        x_pct_2 = pos_pct_2[:, 0]
        z_pct_2 = pos_pct_2[:, 2]
        
        y_pct_1 = np.full_like(z_pct_1, checkpoint)
        y_pct_2 = np.full_like(z_pct_2, checkpoint)
        
        ax.plot(x_pct_1, y_pct_1, z_pct_1, c='#1f77b4', linewidth=2.0, alpha=0.6)
        ax.plot(x_pct_2, y_pct_2, z_pct_2, c='#ff7f0e', linewidth=2.0, alpha=0.6)
        
        ax.scatter(x_pct_1[0], checkpoint, z_pct_1[0], c='#1f77b4', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
        ax.scatter(x_pct_2[0], checkpoint, z_pct_2[0], c='#ff7f0e', s=50, marker='o', edgecolors='black', linewidths=1, alpha=0.9)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='0% MPC-Injection'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='25% MPC-Injection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label='Start Point')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE, loc='upper left')
    
    ax.set_xlabel('Right Foot X Position (m)', fontsize=FONT_SIZE)
    ax.set_ylabel('Training Checkpoint', fontsize=FONT_SIZE)
    ax.set_zlabel('Right Foot Height (m)', fontsize=FONT_SIZE)
    ax.set_title('Right Foot 2D Trajectory Evolution (X-Z plane)', fontsize=FONT_SIZE + 2, fontweight='bold')
    apply_axis_font_sizes(ax)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "right_foot_xz_trajectory_evolution.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory evolution plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Define directories for trajectory data
    base_dir = Path(__file__).parent / "model_traj_data_walker"
    dir1 = base_dir / "walker-walk-SAC-MPC-20260107-112012-percentage-0pct"
    dir2 = base_dir / "walker-walk-SAC-MPC-20260107-112659-percentage-25pct"
    
    # Define checkpoints to plot
    checkpoints = [25000, 100000, 150000, 200000]
    
    # Generate all plots
    plot_torso_position(dir1, dir2, checkpoints=checkpoints)
    #plot_left_thigh_position(dir1, dir2, checkpoints=checkpoints)
    #plot_left_leg_position(dir1, dir2, checkpoints=checkpoints)
    #plot_left_foot_position(dir1, dir2, checkpoints=checkpoints)
    #plot_right_thigh_position(dir1, dir2, checkpoints=checkpoints)
    #plot_right_leg_position(dir1, dir2, checkpoints=checkpoints)
    #plot_right_foot_position(dir1, dir2, checkpoints=checkpoints)
