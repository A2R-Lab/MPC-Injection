#!/usr/bin/env python3
"""
Script to create UMAP plots comparing state distributions from different training runs.

This script loads trajectory data from multiple model checkpoints and creates UMAP 
visualizations to compare how the state distributions evolve during training.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import re

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = 26
LEGEND_FONT_SIZE = 24

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


def find_matching_trajectory_files(dir1: Path, dir2: Path):
    """
    Find pairs of trajectory files with matching checkpoint numbers in two directories.
    
    Args:
        dir1: First directory path
        dir2: Second directory path
    
    Returns:
        List of tuples: [(checkpoint_num, path1, path2), ...]
    """
    # Get all trajectory files from both directories
    files1 = {extract_checkpoint_number(f.name): f 
              for f in dir1.glob('trajectories_step_*.npz')}
    files2 = {extract_checkpoint_number(f.name): f 
              for f in dir2.glob('trajectories_step_*.npz')}
    
    # Find matching checkpoint numbers
    common_checkpoints = set(files1.keys()) & set(files2.keys())
    
    # Create list of matching pairs
    matching_pairs = [(checkpoint, files1[checkpoint], files2[checkpoint]) 
                      for checkpoint in sorted(common_checkpoints)]
    
    return matching_pairs


def create_umap_comparison_plot(obs1, obs2, label1, label2, checkpoint_num, output_path: Path, data_type: str = 'observations'):
    """
    Create a UMAP plot comparing two sets of trajectory data.
    
    Args:
        obs1: Data from first trajectory (N1, feature_dim)
        obs2: Data from second trajectory (N2, feature_dim)
        label1: Label for first dataset
        label2: Label for second dataset
        checkpoint_num: Checkpoint number for the title
        output_path: Path to save the plot
        data_type: Type of data being plotted ('observations' or 'body_physics')
    """
    # Combine observations for consistent UMAP embedding
    combined_obs = np.vstack([obs1, obs2])
    n1 = len(obs1)
    
    print(f"  Creating UMAP embedding for {len(combined_obs)} total observations...")
    
    # Create UMAP embedding
    """
    Notes on what each parameter does for UMAP embedding:

    - n_neighbors:
        Controls how UMAP balances local vs global structure in the data.
        - Smaller values (5-10) focus on local relationships (tight clusters become more separated),
          but may fragment natural groupings. Good to find fine-grained structure.
        - Larger values (30-100) emphasize global structure. Clusters may merge together, but overall
          data topology is more accurate
    - min_dist:
        Controls how tightly UMAP packs points together in the embedding space.
        - Smaller values (0-0.05) tightly packs points together. Better for seeing cluster structure,
          but may hide variation within clusters.
        - Larger values (0.3-0.99) spread points more uniformly. Creates looser, more evenly distributed
          embeddings. Better for seeing continuum of variation but clusters become less distinct.
    - n_components:
        Number of dimensions in the output embedding space.
    - metric:
        Distance metric used to measure similarity b/w points in high-dim space
        - euclidean: best for continuous features
        - manhattan: less sensitive to outliers
        - cosine: measures angles b/w vectors. Good for when magnitude doesn't matter
        - correlation: good for time series or when you care about patterns, not absolute values

    """
    reducer = UMAP(
        n_neighbors=10,
        min_dist=0.3,
        n_components=3,
        random_state=42,
        metric='correlation'
    )
    
    embedding = reducer.fit_transform(combined_obs)
    
    # Split back into two datasets
    embedding1 = embedding[:n1]
    embedding2 = embedding[n1:]
    
    # Create 3D plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot both distributions
    ax.scatter(embedding1[:, 0], embedding1[:, 1], embedding1[:, 2], # embedding1[:, 2] for 3D
              c='#1f77b4', alpha=0.6, s=10, label=label1, rasterized=True)
    ax.scatter(embedding2[:, 0], embedding2[:, 1], embedding2[:, 2], # embedding1[:, 2] for 3D
              c='#ff7f0e', alpha=0.6, s=10, label=label2, rasterized=True)
    
    ax.set_xlabel('UMAP Dim 1', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=12)
    ax.set_ylabel('UMAP Dim 2', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=12)
    ax.set_zlabel('UMAP Dim 3', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=12)  # For 3D plot
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='z', labelsize=TICK_LABEL_FONT_SIZE)
    
    # Create title based on data type
    data_type_label = 'Observations' if data_type == 'observations' else 'State'
    ax.set_title(f'{data_type_label} Distributions at Checkpoint {checkpoint_num:,}', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='best')
    ax.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.04, right=0.82, bottom=0.08, top=0.90)
    bbox_extra_artists = [
        ax.xaxis.label,
        ax.yaxis.label,
        ax.zaxis.label,
        ax.title,
    ]
    if ax.legend_ is not None:
        bbox_extra_artists.append(ax.legend_)

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        bbox_extra_artists=bbox_extra_artists,
        pad_inches=0.3,
    )
    plt.close(fig)
    
    print(f"  Saved plot to: {output_path}")


def main():
    """
    Main function to create UMAP comparison plots for all matching checkpoint pairs.
    
    EDIT THESE PARAMETERS:
    """
    # ============================================================================
    # USER CONFIGURATION - Edit these values as needed
    # ============================================================================
    
    # Directories containing trajectory data
    DIR1 = Path(__file__).parent / "model_traj_data_walker/walker-walk-SAC-MPC-20260107-112012-percentage-0pct"
    DIR2 = Path(__file__).parent / "model_traj_data_walker/walker-walk-SAC-MPC-20260107-112659-percentage-25pct"
    
    # Labels for the two datasets (used in plot legend)
    LABEL1 = "0% MPC-Injection"
    LABEL2 = "25% MPC-Injection"
    
    # Data type to plot: 'observations' or 'body_physics'
    # - 'observations': Use the policy observations (normalized state vector)
    # - 'body_physics': Use raw body positions and orientations from physics engine
    DATA_TYPE = 'body_physics'
    
    # Output directory for plots
    OUTPUT_DIR = Path(__file__).parent.parent / "plots/umap_plots3d"
    
    # Output filename pattern
    OUTPUT_PATTERN = "umap_graph_0pct_25pct_at_checkpoint_{checkpoint}.png"
    
    # ============================================================================
    # END USER CONFIGURATION
    # ============================================================================
    
    # Verify directories exist
    if not DIR1.exists():
        print(f"ERROR: Directory not found: {DIR1}")
        return
    if not DIR2.exists():
        print(f"ERROR: Directory not found: {DIR2}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Creating UMAP State Distribution Comparison Plots")
    print("="*80)
    print(f"\nDirectory 1: {DIR1}")
    print(f"Directory 2: {DIR2}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Find matching trajectory files
    print("\nFinding matching trajectory files...")
    matching_pairs = find_matching_trajectory_files(DIR1, DIR2)
    
    if not matching_pairs:
        print("ERROR: No matching trajectory files found!")
        return
    
    print(f"Found {len(matching_pairs)} matching checkpoint pairs")
    
    # Process each matching pair
    print("\n" + "="*80)
    for i, (checkpoint_num, file1, file2) in enumerate(matching_pairs, 1):
        print(f"\n[{i}/{len(matching_pairs)}] Processing checkpoint: {checkpoint_num}")
        print("-"*80)
        
        try:
            # Load trajectory data
            data_type_label = 'observations' if DATA_TYPE == 'observations' else 'body physics data'
            print(f"  Loading {data_type_label} from: {file1.name}")
            obs1 = load_trajectory_data(file1, data_type=DATA_TYPE)
            print(f"    Shape: {obs1.shape}")
            
            print(f"  Loading {data_type_label} from: {file2.name}")
            obs2 = load_trajectory_data(file2, data_type=DATA_TYPE)
            print(f"    Shape: {obs2.shape}")
            
            # Create output filename
            output_filename = OUTPUT_PATTERN.format(checkpoint=checkpoint_num)
            output_path = OUTPUT_DIR / output_filename
            
            # Create UMAP plot
            create_umap_comparison_plot(
                obs1, obs2, 
                LABEL1, LABEL2, 
                checkpoint_num, 
                output_path,
                data_type=DATA_TYPE
            )
            
        except Exception as e:
            print(f"  ERROR: Failed to process checkpoint {checkpoint_num}")
            print(f"  Error details: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
