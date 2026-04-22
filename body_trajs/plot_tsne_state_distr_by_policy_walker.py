#!/usr/bin/env python3
"""
Script to create t-SNE plots comparing state distributions from different training runs.

This script loads trajectory data from multiple model checkpoints and creates t-SNE 
visualizations to compare how the state distributions evolve during training.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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


def create_tsne_comparison_plot(obs1, obs2, label1, label2, checkpoint_num, output_path: Path, data_type: str = 'observations'):
    """
    Create a t-SNE plot comparing two sets of trajectory data.
    
    Args:
        obs1: Data from first trajectory (N1, feature_dim)
        obs2: Data from second trajectory (N2, feature_dim)
        label1: Label for first dataset
        label2: Label for second dataset
        checkpoint_num: Checkpoint number for the title
        output_path: Path to save the plot
        data_type: Type of data being plotted ('observations' or 'body_physics')
    """
    # Combine observations for consistent t-SNE embedding
    combined_obs = np.vstack([obs1, obs2])
    n1 = len(obs1)
    
    print(f"  Creating t-SNE embedding for {len(combined_obs)} total observations...")
    
    # Create t-SNE embedding
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42,
        metric='euclidean',
        init='pca'
    )
    
    embedding = tsne.fit_transform(combined_obs)
    
    # Split back into two datasets
    embedding1 = embedding[:n1]
    embedding2 = embedding[n1:]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot both distributions
    ax.scatter(embedding1[:, 0], embedding1[:, 1], 
              c='#1f77b4', alpha=0.6, s=10, label=label1, rasterized=True)
    ax.scatter(embedding2[:, 0], embedding2[:, 1], 
              c='#ff7f0e', alpha=0.6, s=10, label=label2, rasterized=True)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=FONT_SIZE)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
    
    # Create title based on data type
    data_type_label = 'Observations' if data_type == 'observations' else 'Body Physics'
    ax.set_title(f'{data_type_label} Distribution Comparison at Checkpoint {checkpoint_num:,}', 
                fontsize=FONT_SIZE + 2, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot to: {output_path}")


def main():
    """
    Main function to create t-SNE comparison plots for all matching checkpoint pairs.
    
    EDIT THESE PARAMETERS:
    """
    # ============================================================================
    # USER CONFIGURATION - Edit these values as needed
    # ============================================================================
    
    # Directories containing trajectory data
    DIR1 = Path(__file__).parent / "model_traj_data/walker-walk-SAC-MPC-20260107-112012-percentage-0pct"
    DIR2 = Path(__file__).parent / "model_traj_data/walker-walk-SAC-MPC-20260107-113507-percentage-50pct"
    
    # Labels for the two datasets (used in plot legend)
    LABEL1 = "0% MPC-Injection"
    LABEL2 = "50% MPC-Injection"
    
    # Data type to plot: 'observations' or 'body_physics'
    # - 'observations': Use the policy observations (normalized state vector)
    # - 'body_physics': Use raw body positions and orientations from physics engine
    DATA_TYPE = 'body_physics'
    
    # Output directory for plots
    OUTPUT_DIR = Path(__file__).parent.parent / "plots/tsne_plots"
    
    # Output filename pattern
    OUTPUT_PATTERN = "tsne_graph_0pct_50pct_at_checkpoint_{checkpoint}.png"
    
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
    print("Creating t-SNE State Distribution Comparison Plots")
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
            
            # Create t-SNE plot
            create_tsne_comparison_plot(
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
