"""Surface generation utilities - core reward surface logic."""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import json
import os


def filter_normalize(param: np.ndarray) -> np.ndarray:
    """
    Generate filter-normalized random direction for a parameter tensor.
    
    From "Visualizing the Loss Landscape of Neural Nets" (Li et al. 2018).
    Generates random directions with the same norm structure as the parameters.
    
    Args:
        param: Parameter tensor
    
    Returns:
        Filter-normalized random direction
    """
    ndims = len(param.shape)
    
    if ndims == 1 or ndims == 0:
        # Don't apply random direction for scalars/biases
        return np.zeros_like(param)
    elif ndims == 2:
        # For 2D tensors (fully connected layers)
        # Normalize along input dimension (axis=0), preserving output structure
        direction = np.random.normal(size=param.shape)
        direction /= np.sqrt(np.sum(np.square(direction), axis=0, keepdims=True))
        direction *= np.sqrt(np.sum(np.square(param), axis=0, keepdims=True))
        return direction
    elif ndims == 3:
        # For 3D tensors (e.g., JAX/Flax ensemble of FC layers)
        # Shape is typically (n_ensemble, input_dim, output_dim)
        # Treat as batch of 2D weight matrices - normalize along axis=1 (input dim)
        direction = np.random.normal(size=param.shape)
        direction /= np.sqrt(np.sum(np.square(direction), axis=1, keepdims=True))
        direction *= np.sqrt(np.sum(np.square(param), axis=1, keepdims=True))
        return direction
    elif ndims == 4:
        # For 4D tensors (convolutional layers: [out_channels, in_channels, height, width])
        # Normalize over spatial and input dimensions
        direction = np.random.normal(size=param.shape)
        direction /= np.sqrt(np.sum(np.square(direction), axis=(0, 1, 2), keepdims=True))
        direction *= np.sqrt(np.sum(np.square(param), axis=(0, 1, 2), keepdims=True))
        return direction
    else:
        raise ValueError(f"Only 1, 2, 3, 4 dimensional tensors supported, got shape {param.shape}")


def filter_normalized_params(evaluator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate two orthogonal filter-normalized random directions.
    
    Args:
        evaluator: RewardSurfaceEvaluator instance with get_weights() method
    
    Returns:
        Tuple of (dir1, dir2), each a list of numpy arrays
    """
    weights = evaluator.get_weights()
    dir1 = [filter_normalize(p) for p in weights]
    dir2 = [filter_normalize(p) for p in weights]
    return dir1, dir2


def generate_plane_data(
    output_path: Path,
    dir1_vec: List[np.ndarray],
    dir2_vec: List[np.ndarray],
    magnitude: float,
    metadata: dict,
    grid_size: int = 31,
    num_episodes: int = 50,
) -> None:
    """
    Generate grid evaluation jobs for a 2D parameter plane.
    
    Args:
        output_path: Directory to save job files and metadata
        dir1_vec: First direction vector (list of numpy arrays)
        dir2_vec: Second direction vector (list of numpy arrays)
        magnitude: Scale factor for directions
        metadata: Dictionary with experiment metadata
        grid_size: Number of grid points per dimension (must be odd)
        num_episodes: Number of episodes per evaluation
    """
    assert isinstance(dir1_vec, list) and isinstance(dir1_vec[0], np.ndarray), \
        "dir1_vec must be a list of numpy arrays"
    assert isinstance(dir2_vec, list) and isinstance(dir2_vec[0], np.ndarray), \
        "dir2_vec must be a list of numpy arrays"
    assert grid_size % 2 == 1, "grid_size must be odd"
    assert magnitude > 0, "magnitude must be positive"
    
    if magnitude > 2:
        print("Warning: Large magnitude may cause unstable behavior or NaN actions.")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save direction vectors
    np.savez(output_path / "dir1.npz", *dir1_vec)
    np.savez(output_path / "dir2.npz", *dir2_vec)
    
    # Update metadata
    metadata.update({
        'experiment_type': 'plane',
        'grid_size': grid_size,
        'magnitude': magnitude,
        'num_episodes': num_episodes,
    })
    
    # Save metadata
    with open(output_path / "info.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Create results directory
    (output_path / "results").mkdir(exist_ok=True)
    
    # Generate job list
    job_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = i - grid_size // 2
            y = j - grid_size // 2
            job = (
                f"python reward_surfaces/scripts/eval_plane_job.py "
                f"{output_path} --offset1={x} --offset2={y}"
            )
            job_list.append(job)
    
    # Save jobs file
    jobs_content = "\n".join(job_list) + "\n"
    with open(output_path / "jobs.sh", 'w') as f:
        f.write(jobs_content)
    
    print(f"Generated {len(job_list)} jobs at {output_path / 'jobs.sh'}")
