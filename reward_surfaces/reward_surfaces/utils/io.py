"""I/O utilities for loading and saving data."""

import numpy as np
from pathlib import Path
from typing import List


def readz(fname: str) -> List[np.ndarray]:
    """
    Read numpy arrays from .npz file.
    
    Args:
        fname: Path to .npz file
    
    Returns:
        List of numpy arrays
    """
    outvecs = []
    with np.load(fname) as data:
        for item in data:
            outvecs.append(data[item])
    return outvecs


def savez(fname: str, *arrays: np.ndarray) -> None:
    """
    Save numpy arrays to .npz file.
    
    Args:
        fname: Path to .npz file
        *arrays: Variable number of numpy arrays to save
    """
    np.savez(fname, *arrays)
