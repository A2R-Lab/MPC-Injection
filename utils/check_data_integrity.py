#!/usr/bin/env python3
"""
Utility script to check for corrupted .npz files in trajectory data directories.
This helps identify problematic files before running experiments.
"""

import numpy as np
import zipfile
from pathlib import Path
import sys


def check_npz_file(file_path):
    """
    Check if an .npz file can be loaded successfully.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        data = np.load(file_path)
        # Try to access all arrays to ensure they're not corrupted
        for key in data.files:
            _ = data[key]
        data.close()
        return True, None
    except (zipfile.BadZipFile, EOFError, IOError, ValueError) as e:
        return False, str(e)


def check_directory(data_dir, verbose=True):
    """
    Check all .npz files in a directory for corruption.
    
    Args:
        data_dir: Path to directory containing .npz files
        verbose: If True, print progress for each file
        
    Returns:
        tuple: (num_valid, num_corrupted, corrupted_files)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}")
        return 0, 0, []
    
    npz_files = list(data_path.glob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return 0, 0, []
    
    print(f"Checking {len(npz_files)} files in {data_dir}...")
    print()
    
    num_valid = 0
    num_corrupted = 0
    corrupted_files = []
    
    for i, file_path in enumerate(npz_files, 1):
        is_valid, error = check_npz_file(file_path)
        
        if is_valid:
            num_valid += 1
            if verbose:
                print(f"[{i}/{len(npz_files)}] ✓ {file_path.name}")
        else:
            num_corrupted += 1
            corrupted_files.append((file_path, error))
            print(f"[{i}/{len(npz_files)}] ✗ {file_path.name}")
            print(f"    Error: {error}")
    
    return num_valid, num_corrupted, corrupted_files


def main():
    """Main function to check data integrity."""
    
    # ========== CONFIGURATION ==========
    # Add the data directories you want to check:
    data_dirs = [
        'data/cartpole_0_001dt/',  # Old cartpole data at 0.001s timestep
        'data/walker_0_0025dt/',
        # Add more directories as needed
    ]
    # ===================================
    
    print("=" * 70)
    print("Data Integrity Checker")
    print("=" * 70)
    print()
    
    all_corrupted = []
    total_valid = 0
    total_corrupted = 0
    
    for data_dir in data_dirs:
        num_valid, num_corrupted, corrupted_files = check_directory(data_dir, verbose=False)
        total_valid += num_valid
        total_corrupted += num_corrupted
        all_corrupted.extend(corrupted_files)
        
        print()
        print(f"Results for {data_dir}:")
        print(f"  Valid files:     {num_valid}")
        print(f"  Corrupted files: {num_corrupted}")
        print()
    
    print("=" * 70)
    print(f"Total valid files:     {total_valid}")
    print(f"Total corrupted files: {total_corrupted}")
    print("=" * 70)
    
    if all_corrupted:
        print()
        print("Corrupted files found:")
        for file_path, error in all_corrupted:
            print(f"  {file_path}")
            print(f"    Error: {error}")
        print()
        print("Recommendation: Delete corrupted files and regenerate them if needed.")
        sys.exit(1)
    else:
        print()
        print("✓ All files are valid!")
        sys.exit(0)


if __name__ == '__main__':
    main()
