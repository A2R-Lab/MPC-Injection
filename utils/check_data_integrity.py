#!/usr/bin/env python3
"""Validate NPZ archives, with strict task-aware barrel-roll checks."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np


def _looks_like_barrel_roll(file_path: Path) -> bool:
    if file_path.name.startswith("go2_barrel_roll_"):
        return True
    try:
        with np.load(file_path, allow_pickle=False) as data:
            return (
                "task_id" in data.files
                and np.asarray(data["task_id"]).shape == ()
                and str(data["task_id"]) == "go2_barrel_roll"
            )
    except (zipfile.BadZipFile, EOFError, OSError, ValueError):
        return False


def _check_npz_file_details(file_path):
    file_path = Path(file_path)
    if _looks_like_barrel_roll(file_path):
        from mpc_rl.planner.barrel_roll_dataset import validate_barrel_roll_file

        report = validate_barrel_roll_file(file_path)
        return (
            report.valid,
            None if report.valid else "; ".join(report.errors),
            report.metrics,
        )
    try:
        with np.load(file_path, allow_pickle=False) as data:
            for key in data.files:
                _ = data[key]
        return True, None, {}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError) as error:
        return False, str(error), {}


def check_npz_file(file_path):
    """Return ``(valid, error)`` after loading every array without pickle."""
    valid, error, _ = _check_npz_file_details(file_path)
    return valid, error


def check_directory(data_dir, verbose=True):
    """Check every top-level NPZ file while preserving the legacy return tuple."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}")
        return 0, 0, []
    npz_files = sorted(data_path.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return 0, 0, []

    print(f"Checking {len(npz_files)} files in {data_dir}...")
    num_valid = 0
    corrupted_files = []
    for index, file_path in enumerate(npz_files, 1):
        valid, error, metrics = _check_npz_file_details(file_path)
        if valid:
            num_valid += 1
            if verbose:
                suffix = ""
                if metrics:
                    suffix = (
                        f" action_clip={metrics['action_clip_fraction']:.6f}"
                        f" mpx_sat={metrics['mpx_torque_saturation_fraction']:.6f}"
                        f" applied_sat={metrics['torque_saturation_fraction']:.6f}"
                        " direct_torque_lpf_replay_parity=false"
                    )
                print(f"[{index}/{len(npz_files)}] VALID {file_path.name}{suffix}")
        else:
            corrupted_files.append((file_path, error))
            print(f"[{index}/{len(npz_files)}] INVALID {file_path.name}")
            print(f"    Error: {error}")
    return num_valid, len(corrupted_files), corrupted_files


def _paths_from_args(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    return [Path("data/cartpole_0_001dt"), Path("data/walker_0_0025dt")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--aggregate-barrel-roll",
        action="store_true",
        help="write aggregate manifest, checksum index, and summary after validation",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    total_valid = 0
    total_corrupted = 0
    missing_paths = 0
    directories_for_aggregation: list[Path] = []
    for path in _paths_from_args(args.paths):
        if not path.exists():
            print(f"Error: Path not found: {path}")
            missing_paths += 1
            continue
        if path.is_dir():
            valid, corrupted, failures = check_directory(path, verbose=not args.quiet)
            directories_for_aggregation.append(path)
        else:
            is_valid, error, metrics = _check_npz_file_details(path)
            valid = int(is_valid)
            corrupted = int(not is_valid)
            failures = [] if is_valid else [(path, error)]
            if not args.quiet or not is_valid:
                suffix = ""
                if metrics:
                    suffix = (
                        f" action_clip={metrics['action_clip_fraction']:.6f}"
                        f" mpx_sat={metrics['mpx_torque_saturation_fraction']:.6f}"
                        f" applied_sat={metrics['torque_saturation_fraction']:.6f}"
                        " direct_torque_lpf_replay_parity=false"
                    )
                print(f"{'VALID' if is_valid else 'INVALID'} {path}{suffix}")
                if error:
                    print(f"    Error: {error}")
        total_valid += valid
        total_corrupted += corrupted
        for file_path, error in failures:
            if args.quiet:
                print(f"INVALID {file_path}: {error}")

    if args.aggregate_barrel_roll and not total_corrupted and not missing_paths:
        from mpc_rl.planner.barrel_roll_dataset import aggregate_barrel_roll_dataset

        for directory in directories_for_aggregation:
            summary = aggregate_barrel_roll_dataset(directory)
            print(
                f"Aggregated {summary['file_count']} barrel-roll files in {directory}; "
                f"checksum index: {summary['checksum_index']}"
            )

    print(f"Total valid files: {total_valid}")
    print(f"Total corrupted files: {total_corrupted}")
    raise SystemExit(1 if total_corrupted or missing_paths else 0)


if __name__ == "__main__":
    main()
