#!/usr/bin/env python3
"""Create a base-height-pruned copy of quadruped MPC trajectory data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path("data/quadruped_dr/sysid_dyn20_mjlab_10k")
DEFAULT_DESTINATION = Path(
    "data/quadruped_dr/sysid_dyn20_mjlab_10k_height_pruned"
)
DEFAULT_THRESHOLD = 0.22


def base_height_samples(
    qpos: np.ndarray,
    sample_count: int | None = None,
) -> np.ndarray:
    """Return saved base-height samples from qpos shaped (nq, T) or (T, nq)."""
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2-D, got shape {qpos.shape}")
    if min(qpos.shape) < 3:
        raise ValueError(f"qpos shape {qpos.shape} does not contain qpos[2]")

    if sample_count is not None:
        if qpos.shape[1] == sample_count:
            return qpos[2]
        if qpos.shape[0] == sample_count:
            return qpos[:, 2]

    # In these files nq is small and T is large. Use the smaller axis as nq so
    # both (nq, T) and (T, nq) layouts are accepted without model metadata.
    if qpos.shape[0] <= qpos.shape[1]:
        return qpos[2]
    return qpos[:, 2]


def min_base_height(npz_path: Path) -> float:
    with np.load(npz_path, allow_pickle=False) as data:
        if "qpos" not in data:
            raise KeyError(f"{npz_path} is missing required array 'qpos'")
        qpos = np.asarray(data["qpos"])
        sample_count = (
            len(data["time"])
            if "time" in data and np.asarray(data["time"]).ndim == 1
            else None
        )

    return float(np.min(base_height_samples(qpos, sample_count=sample_count)))


def create_height_pruned_dataset(
    source: Path,
    destination: Path,
    threshold: float,
    force: bool,
) -> dict[str, object]:
    source = source.expanduser()
    destination = destination.expanduser()
    tmp_destination = destination.with_name(f"{destination.name}.tmp")

    if not source.is_dir():
        raise NotADirectoryError(f"source directory not found: {source}")

    if force:
        if destination.exists():
            shutil.rmtree(destination)
        if tmp_destination.exists():
            shutil.rmtree(tmp_destination)
    elif destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    elif tmp_destination.exists():
        raise FileExistsError(f"temporary destination already exists: {tmp_destination}")

    npz_files = sorted(path for path in source.glob("*.npz") if path.is_file())
    if not npz_files:
        raise FileNotFoundError(f"no .npz files found in {source}")

    tmp_destination.mkdir(parents=True)

    rejected_files: list[dict[str, object]] = []
    kept_count = 0

    for index, npz_path in enumerate(npz_files, start=1):
        min_height = min_base_height(npz_path)
        if min_height >= threshold:
            shutil.copy2(npz_path, tmp_destination / npz_path.name)
            kept_count += 1
        else:
            rejected_files.append(
                {
                    "file": npz_path.name,
                    "min_base_height": min_height,
                }
            )

        if index == 1 or index == len(npz_files) or index % 250 == 0:
            print(
                f"[{index}/{len(npz_files)}] kept={kept_count} "
                f"rejected={len(rejected_files)}",
                flush=True,
            )

    summary: dict[str, object] = {
        "source_path": str(source),
        "destination_path": str(destination),
        "threshold": threshold,
        "total_count": len(npz_files),
        "kept_count": kept_count,
        "rejected_count": len(rejected_files),
        "rejected_files": rejected_files,
    }

    summary_path = tmp_destination / "height_prune_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    tmp_destination.rename(destination)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy quadruped MPC .npz trajectories whose saved qpos[2] base "
            "height never falls below a threshold."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing destination or temporary destination first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = create_height_pruned_dataset(
            args.source,
            args.destination,
            args.threshold,
            args.force,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        "done: "
        f"kept={summary['kept_count']} rejected={summary['rejected_count']} "
        f"destination={summary['destination_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
