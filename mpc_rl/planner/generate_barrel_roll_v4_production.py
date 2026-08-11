"""Generate, reconcile, and atomically promote the schema-v4 production set.

The worker layout and seed budgets are intentionally fixed. This command is
refusal-to-overwrite and preserves partial worker output on every failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from mpc_rl.envs.barrel_roll_common import SCHEMA_VERSION
from mpc_rl.planner.barrel_roll_dataset import (
    aggregate_barrel_roll_dataset,
    canonical_json,
    expected_effective_config,
    sha256_bytes,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "mpc_rl/planner/gen_traj_data_barrel_roll.py"
WORK_ROOT = REPO_ROOT / "logs/go2_barrel_roll_v4/production"
WORKERS_ROOT = WORK_ROOT / "workers"
STAGING_DIRECTORY = REPO_ROOT / "data/go2_barrel_roll/v4.staging_production"
FINAL_DIRECTORY = REPO_ROOT / "data/go2_barrel_roll/v4"
ACCEPTED_PER_WORKER = 500
EXPECTED_FILE_COUNT = 2_000
EXPECTED_TRANSITION_COUNT = 250_000
GPU_SAMPLE_INTERVAL_SECONDS = 2.0

RETAINED_CHECKSUM_INDEX_HASHES = {
    "schema_v2": "153334c544fec09e8aee4fa74223bde0f5b1c6b4f0018ce1e65328fbf0ccee70",
    "schema_v3": "3c4401e353b3195e7c8801ae29257e84598f099c3bfab05b37dcb1c80c62c2cc",
}


@dataclass(frozen=True)
class WorkerSpec:
    index: int
    start_seed: int
    max_attempts: int = 5_000
    accepted_count: int = ACCEPTED_PER_WORKER

    @property
    def name(self) -> str:
        return f"worker{self.index}"

    @property
    def final_seed(self) -> int:
        return self.start_seed + self.max_attempts - 1


WORKER_SPECS = (
    WorkerSpec(index=0, start_seed=6_000_000),
    WorkerSpec(index=1, start_seed=6_100_000),
    WorkerSpec(index=2, start_seed=6_200_000),
    WorkerSpec(index=3, start_seed=6_300_000),
)

FIXED_RESERVED_SEEDS = {
    "validation": set(range(2_000_000, 2_000_100)),
    "superseded_final": set(range(3_000_000, 3_000_100)),
    "sealed_v4_final": set(range(8_000_000, 8_000_100)),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _seed_from_filename(filename: str) -> int:
    match = re.search(r"_seed_(\d+)_ep_", filename)
    if match is None:
        raise ValueError(f"cannot extract rollout seed from {filename!r}")
    return int(match.group(1))


def _checksum_index_seeds(path: Path) -> set[int]:
    seeds: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid checksum entry at {path}:{line_number}")
        seed = _seed_from_filename(parts[1])
        if seed in seeds:
            raise ValueError(f"duplicate seed {seed} in {path}")
        seeds.add(seed)
    if not seeds:
        raise ValueError(f"checksum index is empty: {path}")
    return seeds


def _manifest_seeds(path: Path) -> set[int]:
    seeds: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        seed = int(record["seed"])
        if seed in seeds:
            raise ValueError(f"duplicate seed {seed} in {path}")
        seeds.add(seed)
    return seeds


def _reserved_seed_sets() -> dict[str, set[int]]:
    reserved = dict(FIXED_RESERVED_SEEDS)
    for version in ("v2", "v3"):
        reserved[f"retained_{version}"] = _checksum_index_seeds(
            REPO_ROOT / f"data/go2_barrel_roll/{version}/checksums.sha256"
        )
    demonstration_manifests = sorted(
        (REPO_ROOT / "logs/go2_barrel_roll_v4").glob(
            "demonstrations*/generation_manifest.jsonl"
        )
    )
    if len(demonstration_manifests) != 2:
        raise ValueError(
            "expected the preserved commissioning and approved demonstration manifests"
        )
    reserved["v4_commissioning_and_demonstrations"] = set().union(
        *(_manifest_seeds(path) for path in demonstration_manifests)
    )
    return reserved


def _worker_command(spec: WorkerSpec) -> list[str]:
    worker_dir = WORKERS_ROOT / spec.name
    return [
        "/usr/bin/time",
        "-v",
        "-o",
        str(worker_dir / "resource_usage.txt"),
        sys.executable,
        str(GENERATOR),
        f"--num-trajectories={spec.accepted_count}",
        f"--start-seed={spec.start_seed}",
        f"--max-attempts={spec.max_attempts}",
        f"--output-dir={worker_dir}",
        "--manifest-filename=generation_manifest.jsonl",
        "--verbose=0",
    ]


def build_preflight_report() -> dict[str, Any]:
    """Prove fixed ranges and destinations are safe before launching workers."""
    existing = [
        str(path)
        for path in (WORK_ROOT, STAGING_DIRECTORY, FINAL_DIRECTORY)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite production paths: " + ", ".join(existing)
        )
    if not GENERATOR.is_file():
        raise FileNotFoundError(GENERATOR)
    if not Path("/usr/bin/time").is_file():
        raise FileNotFoundError("/usr/bin/time")
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi")

    retained_hashes = {
        version: sha256_file(
            REPO_ROOT / f"data/go2_barrel_roll/{version.removeprefix('schema_')}/checksums.sha256"
        )
        for version in RETAINED_CHECKSUM_INDEX_HASHES
    }
    if retained_hashes != RETAINED_CHECKSUM_INDEX_HASHES:
        raise ValueError(
            "retained checksum index changed: "
            + canonical_json({
                "expected": RETAINED_CHECKSUM_INDEX_HASHES,
                "actual": retained_hashes,
            })
        )

    reserved = _reserved_seed_sets()
    worker_sets = {
        spec.name: set(range(spec.start_seed, spec.final_seed + 1))
        for spec in WORKER_SPECS
    }
    overlap_proof: dict[str, dict[str, int]] = {}
    for name, seeds in worker_sets.items():
        overlap_proof[name] = {
            reserved_name: len(seeds & reserved_seeds)
            for reserved_name, reserved_seeds in reserved.items()
        }
        nonzero = {
            key: value for key, value in overlap_proof[name].items() if value != 0
        }
        if nonzero:
            raise ValueError(f"{name} overlaps reserved seeds: {nonzero}")
    for index, left in enumerate(WORKER_SPECS):
        for right in WORKER_SPECS[index + 1 :]:
            if worker_sets[left.name] & worker_sets[right.name]:
                raise ValueError(f"worker ranges overlap: {left.name}, {right.name}")

    data_device = (REPO_ROOT / "data").stat().st_dev
    logs_device = (REPO_ROOT / "logs").stat().st_dev
    if data_device != logs_device:
        raise ValueError("data and logs paths must share a filesystem for hard-link staging")
    disk = shutil.disk_usage(REPO_ROOT)
    effective_config_sha256 = sha256_bytes(
        canonical_json(expected_effective_config()).encode("utf-8")
    )
    return {
        "gate": "V4.3",
        "status": "predeclared",
        "created_at_utc": _utc_now(),
        "schema_version": SCHEMA_VERSION,
        "accepted_per_worker": ACCEPTED_PER_WORKER,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "expected_transition_count": EXPECTED_TRANSITION_COUNT,
        "effective_config_sha256": effective_config_sha256,
        "work_root": str(WORK_ROOT),
        "staging_directory": str(STAGING_DIRECTORY),
        "final_directory": str(FINAL_DIRECTORY),
        "worker_specs": [
            {
                **asdict(spec),
                "name": spec.name,
                "final_seed": spec.final_seed,
                "command": _worker_command(spec),
            }
            for spec in WORKER_SPECS
        ],
        "range_overlap_counts": overlap_proof,
        "reserved_seed_evidence": {
            name: {
                "count": len(seeds),
                "minimum": min(seeds),
                "maximum": max(seeds),
            }
            for name, seeds in reserved.items()
        },
        "retained_checksum_indexes": retained_hashes,
        "filesystem_device": data_device,
        "disk_free_bytes_before": disk.free,
    }


def _sample_resources() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gpus = []
    for line in completed.stdout.splitlines():
        index, name, used, total = (part.strip() for part in line.split(",", 3))
        gpus.append({
            "index": int(index),
            "name": name,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
        })
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            meminfo[name] = int(value.strip().split()[0])
    return {
        "timestamp_utc": _utc_now(),
        "gpus": gpus,
        "host_memory_total_kib": meminfo["MemTotal"],
        "host_memory_available_kib": meminfo["MemAvailable"],
    }


def _parse_time_report(path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if ": " in line:
            key, value = line.rsplit(": ", 1)
            values[key] = value
    return {
        "elapsed_wall_clock": values.get("Elapsed (wall clock) time (h:mm:ss or m:ss)"),
        "maximum_resident_set_size_kib": int(
            values["Maximum resident set size (kbytes)"]
        ),
        "swaps": int(values["Swaps"]),
        "exit_status": int(values["Exit status"]),
    }


def _terminate_workers(processes: dict[str, subprocess.Popen[bytes]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_workers() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processes: dict[str, subprocess.Popen[bytes]] = {}
    streams: list[TextIO] = []
    samples: list[dict[str, Any]] = []
    gpu_log_path = WORK_ROOT / "resource_samples.jsonl"
    start = time.monotonic()
    try:
        for spec in WORKER_SPECS:
            worker_dir = WORKERS_ROOT / spec.name
            worker_dir.mkdir(parents=True, exist_ok=False)
            stdout = (worker_dir / "stdout.log").open("x", encoding="utf-8")
            stderr = (worker_dir / "stderr.log").open("x", encoding="utf-8")
            streams.extend((stdout, stderr))
            process = subprocess.Popen(
                _worker_command(spec),
                cwd=REPO_ROOT,
                stdout=stdout,
                stderr=stderr,
            )
            processes[spec.name] = process
            print(f"launched {spec.name} pid={process.pid}", flush=True)

        with gpu_log_path.open("x", encoding="utf-8") as resource_log:
            while any(process.poll() is None for process in processes.values()):
                sample = _sample_resources()
                sample["worker_processes"] = {
                    name: {
                        "pid": process.pid,
                        "returncode": process.poll(),
                    }
                    for name, process in processes.items()
                }
                samples.append(sample)
                resource_log.write(canonical_json(sample) + "\n")
                resource_log.flush()
                time.sleep(GPU_SAMPLE_INTERVAL_SECONDS)
            final_sample = _sample_resources()
            final_sample["worker_processes"] = {
                name: {"pid": process.pid, "returncode": process.poll()}
                for name, process in processes.items()
            }
            samples.append(final_sample)
            resource_log.write(canonical_json(final_sample) + "\n")
            resource_log.flush()
    except BaseException:
        _terminate_workers(processes)
        raise
    finally:
        for stream in streams:
            stream.close()

    process_reports = []
    failures = []
    for spec in WORKER_SPECS:
        process = processes[spec.name]
        returncode = process.wait()
        resource_path = WORKERS_ROOT / spec.name / "resource_usage.txt"
        report = {
            "worker": spec.name,
            "pid": process.pid,
            "returncode": returncode,
            "resource_usage": (
                _parse_time_report(resource_path) if resource_path.is_file() else None
            ),
        }
        process_reports.append(report)
        if returncode != 0:
            failures.append(report)
    if failures:
        raise RuntimeError("production worker failures: " + canonical_json(failures))

    gpu_used = [gpu["memory_used_mib"] for sample in samples for gpu in sample["gpus"]]
    host_available = [sample["host_memory_available_kib"] for sample in samples]
    resources = {
        "wall_seconds": time.monotonic() - start,
        "sample_count": len(samples),
        "gpu_memory_used_mib_max": max(gpu_used),
        "host_memory_available_kib_min": min(host_available),
        "process_failures": [],
    }
    return process_reports, resources


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        records.append(record)
    return records


def _reconcile_workers() -> dict[str, Any]:
    all_files: list[Path] = []
    all_accepted_seeds: list[int] = []
    worker_reports = []
    for spec in WORKER_SPECS:
        worker_dir = WORKERS_ROOT / spec.name
        manifest_path = worker_dir / "generation_manifest.jsonl"
        records = _read_manifest(manifest_path)
        if len(records) > spec.max_attempts:
            raise ValueError(f"{spec.name} exceeded its maximum attempt count")
        expected_attempt_seeds = list(range(spec.start_seed, spec.start_seed + len(records)))
        actual_attempt_seeds = [int(record["seed"]) for record in records]
        if actual_attempt_seeds != expected_attempt_seeds:
            raise ValueError(f"{spec.name} manifest seeds are not the declared sequence")
        accepted_records = [record for record in records if record.get("accepted")]
        if len(accepted_records) != spec.accepted_count:
            raise ValueError(
                f"{spec.name} accepted {len(accepted_records)}/{spec.accepted_count}"
            )
        files = sorted(worker_dir.glob("go2_barrel_roll_v4_*.npz"))
        accepted_names = [str(record.get("trajectory_file")) for record in accepted_records]
        if len(files) != spec.accepted_count:
            raise ValueError(f"{spec.name} has {len(files)} production archives")
        if {path.name for path in files} != set(accepted_names):
            raise ValueError(f"{spec.name} manifest/archive mismatch")
        accepted_seeds = [int(record["seed"]) for record in accepted_records]
        if any(seed < spec.start_seed or seed > spec.final_seed for seed in accepted_seeds):
            raise ValueError(f"{spec.name} accepted a seed outside its declared range")
        all_files.extend(files)
        all_accepted_seeds.extend(accepted_seeds)
        worker_reports.append({
            "worker": spec.name,
            "attempt_count": len(records),
            "accepted_count": len(accepted_records),
            "rejected_count": len(records) - len(accepted_records),
            "first_attempt_seed": actual_attempt_seeds[0],
            "last_attempt_seed": actual_attempt_seeds[-1],
            "manifest_sha256": sha256_file(manifest_path),
        })

    if len(all_files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"expected {EXPECTED_FILE_COUNT} archives, found {len(all_files)}")
    if len(set(all_accepted_seeds)) != EXPECTED_FILE_COUNT:
        raise ValueError("production accepted seeds are not unique")
    if len({path.name for path in all_files}) != EXPECTED_FILE_COUNT:
        raise ValueError("production archive names are not unique")

    STAGING_DIRECTORY.mkdir(parents=False, exist_ok=False)
    for path in all_files:
        os.link(path, STAGING_DIRECTORY / path.name)
    for spec in WORKER_SPECS:
        shutil.copy2(
            WORKERS_ROOT / spec.name / "generation_manifest.jsonl",
            STAGING_DIRECTORY / f"generation_manifest_{spec.name}.jsonl",
        )
    reconciliation = {
        "status": "reconciled",
        "file_count": len(all_files),
        "unique_seed_count": len(set(all_accepted_seeds)),
        "workers": worker_reports,
    }
    _write_json_exclusive(
        STAGING_DIRECTORY / "worker_reconciliation.json", reconciliation
    )
    return reconciliation


def _verify_checksum_index(directory: Path) -> dict[str, Any]:
    checksum_path = directory / "checksums.sha256"
    entries = []
    seen_names = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid checksum entry {checksum_path}:{line_number}")
        expected, filename = parts
        if Path(filename).name != filename or filename in seen_names:
            raise ValueError(f"unsafe or duplicate checksum filename {filename!r}")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {filename}")
        seen_names.add(filename)
        entries.append(filename)
    if len(entries) != EXPECTED_FILE_COUNT:
        raise ValueError(
            f"checksum index contains {len(entries)}/{EXPECTED_FILE_COUNT} files"
        )
    return {
        "verified_count": len(entries),
        "checksum_index_sha256": sha256_file(checksum_path),
    }


def _retained_checksum_hashes() -> dict[str, str]:
    return {
        version: sha256_file(
            REPO_ROOT / f"data/go2_barrel_roll/{version.removeprefix('schema_')}/checksums.sha256"
        )
        for version in RETAINED_CHECKSUM_INDEX_HASHES
    }


def execute_production() -> dict[str, Any]:
    preflight = build_preflight_report()
    WORKERS_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(WORK_ROOT / "production_plan.json", preflight)
    production_start = time.monotonic()
    process_reports, resources = _run_workers()
    reconciliation = _reconcile_workers()
    summary = aggregate_barrel_roll_dataset(STAGING_DIRECTORY)
    expected_config_hash = preflight["effective_config_sha256"]
    expected_summary = {
        "schema_version": SCHEMA_VERSION,
        "file_count": EXPECTED_FILE_COUNT,
        "transition_count": EXPECTED_TRANSITION_COUNT,
        "accepted_count": EXPECTED_FILE_COUNT,
        "effective_config_sha256": expected_config_hash,
    }
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in expected_summary.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise ValueError("production summary mismatch: " + canonical_json(mismatches))
    checksum_verification = _verify_checksum_index(STAGING_DIRECTORY)
    retained_after = _retained_checksum_hashes()
    if retained_after != RETAINED_CHECKSUM_INDEX_HASHES:
        raise ValueError(
            "retained checksum indexes changed during production: "
            + canonical_json(retained_after)
        )

    report = {
        "gate": "V4.3_production_data",
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "wall_seconds_total": time.monotonic() - production_start,
        "preflight": preflight,
        "worker_processes": process_reports,
        "resources": resources,
        "reconciliation": reconciliation,
        "dataset_summary": summary,
        "dataset_summary_sha256": sha256_file(
            STAGING_DIRECTORY / "dataset_summary.json"
        ),
        "checksum_verification": checksum_verification,
        "retained_checksum_indexes_after": retained_after,
        "disk_free_bytes_after": shutil.disk_usage(REPO_ROOT).free,
        "atomic_promotion": {
            "source": str(STAGING_DIRECTORY),
            "destination": str(FINAL_DIRECTORY),
            "same_filesystem": True,
        },
        "process_failures": [],
    }
    _write_json_exclusive(STAGING_DIRECTORY / "production_report.json", report)
    _write_json_exclusive(WORK_ROOT / "production_report.json", report)
    _write_json_exclusive(
        STAGING_DIRECTORY / "COMPLETE",
        {
            "status": "complete",
            "file_count": EXPECTED_FILE_COUNT,
            "transition_count": EXPECTED_TRANSITION_COUNT,
            "dataset_summary_sha256": report["dataset_summary_sha256"],
        },
    )
    STAGING_DIRECTORY.replace(FINAL_DIRECTORY)
    _write_json_exclusive(
        WORK_ROOT / "PROMOTED.json",
        {
            "promoted_at_utc": _utc_now(),
            "destination": str(FINAL_DIRECTORY),
            "production_report_sha256": sha256_file(
                FINAL_DIRECTORY / "production_report.json"
            ),
        },
    )
    print(
        "PRODUCTION_COMPLETE "
        + canonical_json({
            "file_count": summary["file_count"],
            "transition_count": summary["transition_count"],
            "destination": str(FINAL_DIRECTORY),
        }),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and print the locked launch plan without writing or launching.",
    )
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(build_preflight_report(), indent=2, sort_keys=True))
        return
    try:
        execute_production()
    except BaseException as error:
        if WORK_ROOT.is_dir() and not (WORK_ROOT / "failure.json").exists():
            _write_json_exclusive(
                WORK_ROOT / "failure.json",
                {
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
