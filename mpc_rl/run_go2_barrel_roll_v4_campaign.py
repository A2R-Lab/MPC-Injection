"""Run the two fixed schema-v4 production jobs and build Gate V4.5 review."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from mpc_rl.planner.barrel_roll_dataset import canonical_json, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPO_ROOT / "data/go2_barrel_roll/v4"
CAMPAIGN_DIRECTORY = REPO_ROOT / "logs/go2_barrel_roll_v4/campaign"
TRAIN_SCRIPT = REPO_ROOT / "mpc_rl/train.py"
EVALUATION_SCRIPT = REPO_ROOT / "mpc_rl/evaluate_go2_barrel_roll_g9.py"
TRAINING_SEEDS = (1, 2)
RESOURCE_SAMPLE_INTERVAL_SECONDS = 5.0
RETAINED_CHECKSUM_INDEX_HASHES = {
    "schema_v2": "153334c544fec09e8aee4fa74223bde0f5b1c6b4f0018ce1e65328fbf0ccee70",
    "schema_v3": "3c4401e353b3195e7c8801ae29257e84598f099c3bfab05b37dcb1c80c62c2cc",
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


def _training_command(seed: int) -> list[str]:
    seed_root = CAMPAIGN_DIRECTORY / "runs" / f"seed{seed}"
    return [
        "/usr/bin/time",
        "-v",
        "-o",
        str(CAMPAIGN_DIRECTORY / f"seed{seed}_resource_usage.txt"),
        sys.executable,
        str(TRAIN_SCRIPT),
        "--env_name=quadruped-barrel_roll",
        "--algorithm=SAC-MPC",
        "--robot=go2",
        "--use_go2_sysid=true",
        "--seed=" + str(seed),
        "--total_timesteps=500000",
        "--num_envs=4",
        "--max_episode_steps=125",
        "--logdir=" + str(seed_root),
        "--suffix=v4-production-seed" + str(seed),
        "--enable_logging=true",
        "--learning_rate=0.0003",
        "--buffer_size=1000000",
        "--learning_starts=10000",
        "--batch_size=256",
        "--tau=0.005",
        "--gamma=0.99",
        "--gradient_steps=-1",
        "--policy_delay=2",
        "--inject_type=percentage",
        "--percentage=25",
        "--random_select=true",
        "--data_dir=" + str(DATASET_DIRECTORY),
        "--quadruped_mpc_replay_mode=direct",
        "--checkpoint_freq=25000",
        "--eval_freq=10000",
        "--save_replay_buffer_checkpoints=false",
        "--save_replay_buffer_final=false",
        "--domain_rand=false",
        "--domain_rand_config_type=disabled",
    ]


def _source_hashes() -> dict[str, str]:
    return {
        "campaign": sha256_file(Path(__file__).resolve()),
        "train": sha256_file(TRAIN_SCRIPT),
        "evaluation": sha256_file(EVALUATION_SCRIPT),
    }


def build_preflight_report() -> dict[str, Any]:
    if CAMPAIGN_DIRECTORY.exists():
        raise FileExistsError(
            f"refusing to overwrite campaign directory: {CAMPAIGN_DIRECTORY}"
        )
    for path in (TRAIN_SCRIPT, EVALUATION_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    for executable in ("/usr/bin/time",):
        if not Path(executable).is_file():
            raise FileNotFoundError(executable)
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi")

    required_dataset = {
        "summary": DATASET_DIRECTORY / "dataset_summary.json",
        "checksums": DATASET_DIRECTORY / "checksums.sha256",
        "manifest": DATASET_DIRECTORY / "generation_manifest_aggregate.jsonl",
        "complete": DATASET_DIRECTORY / "COMPLETE",
        "production_report": DATASET_DIRECTORY / "production_report.json",
    }
    missing = [str(path) for path in required_dataset.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "production dataset is incomplete: " + ", ".join(missing)
        )
    summary = json.loads(
        required_dataset["summary"].read_text(encoding="utf-8")
    )
    expected_summary = {
        "schema_version": 4,
        "file_count": 2_000,
        "transition_count": 250_000,
        "effective_config_sha256": (
            "34e0644dfbc2b42ef800ed00e12d9fc4d24ee82825aaddbf7a970c6792953a66"
        ),
        "checksum_index_sha256": sha256_file(required_dataset["checksums"]),
        "aggregate_manifest_sha256": sha256_file(required_dataset["manifest"]),
    }
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in expected_summary.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise ValueError("production dataset mismatch: " + canonical_json(mismatches))
    complete = json.loads(
        required_dataset["complete"].read_text(encoding="utf-8")
    )
    if (
        complete.get("status") != "complete"
        or complete.get("file_count") != 2_000
        or complete.get("transition_count") != 250_000
        or complete.get("dataset_summary_sha256")
        != sha256_file(required_dataset["summary"])
    ):
        raise ValueError("production dataset COMPLETE marker mismatch")
    retained_hashes = {
        version: sha256_file(
            REPO_ROOT
            / f"data/go2_barrel_roll/{version.removeprefix('schema_')}/checksums.sha256"
        )
        for version in RETAINED_CHECKSUM_INDEX_HASHES
    }
    if retained_hashes != RETAINED_CHECKSUM_INDEX_HASHES:
        raise ValueError("retained schema-v2/v3 checksum index changed")
    return {
        "gate": "V4.4",
        "status": "predeclared",
        "created_at_utc": _utc_now(),
        "campaign_directory": str(CAMPAIGN_DIRECTORY),
        "training_seeds": list(TRAINING_SEEDS),
        "total_timesteps_per_seed": 500_000,
        "concurrent": True,
        "commands": {
            str(seed): _training_command(seed) for seed in TRAINING_SEEDS
        },
        "dataset": {
            "path": str(DATASET_DIRECTORY),
            "summary_sha256": sha256_file(required_dataset["summary"]),
            "checksum_index_sha256": sha256_file(required_dataset["checksums"]),
            "aggregate_manifest_sha256": sha256_file(required_dataset["manifest"]),
            "complete_marker_sha256": sha256_file(required_dataset["complete"]),
            "production_report_sha256": sha256_file(
                required_dataset["production_report"]
            ),
        },
        "retained_checksum_indexes": retained_hashes,
        "source_hashes": _source_hashes(),
        "disk_free_bytes_before": shutil.disk_usage(REPO_ROOT).free,
    }


def _sample_resources() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gpus = []
    for line in completed.stdout.splitlines():
        index, name, used, total, utilization = (
            part.strip() for part in line.split(",", 4)
        )
        gpus.append({
            "index": int(index),
            "name": name,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": int(utilization),
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
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if ": " in line:
            key, value = line.rsplit(": ", 1)
            values[key] = value
    return {
        "elapsed_wall_clock": values.get(
            "Elapsed (wall clock) time (h:mm:ss or m:ss)"
        ),
        "maximum_resident_set_size_kib": int(
            values["Maximum resident set size (kbytes)"]
        ),
        "swaps": int(values["Swaps"]),
        "exit_status": int(values["Exit status"]),
    }


def _terminate(processes: dict[int, subprocess.Popen[bytes]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes.values():
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_training_jobs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processes: dict[int, subprocess.Popen[bytes]] = {}
    streams: list[TextIO] = []
    samples = []
    start = time.monotonic()
    try:
        for seed in TRAINING_SEEDS:
            seed_root = CAMPAIGN_DIRECTORY / "runs" / f"seed{seed}"
            seed_root.mkdir(parents=True, exist_ok=False)
            stdout = (CAMPAIGN_DIRECTORY / f"seed{seed}_stdout.log").open(
                "x", encoding="utf-8"
            )
            stderr = (CAMPAIGN_DIRECTORY / f"seed{seed}_stderr.log").open(
                "x", encoding="utf-8"
            )
            streams.extend((stdout, stderr))
            process = subprocess.Popen(
                _training_command(seed),
                cwd=REPO_ROOT,
                stdout=stdout,
                stderr=stderr,
            )
            processes[seed] = process
            print(f"launched training seed={seed} pid={process.pid}", flush=True)
        with (CAMPAIGN_DIRECTORY / "resource_samples.jsonl").open(
            "x", encoding="utf-8"
        ) as resource_log:
            while any(process.poll() is None for process in processes.values()):
                sample = _sample_resources()
                sample["processes"] = {
                    str(seed): {
                        "pid": process.pid,
                        "returncode": process.poll(),
                    }
                    for seed, process in processes.items()
                }
                samples.append(sample)
                resource_log.write(canonical_json(sample) + "\n")
                resource_log.flush()
                failed = {
                    seed: process.returncode
                    for seed, process in processes.items()
                    if process.poll() not in (None, 0)
                }
                if failed:
                    raise RuntimeError(
                        "training process failed; stopping peer: "
                        + canonical_json(failed)
                    )
                time.sleep(RESOURCE_SAMPLE_INTERVAL_SECONDS)
            sample = _sample_resources()
            samples.append(sample)
            resource_log.write(canonical_json(sample) + "\n")
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for stream in streams:
            stream.close()

    process_reports = []
    failures = []
    for seed, process in processes.items():
        returncode = process.wait()
        report = {
            "seed": seed,
            "pid": process.pid,
            "returncode": returncode,
            "resource_usage": _parse_time_report(
                CAMPAIGN_DIRECTORY / f"seed{seed}_resource_usage.txt"
            ),
        }
        process_reports.append(report)
        if returncode != 0:
            failures.append(report)
    if failures:
        raise RuntimeError("training process failure: " + canonical_json(failures))
    return process_reports, {
        "wall_seconds": time.monotonic() - start,
        "sample_count": len(samples),
        "gpu_memory_used_mib_max": max(
            gpu["memory_used_mib"] for sample in samples for gpu in sample["gpus"]
        ),
        "host_memory_available_kib_min": min(
            sample["host_memory_available_kib"] for sample in samples
        ),
        "process_failures": [],
    }


def _single_run_dir(seed: int) -> Path:
    configs = sorted(
        (CAMPAIGN_DIRECTORY / "runs" / f"seed{seed}").glob("*/config.json")
    )
    if len(configs) != 1:
        raise RuntimeError(f"seed {seed} produced {len(configs)} run configs")
    run_dir = configs[0].parent
    marker_path = run_dir / "COMPLETE"
    if not marker_path.is_file():
        raise RuntimeError(f"seed {seed} did not produce COMPLETE")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or marker.get("training_seed") != seed
        or marker.get("timesteps") != 500_000
    ):
        raise RuntimeError(f"seed {seed} completion marker mismatch")
    return run_dir


def execute_campaign() -> dict[str, Any]:
    preflight = build_preflight_report()
    CAMPAIGN_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(CAMPAIGN_DIRECTORY / "campaign_plan.json", preflight)
    process_reports, resources = _run_training_jobs()
    current_source_hashes = _source_hashes()
    if current_source_hashes != preflight["source_hashes"]:
        raise RuntimeError(
            "campaign source changed during training: "
            + canonical_json({
                "preflight": preflight["source_hashes"],
                "current": current_source_hashes,
            })
        )
    run_records = []
    for seed in TRAINING_SEEDS:
        run_dir = _single_run_dir(seed)
        complete_path = run_dir / "COMPLETE"
        run_records.append({
            "seed": seed,
            "run_dir": str(run_dir.resolve()),
            "config_sha256": sha256_file(run_dir / "config.json"),
            "complete_sha256": sha256_file(complete_path),
            "completion": json.loads(complete_path.read_text(encoding="utf-8")),
        })
    validation = {
        "mode": "production",
        "status": "training_complete",
        "completed_at_utc": _utc_now(),
        "total_timesteps_per_seed": 500_000,
        "concurrent": True,
        "dataset": preflight["dataset"],
        "source_hashes": preflight["source_hashes"],
        "runs": run_records,
        "processes": process_reports,
        "resources": resources,
        "process_failures": [],
    }
    validation_path = CAMPAIGN_DIRECTORY / "campaign_validation.json"
    _write_json_exclusive(validation_path, validation)
    _write_json_exclusive(
        CAMPAIGN_DIRECTORY / "COMPLETE",
        {
            "status": "training_complete",
            "campaign_validation_sha256": sha256_file(validation_path),
            "training_seeds": list(TRAINING_SEEDS),
            "timesteps_per_seed": 500_000,
        },
    )

    review_stdout = CAMPAIGN_DIRECTORY / "validation_review_stdout.log"
    review_stderr = CAMPAIGN_DIRECTORY / "validation_review_stderr.log"
    review_resource = CAMPAIGN_DIRECTORY / "validation_review_resource_usage.txt"
    review_command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(review_resource),
        sys.executable,
        str(EVALUATION_SCRIPT),
        "--mode=validation-review",
        "--campaign-dir=" + str(CAMPAIGN_DIRECTORY),
    ]
    with review_stdout.open("x", encoding="utf-8") as stdout, review_stderr.open(
        "x", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            review_command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"validation review failed with exit code {completed.returncode}"
        )
    review_path = CAMPAIGN_DIRECTORY / "validation_review/report.json"
    if not review_path.is_file():
        raise RuntimeError("validation review did not write report.json")
    report = {
        "gate": "V4.4_and_V4.5",
        "status": "awaiting_user_validation_video_approval",
        "campaign_validation_sha256": sha256_file(validation_path),
        "validation_review_sha256": sha256_file(review_path),
        "validation_review_resource_usage": _parse_time_report(review_resource),
        "final_test_opened": False,
    }
    _write_json_exclusive(
        CAMPAIGN_DIRECTORY / "campaign_execution_report.json", report
    )
    print("VALIDATION_REVIEW_READY " + canonical_json(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and print the fixed campaign without launching it.",
    )
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(build_preflight_report(), indent=2, sort_keys=True))
        return
    try:
        execute_campaign()
    except BaseException as error:
        if CAMPAIGN_DIRECTORY.is_dir() and not (
            CAMPAIGN_DIRECTORY / "failure.json"
        ).exists():
            _write_json_exclusive(
                CAMPAIGN_DIRECTORY / "failure.json",
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
