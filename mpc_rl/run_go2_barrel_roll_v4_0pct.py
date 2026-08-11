#!/usr/bin/env python3
"""Train one audited schema-v4 Go2 barrel-roll policy with 0% MPC replay.

The pending v4 campaign records ``mpc_rl/train.py`` by content hash.  This
launcher leaves that source unchanged and adapts only its two 25%-specific
guards.  Every other frozen v4 training, data, checkpoint, and evaluation
constraint is delegated to the existing implementation.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpc_rl import train as training  # noqa: E402


BASELINE_MODE = "schema_v4_0pct_baseline"
LAUNCHER_PATH = Path(__file__).resolve()
LAUNCHER_RELATIVE_PATH = str(LAUNCHER_PATH.relative_to(REPO_ROOT))

_ORIGINAL_VALIDATE_OPTIONS = training.validate_barrel_roll_training_options
_ORIGINAL_RUN_PROVENANCE = training.barrel_roll_run_provenance


def validate_zero_percent_training_options(**options: Any) -> None:
    """Require 0%, then apply every frozen v4 production option check."""
    if options.get("percentage") != 0:
        raise ValueError("schema-v4 0% baseline requires percentage=0")
    production_equivalent = dict(options)
    production_equivalent["percentage"] = 25
    _ORIGINAL_VALIDATE_OPTIONS(**production_equivalent)


def zero_percent_run_provenance(data_dir: str) -> dict[str, Any]:
    """Add this launcher to the normal schema-v4 data/source provenance."""
    provenance = _ORIGINAL_RUN_PROVENANCE(data_dir)
    provenance["baseline_launcher"] = {
        "mode": BASELINE_MODE,
        "path": LAUNCHER_RELATIVE_PATH,
        "sha256": training.sha256_file(LAUNCHER_PATH),
    }
    return provenance


def _read_evaluation_history(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid evaluation history at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(
                f"invalid evaluation history object at line {line_number}"
            )
        records.append(record)
    return records


def _require_zero_percent_diagnostics(diagnostics: dict[str, Any]) -> None:
    if diagnostics.get("status") != "complete":
        raise RuntimeError("barrel-roll training diagnostics are not complete")
    replay_range = diagnostics.get("ranges", {}).get(
        "replay_buffer/mpc_percentage"
    )
    replay_last = diagnostics.get("replay_percentage_last")
    try:
        replay_low, replay_high = (float(value) for value in replay_range)
        replay_last_value = float(replay_last)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "0% baseline diagnostics omitted MPC replay percentages"
        ) from error
    if (replay_low, replay_high, replay_last_value) != (0.0, 0.0, 0.0):
        raise RuntimeError(
            "0% baseline observed injected MPC transitions: "
            + training.canonical_json({
                "range": replay_range,
                "last": replay_last,
            })
        )


def write_zero_percent_training_completion(
    logdir: Path | str, model: Any
) -> dict[str, Any]:
    """Validate one exact 0% v4 run and create its write-once marker."""
    logdir = Path(logdir)
    complete_path = logdir / "COMPLETE"
    if complete_path.exists():
        raise FileExistsError(
            f"refusing to overwrite barrel-roll completion marker: {complete_path}"
        )
    if int(model.num_timesteps) != 500_000:
        raise RuntimeError(
            f"barrel-roll run ended at {model.num_timesteps}/500000 timesteps"
        )

    required_paths = {
        "config": logdir / "config.json",
        "final_model": logdir / "final_model.zip",
        "final_vecnormalize": logdir / "vec_normalize.pkl",
        "selected_model": logdir / "best_model/best_model.zip",
        "selected_vecnormalize": logdir / "best_model/vec_normalize.pkl",
        "selection": logdir / "best_model/selection.json",
        "evaluation_history": logdir / "barrel_roll_eval_history.jsonl",
        "diagnostics": logdir / "barrel_roll_pilot_diagnostics.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "barrel-roll run is missing required artifacts: " + ", ".join(missing)
        )
    replay_artifacts = sorted(logdir.rglob("*replay_buffer*"))
    if replay_artifacts:
        raise RuntimeError(
            "barrel-roll run unexpectedly saved replay buffers: "
            + ", ".join(str(path) for path in replay_artifacts)
        )

    config = json.loads(required_paths["config"].read_text(encoding="utf-8"))
    expected_config_values = {
        "algorithm": "SAC-MPC",
        "env_name": "quadruped-barrel_roll",
        "total_timesteps": 500_000,
        "num_envs": 4,
        "max_episode_steps": training.BARREL_ROLL_CONTROL_STEPS,
        "eval_freq": 10_000,
        "checkpoint_freq": 25_000,
        "inject_type": "percentage",
        "percentage": 0,
        "random_select": True,
        "quadruped_mpc_replay_mode": "direct",
        "use_go2_sysid": True,
        "save_replay_buffer_checkpoints": False,
        "save_replay_buffer_final": False,
        "enable_logging": True,
        "learning_rate": 3.0e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "gradient_steps": -1,
        "policy_delay": 2,
    }
    config_mismatches = {
        name: {"expected": expected, "actual": config.get(name)}
        for name, expected in expected_config_values.items()
        if config.get(name) != expected
    }
    if config.get("seed") not in (1, 2):
        config_mismatches["seed"] = {
            "expected": [1, 2],
            "actual": config.get("seed"),
        }
    domain_randomization = config.get("domain_randomization", {})
    if (
        not isinstance(domain_randomization, dict)
        or domain_randomization.get("enabled") is not False
        or domain_randomization.get("config_type") != "disabled"
    ):
        config_mismatches["domain_randomization"] = {
            "expected": {"enabled": False, "config_type": "disabled"},
            "actual": domain_randomization,
        }

    barrel_roll = config.get("barrel_roll", {})
    dataset = barrel_roll.get("dataset", {}) if isinstance(barrel_roll, dict) else {}
    expected_dataset_values = {
        "schema_version": training.BARREL_ROLL_SCHEMA_VERSION,
        "target_mpc_percentage": 0,
        "replay_mode": "direct",
    }
    for name, expected in expected_dataset_values.items():
        if dataset.get(name) != expected:
            config_mismatches[f"barrel_roll.dataset.{name}"] = {
                "expected": expected,
                "actual": dataset.get(name),
            }
    if config_mismatches:
        raise RuntimeError(
            "barrel-roll completed with incompatible config: "
            + training.canonical_json(config_mismatches)
        )

    run_provenance = barrel_roll.get("run_provenance", {})
    recorded_source_hashes = run_provenance.get("runtime_source_sha256")
    current_source_hashes = training._barrel_roll_runtime_source_hashes(REPO_ROOT)
    if recorded_source_hashes != current_source_hashes:
        raise RuntimeError(
            "barrel-roll runtime source changed during training: "
            + training.canonical_json({
                "recorded": recorded_source_hashes,
                "current": current_source_hashes,
            })
        )
    expected_launcher = {
        "mode": BASELINE_MODE,
        "path": LAUNCHER_RELATIVE_PATH,
        "sha256": training.sha256_file(LAUNCHER_PATH),
    }
    if run_provenance.get("baseline_launcher") != expected_launcher:
        raise RuntimeError(
            "0% baseline launcher provenance changed during training: "
            + training.canonical_json({
                "recorded": run_provenance.get("baseline_launcher"),
                "current": expected_launcher,
            })
        )

    records = _read_evaluation_history(required_paths["evaluation_history"])
    steps = tuple(int(record.get("timesteps", -1)) for record in records)
    if steps != training.BARREL_ROLL_VALIDATION_STEPS:
        raise RuntimeError(
            "barrel-roll evaluation history does not cover exact 0:10000:500000 steps"
        )
    required_diagnostics = (
        "failure_reasons",
        "final_hold_streak_distribution",
        "standing_subscore_distributions",
        "motion_distributions",
        "reward_component_distributions",
        "roll_progress_distributions",
        "contact_summary",
        "return_distribution",
    )
    for record in records:
        if record.get("seeds") != list(training.BARREL_ROLL_VALIDATION_SEEDS):
            raise RuntimeError("barrel-roll evaluation history used wrong seeds")
        if len(record.get("episodes", [])) != len(
            training.BARREL_ROLL_VALIDATION_SEEDS
        ):
            raise RuntimeError("barrel-roll evaluation history omitted episodes")
        missing_diagnostics = [
            name for name in required_diagnostics if record.get(name) is None
        ]
        if missing_diagnostics:
            raise RuntimeError(
                "barrel-roll evaluation history omitted diagnostics: "
                + ", ".join(missing_diagnostics)
            )
        for name in ("success_rate", "mean_final_hold_standing_score"):
            value = float(record[name])
            if not np.isfinite(value):
                raise RuntimeError(f"barrel-roll evaluation has non-finite {name}")

    winner = max(
        records,
        key=lambda record: (
            float(record["success_rate"]),
            float(record["mean_final_hold_standing_score"]),
            -int(record["timesteps"]),
        ),
    )
    selection = json.loads(required_paths["selection"].read_text(encoding="utf-8"))
    expected_selection = {
        "selection_order": [
            "strict_success_rate_desc",
            "mean_final_hold_standing_score_desc",
            "timesteps_asc",
        ],
        "success_rate": winner["success_rate"],
        "mean_final_hold_standing_score": winner[
            "mean_final_hold_standing_score"
        ],
        "timesteps": winner["timesteps"],
    }
    if selection != expected_selection:
        raise RuntimeError(
            "barrel-roll selected checkpoint does not match locked order: "
            + training.canonical_json({
                "expected": expected_selection,
                "actual": selection,
            })
        )

    checkpoint_steps = tuple(range(25_000, 500_001, 25_000))
    checkpoint_models = [
        logdir / f"checkpoints/model_{step}_steps.zip" for step in checkpoint_steps
    ]
    checkpoint_stats = [
        logdir / f"checkpoints/model_vecnormalize_{step}_steps.pkl"
        for step in checkpoint_steps
    ]
    missing_checkpoints = [
        str(path)
        for path in (*checkpoint_models, *checkpoint_stats)
        if not path.is_file()
    ]
    if missing_checkpoints:
        raise RuntimeError(
            "barrel-roll run is missing exact checkpoints: "
            + ", ".join(missing_checkpoints)
        )
    tensorboard_events = sorted(
        path
        for path in (logdir / "tensorboard").rglob("events.out.tfevents.*")
        if path.is_file() and path.stat().st_size > 0
    )
    if not tensorboard_events:
        raise RuntimeError("barrel-roll run has no nonempty TensorBoard event file")
    diagnostics = json.loads(
        required_paths["diagnostics"].read_text(encoding="utf-8")
    )
    _require_zero_percent_diagnostics(diagnostics)

    artifact_hashes = {
        name: training.sha256_file(path) for name, path in required_paths.items()
    }
    artifact_hashes.update({
        "checkpoint_models_index": training.sha256_bytes(
            training.canonical_json([
                {
                    "path": str(path.relative_to(logdir)),
                    "sha256": training.sha256_file(path),
                }
                for path in checkpoint_models
            ]).encode("utf-8")
        ),
        "checkpoint_vecnormalize_index": training.sha256_bytes(
            training.canonical_json([
                {
                    "path": str(path.relative_to(logdir)),
                    "sha256": training.sha256_file(path),
                }
                for path in checkpoint_stats
            ]).encode("utf-8")
        ),
        "tensorboard_event_index": training.sha256_bytes(
            training.canonical_json([
                {
                    "path": str(path.relative_to(logdir)),
                    "sha256": training.sha256_file(path),
                }
                for path in tensorboard_events
            ]).encode("utf-8")
        ),
    })
    marker = {
        "status": "complete",
        "mode": BASELINE_MODE,
        "target_mpc_percentage": 0,
        "completed_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "training_seed": int(config["seed"]),
        "timesteps": int(model.num_timesteps),
        "validation_steps": list(training.BARREL_ROLL_VALIDATION_STEPS),
        "selected_timesteps": int(selection["timesteps"]),
        "selected_success_rate": float(selection["success_rate"]),
        "selected_mean_final_hold_standing_score": float(
            selection["mean_final_hold_standing_score"]
        ),
        "artifact_hashes": artifact_hashes,
        "replay_percentage_range": [0.0, 0.0],
        "replay_buffers_saved": False,
        "baseline_launcher": expected_launcher,
    }
    with complete_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(marker, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return marker


def install_zero_percent_adapter() -> None:
    """Install the narrow adapter before handing control to ``train.main``."""
    training.validate_barrel_roll_training_options = (
        validate_zero_percent_training_options
    )
    training.barrel_roll_run_provenance = zero_percent_run_provenance
    training.write_barrel_roll_training_completion = (
        write_zero_percent_training_completion
    )


def main() -> None:
    install_zero_percent_adapter()
    print(
        "Schema-v4 0% baseline route enabled; MPC replay must remain exactly 0%.",
        flush=True,
    )
    training.app.run(training.main)


if __name__ == "__main__":
    main()
