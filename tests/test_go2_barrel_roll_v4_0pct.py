from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mpc_rl import train as training
from mpc_rl.evaluate_go2_barrel_roll_v4_0pct import (
    DEFAULT_VIDEO_SEEDS,
    parse_video_seeds,
    validate_zero_percent_run,
)
from mpc_rl import run_go2_barrel_roll_v4_0pct as baseline


def _valid_options() -> dict:
    return {
        "robot": "go2",
        "algorithm": "SAC-MPC",
        "inject_type": "percentage",
        "percentage": 0,
        "replay_mode": "direct",
        "domain_rand_enabled": False,
        "domain_rand_config_type": "disabled",
        "use_go2_sysid": True,
        "data_dir": "data/go2_barrel_roll/v4",
    }


def _write_completed_baseline_inputs(run_dir: Path) -> None:
    launcher = {
        "mode": baseline.BASELINE_MODE,
        "path": baseline.LAUNCHER_RELATIVE_PATH,
        "sha256": training.sha256_file(baseline.LAUNCHER_PATH),
    }
    config = {
        "algorithm": "SAC-MPC",
        "env_name": "quadruped-barrel_roll",
        "seed": 1,
        "total_timesteps": 500_000,
        "num_envs": 4,
        "max_episode_steps": 125,
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
        "domain_randomization": {
            "enabled": False,
            "config_type": "disabled",
        },
        "barrel_roll": {
            "dataset": {
                "schema_version": 4,
                "target_mpc_percentage": 0,
                "replay_mode": "direct",
            },
            "run_provenance": {
                "runtime_source_sha256": (
                    training._barrel_roll_runtime_source_hashes(
                        Path(training.__file__).resolve().parents[1]
                    )
                ),
                "baseline_launcher": launcher,
            },
        },
    }
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for relative in (
        "final_model.zip",
        "vec_normalize.pkl",
        "best_model/best_model.zip",
        "best_model/vec_normalize.pkl",
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())

    records = []
    for step in training.BARREL_ROLL_VALIDATION_STEPS:
        success_rate = 0.4 if step in (20_000, 30_000) else 0.1
        standing_score = 0.5 if step == 30_000 else 0.3
        records.append({
            "timesteps": step,
            "success_rate": success_rate,
            "mean_final_hold_standing_score": standing_score,
            "seeds": list(training.BARREL_ROLL_VALIDATION_SEEDS),
            "episodes": [{} for _ in training.BARREL_ROLL_VALIDATION_SEEDS],
            "failure_reasons": {},
            "final_hold_streak_distribution": {},
            "standing_subscore_distributions": {},
            "motion_distributions": {},
            "reward_component_distributions": {},
            "roll_progress_distributions": {},
            "contact_summary": {},
            "return_distribution": {},
        })
    (run_dir / "barrel_roll_eval_history.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    selection = {
        "selection_order": [
            "strict_success_rate_desc",
            "mean_final_hold_standing_score_desc",
            "timesteps_asc",
        ],
        "success_rate": 0.4,
        "mean_final_hold_standing_score": 0.5,
        "timesteps": 30_000,
    }
    (run_dir / "best_model/selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    (run_dir / "barrel_roll_pilot_diagnostics.json").write_text(
        json.dumps({
            "status": "complete",
            "ranges": {"replay_buffer/mpc_percentage": [0.0, 0.0]},
            "replay_percentage_last": 0.0,
        }),
        encoding="utf-8",
    )
    for step in range(25_000, 500_001, 25_000):
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / f"model_{step}_steps.zip").write_bytes(b"model")
        (checkpoint_dir / f"model_vecnormalize_{step}_steps.pkl").write_bytes(
            b"stats"
        )
    event_path = run_dir / "tensorboard/SAC_1/events.out.tfevents.test"
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(b"events")


def test_zero_percent_adapter_accepts_only_locked_baseline_options():
    baseline.validate_zero_percent_training_options(**_valid_options())

    options = _valid_options()
    options["percentage"] = 25
    with pytest.raises(ValueError, match="requires percentage=0"):
        baseline.validate_zero_percent_training_options(**options)

    options = _valid_options()
    options["num_envs"] = 8
    with pytest.raises(ValueError, match="num_envs"):
        baseline.validate_zero_percent_training_options(**options)


def test_zero_percent_completion_and_video_validation(tmp_path):
    run_dir = tmp_path / "run"
    _write_completed_baseline_inputs(run_dir)

    marker = baseline.write_zero_percent_training_completion(
        run_dir, SimpleNamespace(num_timesteps=500_000)
    )
    validated = validate_zero_percent_run(run_dir)

    assert marker["mode"] == baseline.BASELINE_MODE
    assert marker["target_mpc_percentage"] == 0
    assert marker["selected_timesteps"] == 30_000
    assert marker["replay_percentage_range"] == [0.0, 0.0]
    assert validated["selection"]["timesteps"] == 30_000


def test_zero_percent_completion_rejects_any_injected_replay(tmp_path):
    run_dir = tmp_path / "run"
    _write_completed_baseline_inputs(run_dir)
    diagnostics_path = run_dir / "barrel_roll_pilot_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["ranges"]["replay_buffer/mpc_percentage"] = [0.0, 0.01]
    diagnostics["replay_percentage_last"] = 0.01
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    with pytest.raises(RuntimeError, match="observed injected MPC transitions"):
        baseline.write_zero_percent_training_completion(
            run_dir, SimpleNamespace(num_timesteps=500_000)
        )
    assert not (run_dir / "COMPLETE").exists()


def test_video_seed_parser_requires_unique_integers():
    assert parse_video_seeds("7000000, 7000001") == (7_000_000, 7_000_001)
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        parse_video_seeds("7,7")


def test_default_video_seeds_do_not_open_v4_scoring_ranges():
    assert DEFAULT_VIDEO_SEEDS == tuple(range(7_000_000, 7_000_010))
    assert set(DEFAULT_VIDEO_SEEDS).isdisjoint(
        training.BARREL_ROLL_VALIDATION_SEEDS
    )
    assert set(DEFAULT_VIDEO_SEEDS).isdisjoint(
        training.BARREL_ROLL_FINAL_TEST_SEEDS
    )
