from __future__ import annotations

import json

import pytest

from mpc_rl.evaluate_go2_barrel_roll_0pct import _validate_config


def _write_baseline_run(run_dir, *, percentage: int = 0) -> None:
    config = {
        "algorithm": "SAC-MPC",
        "env_name": "quadruped-barrel_roll",
        "percentage": percentage,
        "seed": 1,
        "total_timesteps": 500_000,
        "num_envs": 4,
        "checkpoint_freq": 20_000,
        "eval_freq": 10_000,
        "use_go2_sysid": True,
        "save_replay_buffer_checkpoints": False,
        "save_replay_buffer_final": False,
        "domain_randomization": {"enabled": False},
    }
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    for path in (
        checkpoint_dir / "model_220000_steps.zip",
        checkpoint_dir / "model_vecnormalize_220000_steps.pkl",
        run_dir / "final_model.zip",
        run_dir / "vec_normalize.pkl",
    ):
        path.touch()


def test_validate_zero_percent_baseline_config(tmp_path):
    run_dir = tmp_path / "run"
    _write_baseline_run(run_dir)
    config = _validate_config(run_dir, 220_000)
    assert config["percentage"] == 0


def test_validate_zero_percent_baseline_rejects_injected_policy(tmp_path):
    run_dir = tmp_path / "run"
    _write_baseline_run(run_dir, percentage=25)
    with pytest.raises(ValueError, match="percentage"):
        _validate_config(run_dir, 220_000)
