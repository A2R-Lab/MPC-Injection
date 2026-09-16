from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mpc_rl.evaluate_go2_barrel_roll_g9 as evaluation_module
from mpc_rl.evaluate_go2_barrel_roll_g9 import (
    EXPECTED_VALIDATION_STEPS,
    choose_official_policy,
    validate_selected_checkpoint,
)
from mpc_rl.train import (
    BARREL_ROLL_FINAL_TEST_SEEDS,
    BARREL_ROLL_VALIDATION_SEEDS,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_choose_official_policy_uses_locked_tiebreak_order():
    results = [
        {
            "training_seed": 1,
            "success_count": 80,
            "validation_success_rate": 0.8,
            "validation_mean_final_hold_standing_score": 0.4,
        },
        {
            "training_seed": 2,
            "success_count": 81,
            "validation_success_rate": 0.1,
            "validation_mean_final_hold_standing_score": 0.3,
        },
    ]
    assert choose_official_policy(results)["training_seed"] == 2

    results[0]["success_count"] = 81
    results[0]["validation_success_rate"] = 0.9
    assert choose_official_policy(results)["training_seed"] == 1

    results[1]["validation_success_rate"] = 0.9
    results[0]["validation_mean_final_hold_standing_score"] = 0.5
    results[1]["validation_mean_final_hold_standing_score"] = 0.4
    assert choose_official_policy(results)["training_seed"] == 1

    results[1]["validation_mean_final_hold_standing_score"] = 0.6
    assert choose_official_policy(results)["training_seed"] == 2

    for result in results:
        result["success_count"] = 79
    assert choose_official_policy(results) is None


def test_validate_selected_checkpoint_requires_earliest_full_validation_winner(
    tmp_path,
):
    run_dir = tmp_path / "runs" / "seed1" / "one-run"
    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True)
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
        "percentage": 25,
        "random_select": True,
        "quadruped_mpc_replay_mode": "direct",
        "use_go2_sysid": True,
        "save_replay_buffer_checkpoints": False,
        "save_replay_buffer_final": False,
        "domain_randomization": {"enabled": False},
        "barrel_roll": {
            "run_provenance": {
                "runtime_source_sha256": evaluation_module._barrel_roll_runtime_source_hashes(
                    Path(evaluation_module.__file__).resolve().parents[1]
                )
            },
            "evaluation": {
                "steps": list(EXPECTED_VALIDATION_STEPS),
                "seeds": list(BARREL_ROLL_VALIDATION_SEEDS),
                "final_test_seeds": list(BARREL_ROLL_FINAL_TEST_SEEDS),
                "checkpoint_selection_order": [
                    "strict_success_rate_desc",
                    "mean_final_hold_standing_score_desc",
                    "timesteps_asc",
                ],
            }
        },
    }
    _write_json(run_dir / "config.json", config)
    episodes = [
        {"seed": seed, "success": False}
        for seed in BARREL_ROLL_VALIDATION_SEEDS
    ]
    records = []
    for step in EXPECTED_VALIDATION_STEPS:
        rate = 0.5 if step in (20_000, 30_000) else 0.1
        records.append(
            {
                "timesteps": step,
                "success_rate": rate,
                "mean_final_hold_standing_score": (
                    0.6 if step == 30_000 else 0.5
                ),
                "seeds": list(BARREL_ROLL_VALIDATION_SEEDS),
                "episodes": episodes,
                "failure_reasons": {},
                "final_hold_streak_distribution": {},
                "standing_subscore_distributions": {},
                "motion_distributions": {},
                "reward_component_distributions": {},
                "roll_progress_distributions": {},
                "contact_summary": {},
                "return_distribution": {},
            }
        )
    (run_dir / "barrel_roll_eval_history.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_json(
        best_dir / "selection.json",
        {
            "selection_order": [
                "strict_success_rate_desc",
                "mean_final_hold_standing_score_desc",
                "timesteps_asc",
            ],
            "success_rate": 0.5,
            "mean_final_hold_standing_score": 0.6,
            "timesteps": 30_000,
        },
    )
    (best_dir / "best_model.zip").write_bytes(b"model")
    (best_dir / "vec_normalize.pkl").write_bytes(b"stats")
    _write_json(
        run_dir / "COMPLETE",
        {
            "status": "complete",
            "training_seed": 1,
            "timesteps": 500_000,
            "selected_timesteps": 30_000,
            "selected_success_rate": 0.5,
            "selected_mean_final_hold_standing_score": 0.6,
            "artifact_hashes": {
                "config": evaluation_module.sha256_file(run_dir / "config.json"),
                "selected_model": evaluation_module.sha256_file(
                    best_dir / "best_model.zip"
                ),
                "selected_vecnormalize": evaluation_module.sha256_file(
                    best_dir / "vec_normalize.pkl"
                ),
                "selection": evaluation_module.sha256_file(
                    best_dir / "selection.json"
                ),
                "evaluation_history": evaluation_module.sha256_file(
                    run_dir / "barrel_roll_eval_history.jsonl"
                ),
            },
        },
    )

    selected = validate_selected_checkpoint(tmp_path, 1)

    assert selected["training_seed"] == 1
    assert selected["selected_timesteps"] == 30_000
    assert selected["validation_success_rate"] == pytest.approx(0.5)
    assert selected["validation_mean_final_hold_standing_score"] == pytest.approx(0.6)

    selection_path = best_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["timesteps"] = 20_000
    _write_json(selection_path, selection)
    with pytest.raises(ValueError, match="locked validation order"):
        validate_selected_checkpoint(tmp_path, 1)


def test_validation_review_reruns_selected_policies_without_opening_final(
    monkeypatch, tmp_path
):
    (tmp_path / "COMPLETE").touch()
    _write_json(
        tmp_path / "campaign_validation.json",
        {
            "mode": "production",
            "source_hashes": {
                "evaluation": evaluation_module.sha256_file(
                    Path(evaluation_module.__file__).resolve()
                )
            },
        },
    )
    evaluations = {}
    selections = {}
    for training_seed in (1, 2):
        episodes = [
            {
                "seed": seed,
                "success": index < 12,
                "failure_reason": None if index < 12 else "rotation_out_of_band",
            }
            for index, seed in enumerate(BARREL_ROLL_VALIDATION_SEEDS)
        ]
        evaluation = {
            "seeds": list(BARREL_ROLL_VALIDATION_SEEDS),
            "episodes": episodes,
            "success_rate": 0.12,
            "mean_final_hold_standing_score": 0.4,
        }
        evaluations[training_seed] = evaluation
        selections[training_seed] = {
            "training_seed": training_seed,
            "model": str(tmp_path / f"seed{training_seed}_model.zip"),
            "vecnormalize": str(tmp_path / f"seed{training_seed}_stats.pkl"),
            "selected_evaluation": {**evaluation, "timesteps": 20_000},
        }
    monkeypatch.setattr(
        evaluation_module,
        "validate_selected_checkpoint",
        lambda _campaign, seed: selections[seed],
    )

    class _Env:
        def close(self):
            pass

    def _load(**kwargs):
        seed = int(Path(kwargs["model_path"]).name.split("seed")[1].split("_")[0])
        return {"training_seed": seed}, _Env()

    monkeypatch.setattr(evaluation_module, "load_saved_model_for_video_eval", _load)
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_barrel_roll_policy",
        lambda model, _env, seeds: evaluations[model["training_seed"]],
    )
    rendered = []

    def _render(**kwargs):
        assert set(kwargs["seeds"]).isdisjoint(BARREL_ROLL_FINAL_TEST_SEEDS)
        rendered.append(kwargs)
        return {
            "seeds": list(kwargs["seeds"]),
            "outcomes": [kwargs["labels"][seed] == "success" for seed in kwargs["seeds"]],
            "videos": [],
        }

    monkeypatch.setattr(evaluation_module, "_render", _render)

    report = evaluation_module.run_validation_review(tmp_path)

    assert report["status"] == "awaiting_user_visual_approval"
    assert report["final_test_opened"] is False
    assert len(rendered) == 2
    assert all(len(policy["selected_success_video_seeds"]) == 10 for policy in report["policies"])
    assert all(
        policy["representative_failure_video_seeds"]
        == {"rotation_out_of_band": 2_000_012}
        for policy in report["policies"]
    )
    assert (tmp_path / "validation_review/report.json").is_file()
    approval = evaluation_module.record_user_video_approval(
        tmp_path,
        gate="validation",
        user_statement="I approve both selected-policy validation video sets.",
    )
    assert approval["approved"] is True
    assert approval["final_test_opened_at_approval"] is False
    with pytest.raises(FileExistsError):
        evaluation_module.record_user_video_approval(
            tmp_path,
            gate="validation",
            user_statement="duplicate",
        )


def test_final_evaluation_refuses_without_validation_video_approval(tmp_path):
    (tmp_path / "COMPLETE").touch()
    _write_json(
        tmp_path / "campaign_validation.json",
        {
            "mode": "production",
            "total_timesteps_per_seed": 500_000,
            "runs": [{"seed": 1}, {"seed": 2}],
            "source_hashes": {
                "evaluation": evaluation_module.sha256_file(
                    Path(evaluation_module.__file__).resolve()
                )
            },
        },
    )
    _write_json(
        tmp_path / "validation_review/report.json",
        {"status": "awaiting_user_visual_approval"},
    )

    with pytest.raises(PermissionError, match="explicit validation-video approval"):
        evaluation_module.run_final_evaluation(tmp_path)

    assert not (tmp_path / "final_evaluation").exists()


def test_video_validation_uses_recorded_episode_length(monkeypatch, tmp_path):
    class _Env:
        closed = False

        def close(self):
            self.closed = True

    env = _Env()
    monkeypatch.setattr(
        evaluation_module,
        "load_saved_model_for_video_eval",
        lambda **kwargs: (object(), env),
    )

    def _record(**kwargs):
        kwargs["video_dir"].mkdir(parents=True)
        for seed in kwargs["barrel_roll_seeds"]:
            label = kwargs["barrel_roll_video_labels"][seed]
            (kwargs["video_dir"] / f"{label}_seed{seed}.mp4").write_bytes(b"video")
        return {"episode_successes": [True, False]}

    monkeypatch.setattr(evaluation_module, "evaluate_and_record", _record)

    def _probe(command, **kwargs):
        del kwargs
        seed = int(Path(command[-1]).stem.split("seed")[-1])
        frames = 125 if seed == 2_000_000 else 17
        return SimpleNamespace(
            stdout=json.dumps({
                "streams": [{
                    "codec_name": "h264",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "50/1",
                    "nb_read_frames": str(frames),
                    "duration": str(frames / 50.0),
                }]
            })
        )

    monkeypatch.setattr(evaluation_module.subprocess, "run", _probe)
    selected = {
        "model": str(tmp_path / "model.zip"),
        "vecnormalize": str(tmp_path / "stats.pkl"),
        "evaluation": {
            "episodes": [
                {"seed": 2_000_000, "success": True, "control_steps": 125},
                {"seed": 2_000_001, "success": False, "control_steps": 17},
            ]
        },
    }

    report = evaluation_module._render(
        selected=selected,
        seeds=(2_000_000, 2_000_001),
        video_dir=tmp_path / "videos",
        labels={2_000_000: "success", 2_000_001: "non_foot_ground_contact"},
    )

    assert [video["technical"]["frame_count"] for video in report["videos"]] == [
        17,
        125,
    ]
    assert env.closed


def test_final_evaluation_locks_scores_once_and_renders_official(
    monkeypatch, tmp_path
):
    (tmp_path / "COMPLETE").touch()
    _write_json(
        tmp_path / "campaign_validation.json",
        {
            "mode": "production",
            "total_timesteps_per_seed": 500_000,
            "runs": [{"seed": 1}, {"seed": 2}],
            "source_hashes": {
                "evaluation": evaluation_module.sha256_file(
                    Path(evaluation_module.__file__).resolve()
                )
            },
        },
    )
    review_path = tmp_path / "validation_review/report.json"
    _write_json(review_path, {"status": "awaiting_user_visual_approval"})
    _write_json(
        tmp_path / "validation_review/USER_APPROVED.json",
        {
            "approved": True,
            "report_sha256": evaluation_module.sha256_file(review_path),
        },
    )
    selections = {}
    for seed in (1, 2):
        model_path = tmp_path / f"seed{seed}_best_model.zip"
        stats_path = tmp_path / f"seed{seed}_vec_normalize.pkl"
        model_path.write_bytes(f"model{seed}".encode())
        stats_path.write_bytes(f"stats{seed}".encode())
        selections[seed] = {
            "training_seed": seed,
            "model": str(model_path),
            "model_sha256": "a" * 64,
            "vecnormalize": str(stats_path),
            "vecnormalize_sha256": "b" * 64,
            "selected_timesteps": 20_000,
            "validation_success_rate": 0.8 if seed == 1 else 0.7,
            "validation_mean_final_hold_standing_score": 0.6,
        }
    monkeypatch.setattr(
        evaluation_module,
        "validate_selected_checkpoint",
        lambda _campaign, seed: selections[seed].copy(),
    )

    class _Env:
        def close(self):
            pass

    loaded_seed = None
    scored_calls = []

    def _load(**kwargs):
        nonlocal loaded_seed
        loaded_seed = int(Path(kwargs["model_path"]).name.split("seed")[1].split("_")[0])
        return {"training_seed": loaded_seed}, _Env()

    def _score(model, _env, seeds, *, include_critic=False):
        assert include_critic is True
        scored_calls.append(model["training_seed"])
        success_count = 80 if model["training_seed"] == 1 else 70
        episodes = [
            {
                "seed": seed,
                "success": index < success_count,
                "failure_reason": None if index < success_count else "fallen_tilt",
                "terminal_roll_progress": 6.2,
                "terminal_base_height": 0.2,
                "terminal_body_up_tilt": 0.1,
                "terminal_base_linear_speed": 0.2,
                "terminal_base_angular_speed": 0.3,
                "terminal_joint_velocity_norm": 0.4,
                "terminal_action_change_norm": 0.5,
                "reward_components": {
                    "roll_tracking": 100.0,
                    "rate_tracking": 20.0,
                    "action_change": -1.0,
                    "terminal_outcome": 50.0 if index < success_count else -50.0,
                },
            }
            for index, seed in enumerate(seeds)
        ]
        return {
            "seeds": list(seeds),
            "episodes": episodes,
            "success_rate": success_count / 100.0,
            "failure_reasons": {"fallen_tilt": 100 - success_count},
            "final_hold_streak_distribution": {},
            "final_hold_condition_failure_endpoints": {},
            "standing_subscore_distributions": {},
            "motion_distributions": {},
            "roll_progress_distributions": {},
            "reward_component_distributions": {},
            "contact_summary": {},
            "return_distribution": {},
            "critic_values": [{"min": 1.0, "max": 2.0, "mean": 1.5}],
        }

    def _record(**kwargs):
        video_dir = kwargs["video_dir"]
        video_dir.mkdir(parents=True)
        outcomes = []
        for index, seed in enumerate(kwargs["barrel_roll_seeds"]):
            (video_dir / f"rollout{index}_seed{seed}.mp4").write_bytes(b"video")
            outcomes.append(True)
        return {"episode_successes": outcomes}

    rendered = []

    def _render(**kwargs):
        rendered.append(kwargs)
        return {
            "seeds": list(kwargs["seeds"]),
            "outcomes": [
                bool(next(
                    episode["success"]
                    for episode in kwargs["selected"]["evaluation"]["episodes"]
                    if episode["seed"] == seed
                ))
                for seed in kwargs["seeds"]
            ],
            "videos": [
                {"path": f"seed{seed}.mp4", "sha256": "c" * 64}
                for seed in kwargs["seeds"]
            ],
        }

    monkeypatch.setattr(evaluation_module, "load_saved_model_for_video_eval", _load)
    monkeypatch.setattr(evaluation_module, "evaluate_barrel_roll_policy", _score)
    monkeypatch.setattr(evaluation_module, "evaluate_and_record", _record)
    monkeypatch.setattr(evaluation_module, "_render", _render)

    summary = evaluation_module.run_final_evaluation(tmp_path)

    assert scored_calls == [1, 2]
    assert summary["numerical_pass"] is True
    assert summary["accepted"] is False
    assert summary["status"] == "awaiting_user_final_video_approval"
    assert summary["official_training_seed"] == 1
    assert len(summary["video_reports"]["official_successes"]["videos"]) == 10
    assert len(rendered) == 3
    assert (tmp_path / "final_evaluation" / "selection_lock.json").is_file()
    assert (tmp_path / "final_evaluation" / "seed1_final_test.json").is_file()
    assert (tmp_path / "final_evaluation" / "seed2_final_test.json").is_file()
    accepted = evaluation_module.record_user_video_approval(
        tmp_path,
        gate="final",
        user_statement="I approve the final success videos.",
    )
    assert accepted["status"] == "accepted"
    assert accepted["strict_final_metric_passed"] is True
    assert accepted["user_final_video_review_passed"] is True
    with pytest.raises(FileExistsError, match="refusing to rerun"):
        evaluation_module.run_final_evaluation(tmp_path)
