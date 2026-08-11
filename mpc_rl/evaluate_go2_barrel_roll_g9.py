"""Review and finally score the two frozen schema-v4 barrel-roll policies.

The 100-seed final-test results are intentionally write-once.  If this program
fails after creating ``final_evaluation/selection_lock.json``, preserve the
directory and diagnose the failure instead of rerunning the final-test seeds.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mpc_rl.envs.barrel_roll_common import CONTROL_STEPS, EPISODE_HORIZON
from mpc_rl.planner.barrel_roll_dataset import canonical_json, sha256_file
from mpc_rl.train import (
    BARREL_ROLL_FINAL_TEST_SEEDS,
    BARREL_ROLL_VALIDATION_SEEDS,
    BARREL_ROLL_VALIDATION_STEPS,
    _barrel_roll_runtime_source_hashes,
    evaluate_and_record,
    evaluate_barrel_roll_policy,
    load_saved_model_for_video_eval,
)


TRAINING_SEEDS = (1, 2)
EXPECTED_VALIDATION_STEPS = BARREL_ROLL_VALIDATION_STEPS
VALIDATION_VIDEO_SELECTION_SEED = 4_500_000
FINAL_VIDEO_SELECTION_SEED = 4_600_000


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()


def _single_run_dir(campaign_dir: Path, training_seed: int) -> Path:
    seed_root = campaign_dir / "runs" / f"seed{training_seed}"
    configs = sorted(seed_root.glob("*/config.json"))
    if len(configs) != 1:
        raise ValueError(
            f"expected exactly one seed-{training_seed} run config under {seed_root}; "
            f"found {len(configs)}"
        )
    return configs[0].parent


def _read_source_locked_campaign_validation(path: Path) -> dict[str, Any]:
    validation = _read_json(path)
    expected_evaluation_hash = validation.get("source_hashes", {}).get(
        "evaluation"
    )
    actual_evaluation_hash = sha256_file(Path(__file__).resolve())
    if expected_evaluation_hash != actual_evaluation_hash:
        raise ValueError(
            "evaluation source differs from the campaign lock: "
            + canonical_json({
                "expected": expected_evaluation_hash,
                "actual": actual_evaluation_hash,
            })
        )
    return validation


def validate_selected_checkpoint(
    campaign_dir: Path | str, training_seed: int
) -> dict[str, Any]:
    """Prove one checkpoint won the locked three-part validation order."""
    campaign_dir = Path(campaign_dir)
    run_dir = _single_run_dir(campaign_dir, training_seed)
    config_path = run_dir / "config.json"
    history_path = run_dir / "barrel_roll_eval_history.jsonl"
    best_dir = run_dir / "best_model"
    selection_path = best_dir / "selection.json"
    model_path = best_dir / "best_model.zip"
    vecnormalize_path = best_dir / "vec_normalize.pkl"
    complete_path = run_dir / "COMPLETE"
    required = (
        config_path,
        history_path,
        selection_path,
        model_path,
        vecnormalize_path,
        complete_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing selected-checkpoint artifacts: " + ", ".join(missing))

    config = _read_json(config_path)
    expected_options = {
        "algorithm": "SAC-MPC",
        "env_name": "quadruped-barrel_roll",
        "seed": training_seed,
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
    }
    option_mismatches = {
        name: {"expected": expected, "actual": config.get(name)}
        for name, expected in expected_options.items()
        if config.get(name) != expected
    }
    barrel = config.get("barrel_roll")
    if not isinstance(barrel, dict):
        option_mismatches["barrel_roll"] = {"expected": "object", "actual": barrel}
    else:
        evaluation = barrel.get("evaluation", {})
        expected_evaluation_fields = {
            "steps": list(BARREL_ROLL_VALIDATION_STEPS),
            "seeds": list(BARREL_ROLL_VALIDATION_SEEDS),
            "final_test_seeds": list(BARREL_ROLL_FINAL_TEST_SEEDS),
        }
        for name, expected in expected_evaluation_fields.items():
            actual = evaluation.get(name) if isinstance(evaluation, dict) else None
            if actual != expected:
                option_mismatches[f"barrel_roll.evaluation.{name}"] = {
                    "expected": expected,
                    "actual": actual,
                }
        expected_selection_order = [
            "strict_success_rate_desc",
            "mean_final_hold_standing_score_desc",
            "timesteps_asc",
        ]
        actual_order = (
            evaluation.get("checkpoint_selection_order")
            if isinstance(evaluation, dict)
            else None
        )
        if actual_order != expected_selection_order:
            option_mismatches[
                "barrel_roll.evaluation.checkpoint_selection_order"
            ] = {"expected": expected_selection_order, "actual": actual_order}
        recorded_source_hashes = barrel.get("run_provenance", {}).get(
            "runtime_source_sha256"
        )
        current_source_hashes = _barrel_roll_runtime_source_hashes(
            Path(__file__).resolve().parents[1]
        )
        if recorded_source_hashes != current_source_hashes:
            option_mismatches["barrel_roll.run_provenance.runtime_source_sha256"] = {
                "expected": current_source_hashes,
                "actual": recorded_source_hashes,
            }
    domain_randomization = config.get("domain_randomization", {})
    if not isinstance(domain_randomization, dict) or domain_randomization.get("enabled") is not False:
        option_mismatches["domain_randomization.enabled"] = {
            "expected": False,
            "actual": (
                domain_randomization.get("enabled")
                if isinstance(domain_randomization, dict)
                else domain_randomization
            ),
        }
    if option_mismatches:
        raise ValueError("training configuration mismatch: " + canonical_json(option_mismatches))

    records = _read_jsonl(history_path)
    actual_steps = tuple(int(record.get("timesteps", -1)) for record in records)
    if actual_steps != EXPECTED_VALIDATION_STEPS:
        raise ValueError(
            f"validation steps differ from the frozen schedule: {actual_steps}"
        )
    for index, record in enumerate(records):
        if record.get("seeds") != list(BARREL_ROLL_VALIDATION_SEEDS):
            raise ValueError(
                f"validation history record {index} used the wrong seed set"
            )
        episodes = record.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 100:
            raise ValueError(
                f"validation history record {index} does not contain 100 episodes"
            )
        if [episode.get("seed") for episode in episodes] != list(
            BARREL_ROLL_VALIDATION_SEEDS
        ):
            raise ValueError(
                f"validation history record {index} episode seeds are not frozen"
            )
        success_rate = float(record.get("success_rate", np.nan))
        if not np.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
            raise ValueError(
                f"validation history record {index} has invalid success rate"
            )
        standing_score = float(
            record.get("mean_final_hold_standing_score", np.nan)
        )
        if not np.isfinite(standing_score) or not 0.0 <= standing_score <= 1.0:
            raise ValueError(
                f"validation history record {index} has invalid standing score"
            )
        for diagnostic in (
            "failure_reasons",
            "final_hold_streak_distribution",
            "standing_subscore_distributions",
            "motion_distributions",
            "reward_component_distributions",
            "roll_progress_distributions",
            "contact_summary",
            "return_distribution",
        ):
            if record.get(diagnostic) is None:
                raise ValueError(
                    f"validation history record {index} omitted {diagnostic}"
                )

    winner = max(
        records,
        key=lambda record: (
            float(record["success_rate"]),
            float(record["mean_final_hold_standing_score"]),
            -int(record["timesteps"]),
        ),
    )
    best_rate = float(winner["success_rate"])
    best_standing_score = float(winner["mean_final_hold_standing_score"])
    selected_step = int(winner["timesteps"])
    selection = _read_json(selection_path)
    if selection != {
        "selection_order": [
            "strict_success_rate_desc",
            "mean_final_hold_standing_score_desc",
            "timesteps_asc",
        ],
        "success_rate": best_rate,
        "mean_final_hold_standing_score": best_standing_score,
        "timesteps": selected_step,
    }:
        raise ValueError(
            "selected checkpoint does not match the locked validation order: "
            + canonical_json(selection)
        )
    completion = _read_json(complete_path)
    if (
        completion.get("status") != "complete"
        or completion.get("training_seed") != training_seed
        or completion.get("timesteps") != 500_000
        or completion.get("selected_timesteps") != selected_step
        or completion.get("selected_success_rate") != best_rate
        or completion.get("selected_mean_final_hold_standing_score")
        != best_standing_score
    ):
        raise ValueError("training COMPLETE marker contradicts checkpoint selection")
    artifact_hashes = completion.get("artifact_hashes", {})
    completion_hash_checks = {
        "config": config_path,
        "selected_model": model_path,
        "selected_vecnormalize": vecnormalize_path,
        "selection": selection_path,
        "evaluation_history": history_path,
    }
    for name, path in completion_hash_checks.items():
        if artifact_hashes.get(name) != sha256_file(path):
            raise ValueError(f"training COMPLETE marker hash mismatch: {name}")

    return {
        "training_seed": training_seed,
        "run_dir": str(run_dir.resolve()),
        "validation_success_rate": best_rate,
        "validation_mean_final_hold_standing_score": best_standing_score,
        "selected_timesteps": selected_step,
        "selected_evaluation": winner,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "validation_history": str(history_path.resolve()),
        "validation_history_sha256": sha256_file(history_path),
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "vecnormalize": str(vecnormalize_path.resolve()),
        "vecnormalize_sha256": sha256_file(vecnormalize_path),
        "completion": str(complete_path.resolve()),
        "completion_sha256": sha256_file(complete_path),
    }


def choose_official_policy(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [result for result in results if result["success_count"] >= 80]
    if not passing:
        return None
    return max(
        passing,
        key=lambda result: (
            result["success_count"],
            result["validation_success_rate"],
            result["validation_mean_final_hold_standing_score"],
            -result["training_seed"],
        ),
    )


def _render(
    *,
    selected: dict[str, Any],
    seeds: tuple[int, ...],
    video_dir: Path,
    labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    if video_dir.exists():
        raise FileExistsError(f"refusing to overwrite video directory: {video_dir}")
    model, model_env = load_saved_model_for_video_eval(
        algorithm="SAC-MPC",
        model_path=Path(selected["model"]).with_suffix(""),
        vecnormalize_path=Path(selected["vecnormalize"]),
        domain="quadruped",
        task="barrel_roll",
        is_quadruped=True,
        robot="go2",
        simple_reward=False,
        use_go2_sysid=True,
    )
    try:
        result = evaluate_and_record(
            model=model,
            domain="quadruped",
            task="barrel_roll",
            num_episodes=len(seeds),
            num_videos=len(seeds),
            video_dir=video_dir,
            normalize_env=Path(selected["vecnormalize"]),
            seed=None,
            is_quadruped=True,
            robot="go2",
            simple_reward=False,
            use_go2_sysid=True,
            barrel_roll_seeds=seeds,
            barrel_roll_video_labels=labels,
        )
    finally:
        model_env.close()
    observed_outcomes = [bool(value) for value in result["episode_successes"]]
    evaluation = selected.get("evaluation")
    if isinstance(evaluation, dict):
        expected_by_seed = {
            int(episode["seed"]): bool(episode["success"])
            for episode in evaluation.get("episodes", [])
        }
        expected_outcomes = [expected_by_seed[seed] for seed in seeds]
        if observed_outcomes != expected_outcomes:
            raise RuntimeError(
                "video rerender outcome contradicts the scored final evaluation"
            )
    video_paths = sorted(video_dir.glob("*.mp4"))
    if len(video_paths) != len(seeds) or any(path.stat().st_size == 0 for path in video_paths):
        raise RuntimeError(
            f"expected {len(seeds)} nonempty videos in {video_dir}; found {len(video_paths)}"
        )
    expected_steps_by_seed = {}
    if isinstance(evaluation, dict):
        expected_steps_by_seed = {
            int(episode["seed"]): int(episode["control_steps"])
            for episode in evaluation.get("episodes", [])
        }
    videos = []
    for path in video_paths:
        seed_match = re.search(r"_seed(\d+)\.mp4$", path.name)
        if seed_match is None:
            raise RuntimeError(f"cannot recover evaluation seed from video {path}")
        video_seed = int(seed_match.group(1))
        expected_steps = expected_steps_by_seed.get(video_seed, CONTROL_STEPS)
        if not 1 <= expected_steps <= CONTROL_STEPS:
            raise RuntimeError(
                f"invalid recorded control-step count for seed {video_seed}: "
                f"{expected_steps}"
            )
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            raise RuntimeError(f"expected one video stream in {path}")
        stream = streams[0]
        technical = {
            "codec": stream.get("codec_name"),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "frame_rate": stream.get("r_frame_rate"),
            "duration_seconds": float(stream["duration"]),
            "frame_count": int(stream["nb_read_frames"]),
        }
        if (
            technical["codec"] != "h264"
            or technical["width"] != 640
            or technical["height"] != 480
            or technical["frame_count"] != expected_steps
            or technical["frame_rate"] != "50/1"
            or not np.isclose(
                technical["duration_seconds"],
                expected_steps * (EPISODE_HORIZON / CONTROL_STEPS),
                atol=1.0e-9,
                rtol=0.0,
            )
        ):
            raise RuntimeError(f"video timing mismatch for {path}: {technical}")
        videos.append({
            "seed": video_seed,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "technical": technical,
        })
    return {
        "seeds": list(seeds),
        "outcomes": observed_outcomes,
        "videos": videos,
    }


def run_validation_review(campaign_dir: Path | str) -> dict[str, Any]:
    """Recheck both selected policies and build the Gate V4.5 review set."""
    campaign_dir = Path(campaign_dir).resolve()
    campaign_complete = campaign_dir / "COMPLETE"
    campaign_validation_path = campaign_dir / "campaign_validation.json"
    if not campaign_complete.is_file() or not campaign_validation_path.is_file():
        raise FileNotFoundError("both production training runs are not complete")
    _read_source_locked_campaign_validation(campaign_validation_path)
    validation_dir = campaign_dir / "validation_review"
    if validation_dir.exists():
        raise FileExistsError(
            f"refusing to rerun or overwrite validation review: {validation_dir}"
        )

    selections = [
        validate_selected_checkpoint(campaign_dir, training_seed)
        for training_seed in TRAINING_SEEDS
    ]
    validation_dir.mkdir(parents=False)
    policy_reports = []
    for selected in selections:
        training_seed = int(selected["training_seed"])
        model, model_env = load_saved_model_for_video_eval(
            algorithm="SAC-MPC",
            model_path=Path(selected["model"]).with_suffix(""),
            vecnormalize_path=Path(selected["vecnormalize"]),
            domain="quadruped",
            task="barrel_roll",
            is_quadruped=True,
            robot="go2",
            simple_reward=False,
            use_go2_sysid=True,
        )
        try:
            evaluation = evaluate_barrel_roll_policy(
                model, model_env, BARREL_ROLL_VALIDATION_SEEDS
            )
        finally:
            model_env.close()
        recorded = dict(selected["selected_evaluation"])
        recorded.pop("timesteps", None)
        if canonical_json(evaluation) != canonical_json(recorded):
            raise RuntimeError(
                f"seed {training_seed} selected-policy validation rerun changed"
            )
        evaluation_path = validation_dir / f"seed{training_seed}_evaluation.json"
        _write_json_exclusive(evaluation_path, evaluation)

        success_seeds = [
            int(episode["seed"])
            for episode in evaluation["episodes"]
            if episode["success"]
        ]
        selection_count = min(10, len(success_seeds))
        if selection_count:
            rng = np.random.default_rng(
                VALIDATION_VIDEO_SELECTION_SEED + training_seed
            )
            selected_successes = tuple(
                int(value)
                for value in rng.choice(
                    success_seeds, size=selection_count, replace=False
                )
            )
        else:
            selected_successes = ()
        representative_failures: dict[str, int] = {}
        for episode in evaluation["episodes"]:
            if episode["success"]:
                continue
            reason = str(episode.get("failure_reason") or "unknown")
            representative_failures.setdefault(reason, int(episode["seed"]))
        failure_seeds = tuple(representative_failures.values())
        video_seeds = selected_successes + failure_seeds
        labels = {seed: "success" for seed in selected_successes}
        labels.update({
            seed: reason.replace("/", "_").replace(":", "_")
            for reason, seed in representative_failures.items()
        })
        selected_for_render = {**selected, "evaluation": evaluation}
        video_report = (
            _render(
                selected=selected_for_render,
                seeds=video_seeds,
                video_dir=validation_dir / f"seed{training_seed}_videos",
                labels=labels,
            )
            if video_seeds
            else {"seeds": [], "outcomes": [], "videos": []}
        )
        policy_reports.append({
            "training_seed": training_seed,
            "selection": selected,
            "evaluation": str(evaluation_path.resolve()),
            "evaluation_sha256": sha256_file(evaluation_path),
            "validation_success_count": int(sum(
                episode["success"] for episode in evaluation["episodes"]
            )),
            "validation_success_rate": float(evaluation["success_rate"]),
            "mean_final_hold_standing_score": float(
                evaluation["mean_final_hold_standing_score"]
            ),
            "selected_success_video_seeds": list(selected_successes),
            "representative_failure_video_seeds": representative_failures,
            "video_report": video_report,
        })

    report = {
        "gate": "V4.5_validation_video_review",
        "status": "awaiting_user_visual_approval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_seeds": list(BARREL_ROLL_VALIDATION_SEEDS),
        "validation_reruns_per_policy": 1,
        "video_selection_seed_base": VALIDATION_VIDEO_SELECTION_SEED,
        "final_test_opened": False,
        "campaign_validation": str(campaign_validation_path.resolve()),
        "campaign_validation_sha256": sha256_file(campaign_validation_path),
        "policies": policy_reports,
    }
    report_path = validation_dir / "report.json"
    _write_json_exclusive(report_path, report)
    _write_json_exclusive(
        validation_dir / "READY_FOR_USER_REVIEW.json",
        {
            "status": "awaiting_user_visual_approval",
            "report_sha256": sha256_file(report_path),
            "final_test_opened": False,
        },
    )
    return report


def _policy_diagnostic_comparison(result: dict[str, Any]) -> dict[str, Any]:
    evaluation = result["evaluation"]
    return {
        "training_seed": result["training_seed"],
        "failure_reasons": evaluation["failure_reasons"],
        "final_hold_streak_distribution": evaluation[
            "final_hold_streak_distribution"
        ],
        "final_hold_condition_failure_endpoints": evaluation[
            "final_hold_condition_failure_endpoints"
        ],
        "standing_subscore_distributions": evaluation[
            "standing_subscore_distributions"
        ],
        "motion_distributions": evaluation["motion_distributions"],
        "roll_progress_distributions": evaluation[
            "roll_progress_distributions"
        ],
        "reward_component_distributions": evaluation[
            "reward_component_distributions"
        ],
        "contact_summary": evaluation["contact_summary"],
        "return_distribution": evaluation["return_distribution"],
        "critic_values": evaluation["critic_values"],
    }


def run_final_evaluation(campaign_dir: Path | str) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir).resolve()
    if not (campaign_dir / "COMPLETE").is_file():
        raise FileNotFoundError(f"campaign is not complete: {campaign_dir / 'COMPLETE'}")
    campaign_validation_path = campaign_dir / "campaign_validation.json"
    if not campaign_validation_path.is_file():
        raise FileNotFoundError(
            f"campaign validation is missing: {campaign_validation_path}"
        )
    campaign_validation = _read_source_locked_campaign_validation(
        campaign_validation_path
    )
    if (
        campaign_validation.get("mode") != "production"
        or campaign_validation.get("total_timesteps_per_seed") != 500_000
        or [run.get("seed") for run in campaign_validation.get("runs", [])]
        != list(TRAINING_SEEDS)
    ):
        raise ValueError("campaign validation does not describe the frozen production run")
    validation_review_path = campaign_dir / "validation_review/report.json"
    validation_approval_path = campaign_dir / "validation_review/USER_APPROVED.json"
    if not validation_review_path.is_file() or not validation_approval_path.is_file():
        raise PermissionError(
            "untouched final evaluation requires explicit validation-video approval"
        )
    validation_approval = _read_json(validation_approval_path)
    if (
        validation_approval.get("approved") is not True
        or validation_approval.get("report_sha256")
        != sha256_file(validation_review_path)
    ):
        raise PermissionError("validation-video approval marker is invalid")
    final_dir = campaign_dir / "final_evaluation"
    if final_dir.exists():
        raise FileExistsError(
            f"refusing to rerun or overwrite final evaluation: {final_dir}"
        )

    selections = [
        validate_selected_checkpoint(campaign_dir, training_seed)
        for training_seed in TRAINING_SEEDS
    ]
    final_dir.mkdir(parents=False)
    lock = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scored_final_test_seed_range": [
            BARREL_ROLL_FINAL_TEST_SEEDS[0],
            BARREL_ROLL_FINAL_TEST_SEEDS[-1],
        ],
        "scored_evaluation_passes_per_policy": 1,
        "campaign_validation": str(campaign_validation_path),
        "campaign_validation_sha256": sha256_file(campaign_validation_path),
        "validation_review": str(validation_review_path),
        "validation_review_sha256": sha256_file(validation_review_path),
        "validation_approval": str(validation_approval_path),
        "validation_approval_sha256": sha256_file(validation_approval_path),
        "selections": selections,
    }
    _write_json_exclusive(final_dir / "selection_lock.json", lock)

    scored_results = []
    for selected in selections:
        training_seed = int(selected["training_seed"])
        model, model_env = load_saved_model_for_video_eval(
            algorithm="SAC-MPC",
            model_path=Path(selected["model"]).with_suffix(""),
            vecnormalize_path=Path(selected["vecnormalize"]),
            domain="quadruped",
            task="barrel_roll",
            is_quadruped=True,
            robot="go2",
            simple_reward=False,
            use_go2_sysid=True,
        )
        try:
            evaluation = evaluate_barrel_roll_policy(
                model,
                model_env,
                BARREL_ROLL_FINAL_TEST_SEEDS,
                include_critic=True,
            )
        finally:
            model_env.close()
        if evaluation.get("seeds") != list(BARREL_ROLL_FINAL_TEST_SEEDS):
            raise RuntimeError(f"seed {training_seed} final evaluator used wrong seeds")
        success_count = int(sum(episode["success"] for episode in evaluation["episodes"]))
        if success_count != int(round(float(evaluation["success_rate"]) * 100)):
            raise RuntimeError(f"seed {training_seed} final success count is inconsistent")
        result = {
            **selected,
            "success_count": success_count,
            "success_rate": float(evaluation["success_rate"]),
            "evaluation": evaluation,
        }
        _write_json_exclusive(final_dir / f"seed{training_seed}_final_test.json", result)
        scored_results.append(result)

    official = choose_official_policy(scored_results)
    video_reports: dict[str, Any] = {}
    if official is not None:
        success_seeds = [
            int(episode["seed"])
            for episode in official["evaluation"]["episodes"]
            if episode["success"]
        ]
        if len(success_seeds) < 10:
            raise RuntimeError("passing final policy has fewer than 10 successes")
        rng = np.random.default_rng(FINAL_VIDEO_SELECTION_SEED)
        official_video_seeds = tuple(
            int(value)
            for value in rng.choice(success_seeds, size=10, replace=False)
        )
        video_reports["official_successes"] = _render(
            selected=official,
            seeds=official_video_seeds,
            video_dir=final_dir / "official_success_videos",
            labels={seed: "success" for seed in official_video_seeds},
        )

    for result in scored_results:
        training_seed = int(result["training_seed"])
        failures_by_reason: dict[str, int] = {}
        for episode in result["evaluation"]["episodes"]:
            if episode["success"]:
                continue
            reason = str(episode.get("failure_reason") or "unknown")
            failures_by_reason.setdefault(reason, int(episode["seed"]))
        failure_seeds = tuple(failures_by_reason.values())
        if not failure_seeds:
            video_reports[f"seed{training_seed}_failures"] = {
                "seeds": [],
                "outcomes": [],
                "videos": [],
            }
            continue
        failure_labels = {
            seed: reason.replace("/", "_").replace(":", "_")
            for reason, seed in failures_by_reason.items()
        }
        video_reports[f"seed{training_seed}_failures"] = _render(
            selected=result,
            seeds=failure_seeds,
            video_dir=final_dir / f"seed{training_seed}_failure_videos",
            labels=failure_labels,
        )

    summary = {
        "status": (
            "awaiting_user_final_video_approval"
            if official is not None
            else "failed_numerical_acceptance"
        ),
        "numerical_pass": official is not None,
        "accepted": False,
        "acceptance_threshold": 80,
        "final_video_selection_seed": FINAL_VIDEO_SELECTION_SEED,
        "training_seed_results": [
            {
                "training_seed": result["training_seed"],
                "selected_timesteps": result["selected_timesteps"],
                "validation_success_rate": result["validation_success_rate"],
                "validation_mean_final_hold_standing_score": result[
                    "validation_mean_final_hold_standing_score"
                ],
                "final_test_success_count": result["success_count"],
                "final_test_success_rate": result["success_rate"],
                "failure_reasons": result["evaluation"]["failure_reasons"],
                "critic_values": result["evaluation"]["critic_values"],
            }
            for result in scored_results
        ],
        "official_training_seed": (
            int(official["training_seed"]) if official is not None else None
        ),
        "official_artifact": (
            {
                key: official[key]
                for key in (
                    "training_seed",
                    "selected_timesteps",
                    "validation_success_rate",
                    "validation_mean_final_hold_standing_score",
                    "success_count",
                    "success_rate",
                    "model",
                    "model_sha256",
                    "vecnormalize",
                    "vecnormalize_sha256",
                )
            }
            if official is not None
            else None
        ),
        "official_tiebreak_order": [
            "final_test_success_rate",
            "validation_success_rate",
            "mean_final_hold_standing_score",
            "lower_training_seed",
        ],
        "videos_are_explicit_rerenders_not_scored_passes": True,
        "video_reports": video_reports,
        "diagnostic_comparison": [
            _policy_diagnostic_comparison(result) for result in scored_results
        ],
    }
    summary_path = final_dir / "summary.json"
    _write_json_exclusive(summary_path, summary)
    if official is not None:
        _write_json_exclusive(
            final_dir / "READY_FOR_USER_REVIEW.json",
            {
                "status": "awaiting_user_final_video_approval",
                "summary_sha256": sha256_file(summary_path),
                "official_training_seed": int(official["training_seed"]),
                "numerical_pass": True,
            },
        )
    return summary


def record_user_video_approval(
    campaign_dir: Path | str,
    *,
    gate: str,
    user_statement: str,
) -> dict[str, Any]:
    """Record an explicit user review without rerunning any rollout."""
    campaign_dir = Path(campaign_dir).resolve()
    statement = user_statement.strip()
    if not statement:
        raise ValueError("user approval statement must be nonempty")
    if gate == "validation":
        review_dir = campaign_dir / "validation_review"
        report_path = review_dir / "report.json"
        ready_path = review_dir / "READY_FOR_USER_REVIEW.json"
        approval_path = review_dir / "USER_APPROVED.json"
        if not report_path.is_file() or not ready_path.is_file():
            raise FileNotFoundError("validation review package is not ready")
        report = _read_json(report_path)
        ready = _read_json(ready_path)
        report_hash = sha256_file(report_path)
        if (
            report.get("status") != "awaiting_user_visual_approval"
            or ready.get("report_sha256") != report_hash
            or ready.get("final_test_opened") is not False
        ):
            raise ValueError("validation review package is inconsistent")
        payload = {
            "gate": "V4.5_validation_video_review",
            "approved": True,
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "user_statement": statement,
            "report_sha256": report_hash,
            "final_test_opened_at_approval": False,
        }
        _write_json_exclusive(approval_path, payload)
        return payload
    if gate == "final":
        final_dir = campaign_dir / "final_evaluation"
        summary_path = final_dir / "summary.json"
        ready_path = final_dir / "READY_FOR_USER_REVIEW.json"
        approval_path = final_dir / "USER_APPROVED.json"
        accepted_path = final_dir / "ACCEPTED.json"
        if not summary_path.is_file() or not ready_path.is_file():
            raise FileNotFoundError("final review package is not ready")
        summary = _read_json(summary_path)
        ready = _read_json(ready_path)
        summary_hash = sha256_file(summary_path)
        official_videos = summary.get("video_reports", {}).get(
            "official_successes", {}
        )
        if (
            summary.get("numerical_pass") is not True
            or summary.get("status") != "awaiting_user_final_video_approval"
            or ready.get("summary_sha256") != summary_hash
            or len(official_videos.get("videos", [])) != 10
            or official_videos.get("outcomes") != [True] * 10
        ):
            raise ValueError("final review package does not satisfy acceptance inputs")
        payload = {
            "gate": "V4.6_final_video_review",
            "approved": True,
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "user_statement": statement,
            "summary_sha256": summary_hash,
            "official_training_seed": summary["official_training_seed"],
        }
        _write_json_exclusive(approval_path, payload)
        accepted = {
            "status": "accepted",
            "strict_final_metric_passed": True,
            "user_final_video_review_passed": True,
            "acceptance_threshold": summary["acceptance_threshold"],
            "official_training_seed": summary["official_training_seed"],
            "official_artifact": summary["official_artifact"],
            "summary_sha256": summary_hash,
            "user_approval_sha256": sha256_file(approval_path),
        }
        _write_json_exclusive(accepted_path, accepted)
        return accepted
    raise ValueError(f"unknown approval gate: {gate!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=Path("logs/go2_barrel_roll_v4/campaign"),
    )
    parser.add_argument(
        "--mode",
        choices=("validation-review", "approve-validation", "final", "approve-final"),
        required=True,
    )
    parser.add_argument("--user-statement")
    args = parser.parse_args()
    if args.mode == "validation-review":
        summary = run_validation_review(args.campaign_dir)
        print(canonical_json(summary))
        return
    if args.mode in {"approve-validation", "approve-final"}:
        if args.user_statement is None:
            parser.error(f"--user-statement is required for {args.mode}")
        gate = "validation" if args.mode == "approve-validation" else "final"
        approval = record_user_video_approval(
            args.campaign_dir,
            gate=gate,
            user_statement=args.user_statement,
        )
        print(canonical_json(approval))
        return
    summary = run_final_evaluation(args.campaign_dir)
    print(canonical_json(summary))
    if not summary["numerical_pass"]:
        raise SystemExit("schema-v4 final acceptance failed: neither policy reached 80/100")


if __name__ == "__main__":
    main()
