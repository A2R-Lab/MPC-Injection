#!/usr/bin/env python3
"""Record deterministic videos of a completed v4 0% baseline's best policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from body_trajs.record_body_trajs_by_policy_quadruped import (  # noqa: E402
    resolve_run_dir,
)
from mpc_rl.planner.barrel_roll_dataset import sha256_file  # noqa: E402
from mpc_rl.train import (  # noqa: E402
    evaluate_and_record,
    load_saved_model_for_video_eval,
)


BASELINE_MODE = "schema_v4_0pct_baseline"
DEFAULT_VIDEO_SEEDS = tuple(range(7_000_000, 7_000_010))


def parse_video_seeds(value: str) -> tuple[int, ...]:
    """Parse a nonempty, unique comma-separated rollout seed list."""
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("video seeds must be integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one video seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("video seeds must be unique")
    return seeds


def validate_zero_percent_run(run_dir: Path) -> dict[str, Any]:
    """Validate the completed baseline identity and selected artifact hashes."""
    required = {
        "config": run_dir / "config.json",
        "complete": run_dir / "COMPLETE",
        "selected_model": run_dir / "best_model/best_model.zip",
        "selected_vecnormalize": run_dir / "best_model/vec_normalize.pkl",
        "selection": run_dir / "best_model/selection.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing completed baseline artifacts: " + ", ".join(missing)
        )

    config = json.loads(required["config"].read_text(encoding="utf-8"))
    complete = json.loads(required["complete"].read_text(encoding="utf-8"))
    selection = json.loads(required["selection"].read_text(encoding="utf-8"))
    expected_config = {
        "algorithm": "SAC-MPC",
        "env_name": "quadruped-barrel_roll",
        "percentage": 0,
        "total_timesteps": 500_000,
        "num_envs": 4,
        "max_episode_steps": 125,
        "checkpoint_freq": 25_000,
        "eval_freq": 10_000,
        "use_go2_sysid": True,
        "quadruped_mpc_replay_mode": "direct",
    }
    mismatches = {
        name: {"expected": expected, "actual": config.get(name)}
        for name, expected in expected_config.items()
        if config.get(name) != expected
    }
    barrel_roll = config.get("barrel_roll", {})
    dataset = barrel_roll.get("dataset", {}) if isinstance(barrel_roll, dict) else {}
    if dataset.get("schema_version") != 4:
        mismatches["barrel_roll.dataset.schema_version"] = {
            "expected": 4,
            "actual": dataset.get("schema_version"),
        }
    if dataset.get("target_mpc_percentage") != 0:
        mismatches["barrel_roll.dataset.target_mpc_percentage"] = {
            "expected": 0,
            "actual": dataset.get("target_mpc_percentage"),
        }
    if complete.get("status") != "complete":
        mismatches["COMPLETE.status"] = {
            "expected": "complete",
            "actual": complete.get("status"),
        }
    if complete.get("mode") != BASELINE_MODE:
        mismatches["COMPLETE.mode"] = {
            "expected": BASELINE_MODE,
            "actual": complete.get("mode"),
        }
    if complete.get("target_mpc_percentage") != 0:
        mismatches["COMPLETE.target_mpc_percentage"] = {
            "expected": 0,
            "actual": complete.get("target_mpc_percentage"),
        }
    if complete.get("selected_timesteps") != selection.get("timesteps"):
        mismatches["COMPLETE.selected_timesteps"] = {
            "expected": selection.get("timesteps"),
            "actual": complete.get("selected_timesteps"),
        }
    artifact_hashes = complete.get("artifact_hashes", {})
    for name in ("config", "selected_model", "selected_vecnormalize", "selection"):
        actual_hash = sha256_file(required[name])
        if artifact_hashes.get(name) != actual_hash:
            mismatches[f"COMPLETE.artifact_hashes.{name}"] = {
                "expected": actual_hash,
                "actual": artifact_hashes.get(name),
            }
    config_launcher = barrel_roll.get("run_provenance", {}).get(
        "baseline_launcher"
    )
    if (
        not isinstance(config_launcher, dict)
        or config_launcher.get("mode") != BASELINE_MODE
        or not isinstance(config_launcher.get("sha256"), str)
    ):
        mismatches["config.baseline_launcher"] = {
            "expected": {"mode": BASELINE_MODE, "sha256": "<sha256>"},
            "actual": config_launcher,
        }
    if complete.get("baseline_launcher") != config_launcher:
        mismatches["baseline_launcher"] = {
            "expected": config_launcher,
            "actual": complete.get("baseline_launcher"),
        }
    if mismatches:
        raise ValueError(
            "schema-v4 0% baseline mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        **required,
        "config": config,
        "complete": complete,
        "selection": selection,
    }


def render_selected_policy(
    *, run: dict[str, Any], video_dir: Path, seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Load the success-selected best model and render the requested seeds."""
    model_path = run["selected_model"]
    vecnormalize_path = run["selected_vecnormalize"]
    model, model_env = load_saved_model_for_video_eval(
        algorithm="SAC-MPC",
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        domain="quadruped",
        task="barrel_roll",
        is_quadruped=True,
        robot="go2",
        simple_reward=False,
        use_go2_sysid=True,
    )
    try:
        evaluation = evaluate_and_record(
            model=model,
            domain="quadruped",
            task="barrel_roll",
            num_episodes=len(seeds),
            num_videos=len(seeds),
            video_dir=video_dir,
            normalize_env=vecnormalize_path,
            is_quadruped=True,
            robot="go2",
            simple_reward=False,
            use_go2_sysid=True,
            barrel_roll_seeds=seeds,
        )
    finally:
        model_env.close()

    videos = sorted(video_dir.glob("*.mp4"))
    if len(videos) != len(seeds) or any(path.stat().st_size == 0 for path in videos):
        raise RuntimeError(
            f"expected {len(seeds)} nonempty videos in {video_dir}; found {len(videos)}"
        )
    episode_successes = [
        bool(value) for value in evaluation["episode_successes"]
    ]
    return {
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "vecnormalize": str(vecnormalize_path.resolve()),
        "vecnormalize_sha256": sha256_file(vecnormalize_path),
        "selected_timesteps": int(run["selection"]["timesteps"]),
        "seeds": list(seeds),
        "success_count": sum(episode_successes),
        "episode_successes": episode_successes,
        "successful_seeds": [
            seed for seed, success in zip(seeds, episode_successes) if success
        ],
        "failure_reasons": evaluation["failure_reasons"],
        "videos": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in videos
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--seeds",
        type=parse_video_seeds,
        default=DEFAULT_VIDEO_SEEDS,
        help=(
            "Comma-separated rollout seeds. The default is the ten non-scored "
            "seeds 7000000 through 7000009."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    run = validate_zero_percent_run(run_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "selected_policy_videos"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite video evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    summary = {
        "mode": BASELINE_MODE,
        "run_dir": str(run_dir.resolve()),
        "training_seed": int(run["config"]["seed"]),
        "selected_policy": render_selected_policy(
            run=run,
            video_dir=output_dir / "videos",
            seeds=args.seeds,
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
