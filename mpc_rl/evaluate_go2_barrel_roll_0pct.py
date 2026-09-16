#!/usr/bin/env python3
"""Render the matched checkpoint and final 0% barrel-roll baseline policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from body_trajs.record_body_trajs_by_policy_quadruped import resolve_run_dir
from mpc_rl.planner.barrel_roll_dataset import sha256_file
from mpc_rl.train import evaluate_and_record, load_saved_model_for_video_eval


DEFAULT_CHECKPOINT_STEP = 220_000
DEFAULT_SEEDS = tuple(range(3_000_000, 3_000_010))


def _validate_config(run_dir: Path, checkpoint_step: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "algorithm": "SAC-MPC",
        "env_name": "quadruped-barrel_roll",
        "percentage": 0,
        "seed": 1,
        "total_timesteps": 500_000,
        "num_envs": 4,
        "checkpoint_freq": 20_000,
        "eval_freq": 10_000,
        "use_go2_sysid": True,
        "save_replay_buffer_checkpoints": False,
        "save_replay_buffer_final": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    domain_randomization = config.get("domain_randomization", {})
    if domain_randomization.get("enabled") is not False:
        mismatches["domain_randomization.enabled"] = {
            "expected": False,
            "actual": domain_randomization.get("enabled"),
        }
    if mismatches:
        raise ValueError(f"baseline configuration mismatch: {mismatches}")

    required = (
        run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip",
        run_dir / "checkpoints" / f"model_vecnormalize_{checkpoint_step}_steps.pkl",
        run_dir / "final_model.zip",
        run_dir / "vec_normalize.pkl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing policy artifacts: " + ", ".join(missing))
    return config


def _render_policy(
    *,
    model_path: Path,
    vecnormalize_path: Path,
    video_dir: Path,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
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
    return {
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "vecnormalize": str(vecnormalize_path.resolve()),
        "vecnormalize_sha256": sha256_file(vecnormalize_path),
        "seeds": list(seeds),
        "success_count": sum(bool(value) for value in evaluation["episode_successes"]),
        "failure_reasons": evaluation["failure_reasons"],
        "episode_rewards": [float(value) for value in evaluation["episode_rewards"]],
        "episode_lengths": [int(value) for value in evaluation["episode_lengths"]],
        "videos": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in videos
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=DEFAULT_CHECKPOINT_STEP)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    _validate_config(run_dir, args.checkpoint_step)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "baseline_evaluation"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output_dir}")
    output_dir.mkdir(parents=True)

    checkpoint_model = (
        run_dir / "checkpoints" / f"model_{args.checkpoint_step}_steps.zip"
    )
    checkpoint_vecnormalize = (
        run_dir
        / "checkpoints"
        / f"model_vecnormalize_{args.checkpoint_step}_steps.pkl"
    )
    summary = {
        "run_dir": str(run_dir),
        "matched_25pct_official_checkpoint_step": args.checkpoint_step,
        "video_seeds": list(DEFAULT_SEEDS),
        "checkpoint": _render_policy(
            model_path=checkpoint_model,
            vecnormalize_path=checkpoint_vecnormalize,
            video_dir=output_dir / f"checkpoint_{args.checkpoint_step}_videos",
            seeds=DEFAULT_SEEDS,
        ),
        "final": _render_policy(
            model_path=run_dir / "final_model.zip",
            vecnormalize_path=run_dir / "vec_normalize.pkl",
            video_dir=output_dir / "final_model_videos",
            seeds=DEFAULT_SEEDS,
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
