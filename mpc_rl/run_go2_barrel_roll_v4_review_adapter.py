"""Run the frozen barrel-roll evaluator with outcome-only video labels.

The production evaluator records representative failure reasons in its review
report, but ``evaluate_and_record`` intentionally accepts only the filename
labels ``success`` and ``failure``.  This launcher preserves the hash-locked
evaluator and translates only the labels passed to its video renderer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mpc_rl import evaluate_go2_barrel_roll_g9 as evaluation
from mpc_rl.planner.barrel_roll_dataset import canonical_json, sha256_file


def outcome_video_labels(labels: dict[int, str] | None) -> dict[int, str] | None:
    """Map diagnostic labels to the recorder's outcome-only contract."""
    if labels is None:
        return None
    return {
        int(seed): "success" if label == "success" else "failure"
        for seed, label in labels.items()
    }


def install_outcome_label_adapter() -> None:
    """Adapt the evaluator renderer without modifying its hash-locked source."""
    original_render = evaluation._render

    def render_with_outcome_labels(**kwargs: Any) -> dict[str, Any]:
        adapted = dict(kwargs)
        adapted["labels"] = outcome_video_labels(kwargs.get("labels"))
        return original_render(**adapted)

    evaluation._render = render_with_outcome_labels


def verify_frozen_evaluator(campaign_dir: Path) -> dict[str, str]:
    """Require the imported evaluator to match the completed campaign lock."""
    validation_path = campaign_dir / "campaign_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    locked_hash = validation.get("source_hashes", {}).get("evaluation")
    evaluator_path = Path(evaluation.__file__).resolve()
    actual_hash = sha256_file(evaluator_path)
    if not isinstance(locked_hash, str) or actual_hash != locked_hash:
        raise ValueError(
            "frozen evaluator hash mismatch: "
            + canonical_json({"locked": locked_hash, "actual": actual_hash})
        )
    return {
        "campaign_validation": str(validation_path.resolve()),
        "campaign_validation_sha256": sha256_file(validation_path),
        "frozen_evaluator": str(evaluator_path),
        "frozen_evaluator_sha256": actual_hash,
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
    }


def write_adapter_provenance(
    *, campaign_dir: Path, mode: str, provenance: dict[str, str]
) -> Path:
    output_dir = (
        campaign_dir / "validation_review"
        if mode == "validation-review"
        else campaign_dir / "final_evaluation"
    )
    marker_path = output_dir / "VIDEO_LABEL_ADAPTER.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "mapping": "success remains success; every diagnostic failure label becomes failure",
        **provenance,
    }
    evaluation._write_json_exclusive(marker_path, payload)
    return marker_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("validation-review", "final"), required=True
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()

    campaign_dir = args.campaign_dir.resolve()
    provenance = verify_frozen_evaluator(campaign_dir)
    install_outcome_label_adapter()
    if args.mode == "validation-review":
        result = evaluation.run_validation_review(campaign_dir)
    else:
        result = evaluation.run_final_evaluation(campaign_dir)
    marker_path = write_adapter_provenance(
        campaign_dir=campaign_dir, mode=args.mode, provenance=provenance
    )
    print(
        "BARREL_ROLL_V4_EVALUATION_READY "
        + canonical_json({
            "mode": args.mode,
            "result_status": result["status"],
            "adapter_provenance": str(marker_path.resolve()),
            "adapter_provenance_sha256": sha256_file(marker_path),
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
