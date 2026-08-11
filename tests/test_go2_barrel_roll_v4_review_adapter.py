from mpc_rl import evaluate_go2_barrel_roll_g9 as evaluation
from mpc_rl.run_go2_barrel_roll_v4_review_adapter import (
    install_outcome_label_adapter,
    outcome_video_labels,
)


def test_outcome_video_labels_preserve_success_and_collapse_failures():
    assert outcome_video_labels(None) is None
    assert outcome_video_labels({
        2_000_000: "success",
        2_000_001: "base_linear_speed_above_maximum",
        2_000_002: "insufficient_foot_support",
    }) == {
        2_000_000: "success",
        2_000_001: "failure",
        2_000_002: "failure",
    }


def test_install_outcome_label_adapter_changes_only_render_labels(monkeypatch):
    captured = {}

    def render(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(evaluation, "_render", render)
    install_outcome_label_adapter()
    result = evaluation._render(
        selected={"training_seed": 1},
        seeds=(2_000_000, 2_000_001),
        video_dir="videos",
        labels={
            2_000_000: "success",
            2_000_001: "rotation_out_of_band",
        },
    )

    assert result == {"ok": True}
    assert captured == {
        "selected": {"training_seed": 1},
        "seeds": (2_000_000, 2_000_001),
        "video_dir": "videos",
        "labels": {2_000_000: "success", 2_000_001: "failure"},
    }
