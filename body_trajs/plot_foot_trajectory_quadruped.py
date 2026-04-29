#!/usr/bin/env python3
"""Plot quadruped foot trajectories from a recorded rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to plot foot trajectories. "
        "Please run this script from the project's configured environment "
        "(see environment.yml)."
    ) from exc


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "body_trajectory_plots"
BASE_FONT_SIZE = 24
TITLE_FONT_SIZE = BASE_FONT_SIZE + 8
SUBPLOT_TITLE_FONT_SIZE = BASE_FONT_SIZE + 6
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE + 2
TICK_LABEL_FONT_SIZE = BASE_FONT_SIZE - 2
LEGEND_FONT_SIZE = BASE_FONT_SIZE
FOOT_COLORS = {
    "FL": "#1f77b4",
    "FR": "#ff7f0e",
    "RL": "#2ca02c",
    "RR": "#d62728",
}

# Edit these to choose which rollout timesteps to visualize.
# Example: 500, 1000 plots timesteps [500, 1000).
PLOT_TIMESTEP_START = 250
PLOT_TIMESTEP_END = 500

X_AXIS_LABEL = "X (m)"
BASE_FRAME_Y_AXIS_LABEL = "Height (m)"
WORLD_FRAME_Y_AXIS_LABEL = "Z (m)"

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "axes.titlesize": SUBPLOT_TITLE_FONT_SIZE,
        "figure.titlesize": TITLE_FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)


INTERACTIVE_BACKEND_NAMES = {
    "gtk3agg",
    "gtk3cairo",
    "gtk4agg",
    "gtk4cairo",
    "macosx",
    "nbagg",
    "notebook",
    "qt5agg",
    "qt5cairo",
    "qtagg",
    "qtcairo",
    "tkagg",
    "tkcairo",
    "webagg",
    "wx",
    "wxagg",
    "wxcairo",
}


def backend_is_interactive() -> bool:
    return plt.get_backend().lower() in INTERACTIVE_BACKEND_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot quadruped foot trajectories from a saved rollout. "
            "Each foot is shown in its own subplot using a body-relative "
            "sagittal plane so the gait cycle is easy to see."
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Path to a quadruped trajectory .npz file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to plots/body_trajectory_plots/<trajectory_stem>_foot_cycles.png",
    )
    parser.add_argument(
        "--frame",
        choices=("base", "world"),
        default="base",
        help="Plot in the base-relative frame or world frame. Default: base",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive display and only save the figure.",
    )
    return parser.parse_args()


def ensure_keys(data: np.lib.npyio.NpzFile, required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise KeyError(
            f"Trajectory file is missing required arrays: {missing}. "
            f"Available keys: {sorted(data.files)}"
        )


def compute_plot_positions(data: np.lib.npyio.NpzFile, frame: str) -> np.ndarray:
    foot_positions = np.asarray(data["foot_positions"], dtype=float)
    if frame == "world":
        return foot_positions

    base_positions = np.asarray(data["base_positions"], dtype=float)
    return foot_positions - base_positions[:, None, :]


def plot_foot_cycle(
    ax: plt.Axes,
    foot_name: str,
    positions: np.ndarray,
    color: str,
    frame: str,
) -> None:
    x = positions[:, 0]
    z = positions[:, 2]
    x_center = 0.5 * (x.min() + x.max())
    x = x - x_center

    if frame == "base":
        y_axis = -z
        y_label = BASE_FRAME_Y_AXIS_LABEL
    else:
        y_axis = z
        y_label = WORLD_FRAME_Y_AXIS_LABEL

    ax.plot(x, y_axis, color=color, linewidth=2.2, alpha=0.95, zorder=2)
    #ax.scatter(
    #    x[0],
    #    y_axis[0],
    #    s=70,
    #    c="black",
    #    marker="o",
    #    label="start",
    #    zorder=5,
    #)

    ax.set_title(foot_name, fontsize=SUBPLOT_TITLE_FONT_SIZE, fontweight="bold")
    ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, alpha=0.25)
    x_extent = np.max(np.abs(x))
    if x_extent > 0:
        ax.set_xlim(-x_extent, x_extent)


def resolve_output_path(trajectory_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = trajectory_path.parent.name
    return DEFAULT_OUTPUT_DIR / f"{run_name}_{trajectory_path.stem}_foot_cycles.png"


def slice_timesteps(foot_positions: np.ndarray) -> slice:
    num_timesteps = foot_positions.shape[0]
    start = PLOT_TIMESTEP_START if PLOT_TIMESTEP_START is not None else 0
    end = PLOT_TIMESTEP_END if PLOT_TIMESTEP_END is not None else num_timesteps

    if start < 0 or end < 0:
        raise ValueError("PLOT_TIMESTEP_START and PLOT_TIMESTEP_END must be non-negative or None.")
    if start > num_timesteps:
        raise ValueError(
            f"PLOT_TIMESTEP_START={start} exceeds available timesteps ({num_timesteps})."
        )
    if end > num_timesteps:
        raise ValueError(
            f"PLOT_TIMESTEP_END={end} exceeds available timesteps ({num_timesteps})."
        )
    if start >= end:
        raise ValueError(
            f"Invalid timestep range: start={start}, end={end}. "
            "Expected start < end."
        )

    return slice(start, end)


def main() -> None:
    args = parse_args()
    trajectory_path = args.trajectory.expanduser().resolve()
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    data = np.load(trajectory_path, allow_pickle=True)
    ensure_keys(data, ["foot_positions", "foot_names", "base_positions"])

    foot_positions = compute_plot_positions(data, args.frame)
    foot_names = [str(name) for name in data["foot_names"]]

    if foot_positions.ndim != 3 or foot_positions.shape[1] != len(foot_names) or foot_positions.shape[2] != 3:
        raise ValueError(
            "Expected foot_positions to have shape (timesteps, num_feet, 3); "
            f"got {foot_positions.shape}"
        )

    timestep_slice = slice_timesteps(foot_positions)
    foot_positions = foot_positions[timestep_slice]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(
        #f"Quadruped Foot Trajectories ({args.frame}-frame)\n{trajectory_path.name}",
        "Quadruped Foot Trajectories",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    axes_flat = axes.flat
    for foot_idx, (ax, foot_name) in enumerate(zip(axes_flat, foot_names)):
        plot_foot_cycle(
            ax,
            foot_name,
            foot_positions[:, foot_idx, :],
            FOOT_COLORS.get(foot_name, "#444444"),
            args.frame,
        )
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    output_path = resolve_output_path(trajectory_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_show and backend_is_interactive():
        plt.show()
    elif not args.no_show:
        print(
            "Matplotlib is using a non-interactive backend "
            f"({plt.get_backend()}); skipping plt.show() and saving directly."
        )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved foot trajectory plot to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
