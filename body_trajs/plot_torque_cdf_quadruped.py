#!/usr/bin/env python3
"""Plot quadruped average motor torque magnitude CDFs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to plot torque CDFs. "
        "Please run this script from the project's configured environment "
        "(see environment.yml)."
    ) from exc


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "body_trajectory_plots"
FONT_SIZE = 18
PURE_RL_COLOR = "#1f77b4"
MPC_INJECTION_COLOR = "#ff7f0e"

# Edit these to choose which rollout timesteps to include in the CDF.
PLOT_TIMESTEP_START = 0
PLOT_TIMESTEP_END = None

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "figure.titlesize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 2,
        "legend.fontsize": FONT_SIZE - 1,
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
            "Plot the empirical CDF of per-timestep average motor torque magnitude "
            "from one or more saved quadruped rollouts."
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to a quadruped trajectory .npz file. "
            "Repeat this flag to overlay multiple CDFs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output image path. Defaults to "
            "plots/body_trajectory_plots/quadruped_torque_cdf.png"
        ),
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


def resolve_output_path(output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / "quadruped_torque_cdf.png"


def slice_timesteps(num_timesteps: int) -> slice:
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


def infer_torque_matrix(data: np.lib.npyio.NpzFile) -> np.ndarray:
    tau_applied = np.asarray(data["tau_applied"], dtype=float)
    if tau_applied.ndim != 2:
        raise ValueError(
            "Expected tau_applied to have 2 dimensions "
            f"(motors, timesteps) or (timesteps, motors); got {tau_applied.shape}"
        )

    if "actuated_joint_names" in data.files:
        num_motors = len(data["actuated_joint_names"])
        if tau_applied.shape[0] == num_motors:
            return tau_applied
        if tau_applied.shape[1] == num_motors:
            return tau_applied.T

    if "num_sim_steps" in data.files:
        num_sim_steps = int(data["num_sim_steps"])
        if tau_applied.shape[1] == num_sim_steps:
            return tau_applied
        if tau_applied.shape[0] == num_sim_steps:
            return tau_applied.T

    if tau_applied.shape[0] <= tau_applied.shape[1]:
        return tau_applied
    return tau_applied.T


def compute_average_torque_magnitude(data: np.lib.npyio.NpzFile) -> np.ndarray:
    tau_applied = infer_torque_matrix(data)
    avg_torque_magnitude = np.mean(np.abs(tau_applied), axis=0)
    timestep_slice = slice_timesteps(avg_torque_magnitude.shape[0])
    return avg_torque_magnitude[timestep_slice]


def compute_empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    cdf = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
    return sorted_values, cdf


def make_trajectory_label(trajectory_path: Path) -> str:
    match = re.search(r"percentage-(\d+)pct", trajectory_path.parent.name)
    if match:
        pct = int(match.group(1))
        if pct == 0:
            return "0% MPC-Injection"
        return f"{pct}% MPC-Injection"
    return trajectory_path.parent.name


def make_trajectory_color(trajectory_path: Path) -> str | None:
    match = re.search(r"percentage-(\d+)pct", trajectory_path.parent.name)
    if not match:
        return None

    pct = int(match.group(1))
    if pct == 0:
        return PURE_RL_COLOR
    if pct == 25:
        return MPC_INJECTION_COLOR
    return None


def summarize_torque_distribution(values: np.ndarray) -> str:
    return (
        f"mean={np.mean(values):.3f} Nm, "
        f"median={np.median(values):.3f} Nm, "
        f"p90={np.percentile(values, 90):.3f} Nm"
    )


def main() -> None:
    args = parse_args()
    trajectory_paths = [path.expanduser().resolve() for path in args.trajectory]
    for trajectory_path in trajectory_paths:
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    fig.suptitle(
        "Quadruped Average Motor Torque\nMagnitude CDF",
        fontweight="bold",
    )

    for trajectory_path in trajectory_paths:
        data = np.load(trajectory_path, allow_pickle=True)
        ensure_keys(data, ["tau_applied"])
        avg_torque_magnitude = compute_average_torque_magnitude(data)
        x_axis, cdf = compute_empirical_cdf(avg_torque_magnitude)
        label = make_trajectory_label(trajectory_path)
        ax.plot(
            x_axis,
            cdf,
            linewidth=2.0,
            alpha=0.95,
            label=label,
            color=make_trajectory_color(trajectory_path),
        )
        print(f"{label}: {summarize_torque_distribution(avg_torque_magnitude)}")

    ax.set_xlabel("Average Motor Torque Magnitude (Nm)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.01)
    ax.margins(x=0.01, y=0.02)
    ax.tick_params(axis="both")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_show and backend_is_interactive():
        plt.show()
    elif not args.no_show:
        print(
            "Matplotlib is using a non-interactive backend "
            f"({plt.get_backend()}); skipping plt.show() and saving directly."
        )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved torque CDF plot to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
