#!/usr/bin/env python3
"""
Script to plot grouped box plots of walker torso height across training checkpoints.

The goal is to compare pure RL against MPC-Injection and highlight that MPC-Injection
keeps the torso upright much earlier in training, while pure RL often drags the torso
near the floor.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FONT_SIZE = 18
PURE_RL_COLOR = '#1f77b4'
MPC_INJECTION_COLOR = '#ff7f0e'
DRAG_THRESHOLD_METERS = 0.35

plt.rcParams.update(
    {
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'figure.titlesize': FONT_SIZE + 2,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
        'legend.fontsize': FONT_SIZE - 1,
    }
)


def extract_checkpoint_number(filename: str):
    """Extract checkpoint step from a filename like trajectories_step_100000.npz."""
    match = re.search(r'trajectories_step_(\d+)\.npz', filename)
    if match:
        return int(match.group(1))
    return None


def get_checkpoint_files(run_dir: Path):
    """Map checkpoint step -> trajectory file path for a recorded walker run."""
    files = {}
    for path in run_dir.glob('trajectories_step_*.npz'):
        checkpoint = extract_checkpoint_number(path.name)
        if checkpoint is not None:
            files[checkpoint] = path
    return files


def load_torso_height(npz_path: Path):
    """Load the torso-height trajectory for a saved walker rollout."""
    data = np.load(npz_path)
    if 'torso_height' in data:
        return data['torso_height']
    return data['pos_torso'][:, 2]


def format_checkpoint_label(checkpoint: int):
    """Format checkpoint steps as compact tick labels."""
    return f'{checkpoint // 1000}k'


def draw_grouped_boxplots(ax, pure_rl_data, mpc_data, checkpoints):
    """Draw side-by-side torso-height box plots for each checkpoint."""
    centers = np.arange(len(checkpoints), dtype=float) * 1.6
    offset = 0.24
    width = 0.36

    pure_positions = centers - offset
    mpc_positions = centers + offset

    common_boxplot_kwargs = dict(
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(color='#333333', linewidth=1.7),
        capprops=dict(color='#333333', linewidth=1.7),
    )

    pure_boxplot_kwargs = dict(common_boxplot_kwargs)
    pure_boxplot_kwargs['boxprops'] = dict(
        facecolor=PURE_RL_COLOR,
        edgecolor=PURE_RL_COLOR,
        linewidth=1.8,
    )
    ax.boxplot(
        pure_rl_data,
        positions=pure_positions,
        **pure_boxplot_kwargs,
    )

    mpc_boxplot_kwargs = dict(common_boxplot_kwargs)
    mpc_boxplot_kwargs['boxprops'] = dict(
        facecolor=MPC_INJECTION_COLOR,
        edgecolor=MPC_INJECTION_COLOR,
        linewidth=1.8,
    )
    ax.boxplot(
        mpc_data,
        positions=mpc_positions,
        **mpc_boxplot_kwargs,
    )

    pure_medians = [float(np.median(values)) for values in pure_rl_data]
    mpc_medians = [float(np.median(values)) for values in mpc_data]

    ax.scatter(
        pure_positions,
        pure_medians,
        color='white',
        edgecolors='black',
        linewidths=1.0,
        s=42,
        zorder=4,
    )
    ax.scatter(
        mpc_positions,
        mpc_medians,
        color='white',
        edgecolors='black',
        linewidths=1.0,
        s=42,
        zorder=4,
    )

    ax.set_xticks(centers)
    ax.set_xticklabels([format_checkpoint_label(cp) for cp in checkpoints])
    ax.set_xlim(centers[0] - 0.8, centers[-1] + 0.8)


def summarize_policy(checkpoints, torso_height_data):
    """Return concise text summaries for terminal output."""
    summaries = []
    for checkpoint, values in zip(checkpoints, torso_height_data):
        below_drag_threshold = 100.0 * np.mean(values < DRAG_THRESHOLD_METERS)
        summaries.append(
            (
                checkpoint,
                float(np.median(values)),
                float(np.mean(values)),
                below_drag_threshold,
            )
        )
    return summaries


def plot_torso_height_boxplots(
    pure_rl_dir: Path,
    mpc_injection_dir: Path,
    checkpoints=(25000, 100000, 150000, 200000),
):
    """
    Plot torso-height distributions across selected training checkpoints.

    Args:
        pure_rl_dir: Trajectory directory for the 0% MPC-Injection policy.
        mpc_injection_dir: Trajectory directory for the MPC-Injection policy.
        checkpoints: Iterable of training steps to compare.
    """
    pure_rl_files = get_checkpoint_files(pure_rl_dir)
    mpc_files = get_checkpoint_files(mpc_injection_dir)

    checkpoints = list(checkpoints)
    common_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint in pure_rl_files and checkpoint in mpc_files
    ]

    if not common_checkpoints:
        raise ValueError(
            "No matching checkpoints found between the two runs for the requested list."
        )

    pure_rl_data = [load_torso_height(pure_rl_files[checkpoint]) for checkpoint in common_checkpoints]
    mpc_data = [load_torso_height(mpc_files[checkpoint]) for checkpoint in common_checkpoints]

    print(f"Found {len(common_checkpoints)} matching checkpoints: {common_checkpoints}")
    print("\nPure RL torso-height summary:")
    for checkpoint, median, mean, pct_dragging in summarize_policy(common_checkpoints, pure_rl_data):
        print(
            f"  {checkpoint:>6d}: median={median:.3f} m, mean={mean:.3f} m, "
            f"below {DRAG_THRESHOLD_METERS:.2f} m={pct_dragging:.1f}%"
        )

    print("\nMPC-Injection torso-height summary:")
    for checkpoint, median, mean, pct_dragging in summarize_policy(common_checkpoints, mpc_data):
        print(
            f"  {checkpoint:>6d}: median={median:.3f} m, mean={mean:.3f} m, "
            f"below {DRAG_THRESHOLD_METERS:.2f} m={pct_dragging:.1f}%"
        )

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    draw_grouped_boxplots(ax, pure_rl_data, mpc_data, common_checkpoints)

    # ax.axhline(
    #     DRAG_THRESHOLD_METERS,
    #     color='#444444',
    #     linestyle='--',
    #     linewidth=1.8,
    #     alpha=0.9,
    # )
    # ax.text(
    #     0.01,
    #     DRAG_THRESHOLD_METERS + 0.015,
    #     'Torso-drag region',
    #     transform=ax.get_yaxis_transform(),
    #     fontsize=FONT_SIZE - 3,
    #     color='#444444',
    #     va='bottom',
    # )

    ax.set_xlabel('Training Checkpoint')
    ax.set_ylabel('Torso Height (m)')
    ax.set_title(
        'Walker Torso Height by Checkpoint:\nPure RL vs MPC-Injection',
        fontweight='bold',
    )
    all_values = np.concatenate([np.concatenate(pure_rl_data), np.concatenate(mpc_data)])
    positive_values = all_values[all_values > 0.0]
    if positive_values.size == 0:
        raise ValueError("Cannot use log y-scale because no torso-height values are positive.")
    ymin = max(float(np.min(positive_values)) * 0.8, 1e-4)
    ymax = float(np.max(positive_values)) * 1.05
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=PURE_RL_COLOR, edgecolor=PURE_RL_COLOR, label='0% MPC-Injection'),
        Patch(facecolor=MPC_INJECTION_COLOR, edgecolor=MPC_INJECTION_COLOR, label='25% MPC-Injection'),
        #Line2D(
        #    [0],
        #    [0],
        #    color='#444444',
        #    linestyle='--',
        #    linewidth=1.8,
        #    label=f'Drag threshold ({DRAG_THRESHOLD_METERS:.2f} m)',
        #),
    ]
    ax.legend(handles=legend_handles, loc='upper left')

    output_dir = Path(__file__).parent.parent / "plots/body_trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "walker_torso_height_boxplots_by_checkpoint.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved box plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    base_dir = Path(__file__).parent / "model_traj_data_walker"
    pure_rl_dir = base_dir / "walker-walk-SAC-MPC-20260107-112012-percentage-0pct"
    mpc_injection_dir = base_dir / "walker-walk-SAC-MPC-20260107-112659-percentage-25pct"

    checkpoints = [25000, 100000, 150000, 200000]

    plot_torso_height_boxplots(
        pure_rl_dir=pure_rl_dir,
        mpc_injection_dir=mpc_injection_dir,
        checkpoints=checkpoints,
    )
