#!/usr/bin/env python3
"""
Plot tensorboard logs from multiple quadruped experiment runs with different seeds.
Reads rollout/ep_rew_mean from each experiment directory and plots them with mean + std ribbon.

Supports both:
- nested layouts like seed/run folders that contain experiment directories
- flat layouts where experiment directories live directly under base_dir and encode seed in the name
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FONT_SIZE = 18
RBG_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "figure.titlesize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 2,
        "legend.fontsize": FONT_SIZE,
    }
)


def extract_percentage(dirname):
    """Extract the percentage value from directory name."""
    match = re.search(r"(\d+)pct", dirname)
    if match:
        return int(match.group(1))
    return None


def extract_seed(dirname):
    """Extract the seed identifier from directory name."""
    match = re.search(r"seed(\d+)", dirname)
    if match:
        return f"seed{match.group(1)}"
    return None


def find_run_directories(base_dir):
    """
    Find run directories for nested layouts.

    Args:
        base_dir: Base directory containing run subdirectories

    Returns:
        List of run directory paths, sorted
    """
    patterns = ["*_run", "run*", "seed*"]
    run_dirs = []

    for pattern in patterns:
        matching = glob.glob(os.path.join(base_dir, pattern))
        run_dirs.extend([d for d in matching if os.path.isdir(d)])

    return sorted(list(set(run_dirs)))


def find_experiment_directories(base_dir, env_pattern):
    """
    Find experiment directories for either nested or flat log layouts.

    Returns:
        List of tuples: (exp_dir, run_name)
    """
    flat_pattern = os.path.join(base_dir, f"{env_pattern}-*-percentage-*pct*")
    flat_exp_dirs = sorted([d for d in glob.glob(flat_pattern) if os.path.isdir(d)])
    if flat_exp_dirs:
        experiments = []
        for exp_dir in flat_exp_dirs:
            dirname = os.path.basename(exp_dir)
            run_name = extract_seed(dirname) or dirname
            experiments.append((exp_dir, run_name))
        return experiments

    experiments = []
    for run_dir in find_run_directories(base_dir):
        run_name = os.path.basename(run_dir)
        nested_pattern = os.path.join(run_dir, f"{env_pattern}-*-percentage-*pct*")
        exp_dirs = sorted([d for d in glob.glob(nested_pattern) if os.path.isdir(d)])
        for exp_dir in exp_dirs:
            experiments.append((exp_dir, run_name))
    return experiments


def interpolate_to_common_grid(data_list, num_points=1000):
    """Interpolate multiple runs to a common step grid."""
    if not data_list:
        return None, []

    min_step = max(data["steps"].min() for data in data_list)
    max_step = min(data["steps"].max() for data in data_list)

    common_steps = np.linspace(min_step, max_step, num_points)

    interpolated_values = []
    for data in data_list:
        interpolation_fn = interp1d(
            data["steps"],
            data["values"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated_values.append(interpolation_fn(common_steps))

    return common_steps, interpolated_values


def load_tensorboard_data(log_dir, tag="rollout/ep_rew_mean"):
    """Load scalar data from tensorboard event files."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if tag not in event_acc.Tags()["scalars"]:
        return None, None

    events = event_acc.Scalars(tag)
    steps = np.array([event.step for event in events])
    values = np.array([event.value for event in events])
    return steps, values


def calculate_time_from_fps(log_dir):
    """Calculate approximate training time from time/fps tensorboard data."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if "time/fps" not in event_acc.Tags()["scalars"]:
        return None, None

    fps_events = event_acc.Scalars("time/fps")

    steps = []
    cumulative_time = 0.0
    time_values = [0.0]

    prev_step = 0
    for event in fps_events:
        current_step = event.step
        fps = event.value

        if fps > 0 and current_step > prev_step:
            steps_elapsed = current_step - prev_step
            cumulative_time += steps_elapsed / fps
            steps.append(current_step)
            time_values.append(cumulative_time)

        prev_step = current_step

    if not steps:
        return None, None

    return np.array(steps), np.array(time_values[1:]) / 3600.0


def get_tensorboard_dir(exp_dir):
    """Find the tensorboard subdirectory used by the run."""
    tb_base = os.path.join(exp_dir, "tensorboard")
    for subdir in ["SAC_1", "TD3_1"]:
        candidate = os.path.join(tb_base, subdir)
        if os.path.exists(candidate):
            return candidate
    return None


def plot_training_time(
    base_dir,
    output_file="training_time_comparison.png",
    percentages_to_plot=None,
    env_pattern="quadruped-velocity_tracking",
    title_suffix="Quadruped Velocity Tracking",
    show_individual_runs=False,
    num_points=1000,
    std_scale=1.0,
):
    """Plot training time across percentages and seeds."""
    experiment_dirs = find_experiment_directories(base_dir, env_pattern)

    if not experiment_dirs:
        print(f"No experiment directories found in: {base_dir}")
        return

    print(f"\nLoading training time data from {len(experiment_dirs)} experiment directories")

    percentage_data = {}

    for exp_dir, run_name in experiment_dirs:
        dirname = os.path.basename(exp_dir)
        percentage = extract_percentage(dirname)

        if percentage is None:
            print(f"Warning: Could not extract percentage from {dirname}")
            continue

        if percentages_to_plot is not None and percentage not in percentages_to_plot:
            continue

        tb_dir = get_tensorboard_dir(exp_dir)
        if tb_dir is None:
            print(f"Warning: Tensorboard directory not found in: {os.path.join(exp_dir, 'tensorboard')}")
            continue

        steps, time_hours = load_tensorboard_data(tb_dir, tag="time/time_elapsed")
        if steps is not None and time_hours is not None:
            time_hours = time_hours / 3600.0
            time_source = "time/time_elapsed"
        else:
            steps, time_hours = calculate_time_from_fps(tb_dir)
            time_source = "time/fps (calculated)"

        if steps is None or time_hours is None:
            print(f"Warning: Could not load time data from {run_name} ({dirname})")
            continue

        percentage_data.setdefault(percentage, []).append(
            {"steps": steps, "values": time_hours, "run_name": run_name}
        )
        print(
            f"Loaded training time for {percentage}% from {run_name} "
            f"using {time_source} ({len(steps)} data points)"
        )

    if not percentage_data:
        print("No training time data loaded. Exiting.")
        return

    plt.figure(figsize=(12, 8))

    percentages = sorted(percentage_data.keys())
    if len(percentages) <= len(RBG_COLORS):
        colors = RBG_COLORS[: len(percentages)]
    else:
        cmap = plt.cm.tab20 if len(percentages) <= 20 else plt.cm.hsv
        colors = [cmap(i / len(percentages)) for i in range(len(percentages))]

    print(f"\nGenerating training time plots for {len(percentages)} percentages...")

    for idx, percentage in enumerate(percentages):
        runs = percentage_data[percentage]
        num_runs = len(runs)
        print(f"\n{percentage}pct: {num_runs} run(s)")

        common_steps, interpolated_values = interpolate_to_common_grid(runs, num_points=num_points)
        if common_steps is None:
            print(f"  Warning: Could not interpolate time data for {percentage}%")
            continue

        interpolated_array = np.array(interpolated_values)
        mean_values = np.mean(interpolated_array, axis=0)
        std_values = np.std(interpolated_array, axis=0)

        if show_individual_runs and num_runs > 1:
            for run_vals in interpolated_values:
                plt.plot(common_steps, run_vals, color=colors[idx], alpha=0.2, linewidth=0.5)

        plt.plot(
            common_steps,
            mean_values,
            label=f"{percentage}pct (n={num_runs})",
            color=colors[idx],
            linewidth=2.5,
            alpha=0.9,
        )

        if num_runs > 1:
            plt.fill_between(
                common_steps,
                mean_values - std_scale * std_values,
                mean_values + std_scale * std_values,
                color=colors[idx],
                alpha=0.2,
            )

    plt.xlabel("Training Steps", fontsize=FONT_SIZE)
    plt.ylabel("Training Time (Hours)", fontsize=FONT_SIZE)

    title_text = f"Training Time - {title_suffix}"
    title_text += f" (±{std_scale}σ)" if std_scale != 1.0 else " (±1σ)"
    plt.title(title_text, fontsize=FONT_SIZE + 2, fontweight="bold")
    plt.legend(loc="best", fontsize=FONT_SIZE - 2, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(base_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nTraining time plot saved to: {output_path}")
    plt.show()


def plot_multiple_experiments(
    base_dir,
    output_file="experiment_comparison.png",
    percentages_to_plot=None,
    env_pattern="quadruped-velocity_tracking",
    title_suffix="Quadruped Velocity Tracking",
    show_individual_runs=False,
    num_points=1000,
    std_scale=1.0,
):
    """Plot rollout/ep_rew_mean from multiple experiment directories across multiple seeds."""
    experiment_dirs = find_experiment_directories(base_dir, env_pattern)

    if not experiment_dirs:
        print(f"No experiment directories found in: {base_dir}")
        return

    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir, run_name in experiment_dirs:
        print(f"  - {run_name}: {os.path.basename(exp_dir)}")

    percentage_data = {}

    for exp_dir, run_name in experiment_dirs:
        dirname = os.path.basename(exp_dir)
        percentage = extract_percentage(dirname)

        if percentage is None:
            print(f"Warning: Could not extract percentage from {dirname}")
            continue

        if percentages_to_plot is not None and percentage not in percentages_to_plot:
            continue

        tb_dir = get_tensorboard_dir(exp_dir)
        if tb_dir is None:
            print(f"Warning: Tensorboard directory not found in: {os.path.join(exp_dir, 'tensorboard')}")
            continue

        steps, values = load_tensorboard_data(tb_dir)
        if steps is None or values is None:
            print(f"Warning: Could not load reward data from {run_name} ({dirname})")
            continue

        percentage_data.setdefault(percentage, []).append(
            {"steps": steps, "values": values, "run_name": run_name}
        )
        print(f"Loaded data for {percentage}% from {run_name} ({len(steps)} data points)")

    if not percentage_data:
        print("No data loaded. Exiting.")
        return

    plt.figure(figsize=(12, 8))

    percentages = sorted(percentage_data.keys())
    if len(percentages) <= len(RBG_COLORS):
        colors = RBG_COLORS[: len(percentages)]
    else:
        cmap = plt.cm.tab20 if len(percentages) <= 20 else plt.cm.hsv
        colors = [cmap(i / len(percentages)) for i in range(len(percentages))]

    print(f"\nGenerating plots for {len(percentages)} percentages...")

    for idx, percentage in enumerate(percentages):
        runs = percentage_data[percentage]
        num_runs = len(runs)
        print(f"\n{percentage}pct: {num_runs} run(s)")

        common_steps, interpolated_values = interpolate_to_common_grid(runs, num_points=num_points)
        if common_steps is None:
            print(f"  Warning: Could not interpolate data for {percentage}%")
            continue

        interpolated_array = np.array(interpolated_values)
        mean_values = np.mean(interpolated_array, axis=0)
        std_values = np.std(interpolated_array, axis=0)

        if show_individual_runs and num_runs > 1:
            for run_vals in interpolated_values:
                plt.plot(
                    common_steps,
                    run_vals,
                    color=colors[idx],
                    alpha=0.2,
                    linewidth=0.5,
                )

        plt.plot(
            common_steps,
            mean_values,
            label=f"{percentage}pct mean (n={num_runs})",
            color=colors[idx],
            linewidth=2.5,
            alpha=0.9,
        )

        if num_runs > 1:
            plt.fill_between(
                common_steps,
                mean_values - std_scale * std_values,
                mean_values + std_scale * std_values,
                color=colors[idx],
                alpha=0.2,
            )
            print(f"  Mean reward range: [{mean_values.min():.2f}, {mean_values.max():.2f}]")
            print(f"  Final mean ± std: {mean_values[-1]:.2f} ± {std_values[-1]:.2f}")
        else:
            print("  Single run - no uncertainty band")
            print(f"  Reward range: [{mean_values.min():.2f}, {mean_values.max():.2f}]")

    plt.xlabel("Training Steps", fontsize=FONT_SIZE)
    plt.ylabel("Episode Reward Mean", fontsize=FONT_SIZE)

    title_text = f"Training Performance - {title_suffix}"
    title_text += f" (±{std_scale}σ)" if std_scale != 1.0 else " (±1σ)"
    plt.title(title_text, fontsize=FONT_SIZE + 2, fontweight="bold")
    plt.legend(loc="best", fontsize=FONT_SIZE - 4, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(base_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def main():
    """Main function to run the plotting script."""
    base_dir = "../logs/SAC-MPC-quadruped_x_vel_only_runs_min_rwrd"

    env_pattern = "quadruped-velocity_tracking"
    title_suffix = "Quadruped Velocity Tracking - Velocity Only Reward"
    reward_output_file = "quadruped_experiment_comparison_ribbon.png"
    time_output_file = "quadruped_training_time_comparison_ribbon.png"

    percentages_to_plot = [0, 25, 50]

    plot_rewards = True
    plot_time = False

    show_individual_runs = False
    num_points = 1000
    std_scale = 1.0

    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_dir)

    print("=" * 70)
    print("MULTI-SEED RIBBON PLOT GENERATION")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Environment pattern: {env_pattern}")
    print(f"Percentages to plot: {percentages_to_plot if percentages_to_plot else 'All available'}")
    print(f"Generating plots: Rewards={plot_rewards}, Time={plot_time}")
    print(f"Show individual runs: {show_individual_runs}")
    print(f"Standard deviation scale: ±{std_scale}σ")
    print("=" * 70)

    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return

    if plot_rewards:
        print("\n" + "=" * 60)
        print("GENERATING REWARD RIBBON PLOT")
        print("=" * 60)
        plot_multiple_experiments(
            base_dir,
            output_file=reward_output_file,
            percentages_to_plot=percentages_to_plot,
            env_pattern=env_pattern,
            title_suffix=title_suffix,
            show_individual_runs=show_individual_runs,
            num_points=num_points,
            std_scale=std_scale,
        )

    if plot_time:
        print("\n" + "=" * 60)
        print("GENERATING TRAINING TIME RIBBON PLOT")
        print("=" * 60)
        plot_training_time(
            base_dir,
            output_file=time_output_file,
            percentages_to_plot=percentages_to_plot,
            env_pattern=env_pattern,
            title_suffix=title_suffix,
            show_individual_runs=show_individual_runs,
            num_points=num_points,
            std_scale=std_scale,
        )


if __name__ == "__main__":
    main()
