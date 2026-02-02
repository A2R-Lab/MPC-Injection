#!/usr/bin/env python3
"""
Plotting script to visualize tensorboard logs from multiple experiment runs with different seeds.
Reads rollout/ep_rew_mean from each experiment directory and plots them with mean + std ribbon.
For each percentage, aggregates data across multiple runs (seeds) and shows uncertainty band.
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_percentage(dirname):
    """Extract the percentage value from directory name."""
    match = re.search(r'(\d+)pct', dirname)
    if match:
        return int(match.group(1))
    return None


def find_run_directories(base_dir):
    """
    Find all run directories (e.g., 1st_run, 2nd_run, 3rd_run, etc.).
    
    Args:
        base_dir: Base directory containing run subdirectories
        
    Returns:
        List of run directory paths, sorted
    """
    # Look for directories matching patterns like "1st_run", "2nd_run", "run_1", "seed_1", etc.
    patterns = ['*_run', 'run*', 'seed*']
    run_dirs = []
    
    for pattern in patterns:
        matching = glob.glob(os.path.join(base_dir, pattern))
        run_dirs.extend([d for d in matching if os.path.isdir(d)])
    
    # Remove duplicates and sort
    run_dirs = sorted(list(set(run_dirs)))
    
    return run_dirs


def interpolate_to_common_grid(data_list, num_points=1000):
    """
    Interpolate multiple runs to a common step grid.
    
    Args:
        data_list: List of dicts, each with 'steps' and 'values' arrays
        num_points: Number of points in the common grid
        
    Returns:
        common_steps: Common step array
        interpolated_values: List of interpolated value arrays
    """
    if not data_list:
        return None, []
    
    # Find the common range of steps across all runs
    min_step = max(data['steps'].min() for data in data_list)
    max_step = min(data['steps'].max() for data in data_list)
    
    # Create common step grid
    common_steps = np.linspace(min_step, max_step, num_points)
    
    # Interpolate each run to the common grid
    interpolated_values = []
    for data in data_list:
        # Create interpolation function
        f = interp1d(data['steps'], data['values'], kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        # Interpolate to common grid
        interp_vals = f(common_steps)
        interpolated_values.append(interp_vals)
    
    return common_steps, interpolated_values


def load_tensorboard_data(log_dir, tag='rollout/ep_rew_mean'):
    """
    Load data from tensorboard event files.
    
    Args:
        log_dir: Path to the tensorboard log directory
        tag: The tag to extract from tensorboard logs
        
    Returns:
        steps: Array of step values
        values: Array of metric values
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Check if the tag exists
    if tag not in event_acc.Tags()['scalars']:
        return None, None
    
    # Get the scalar events
    events = event_acc.Scalars(tag)
    
    steps = [event.step for event in events]
    values = [event.value for event in events]
    
    return np.array(steps), np.array(values)


def calculate_time_from_fps(log_dir):
    """
    Calculate approximate training time from FPS data.
    
    Since time/time_elapsed may not be logged, we can approximate it from:
    - Steps at each logging point
    - FPS (frames per second) at each logging point
    
    Args:
        log_dir: Path to the tensorboard log directory
        
    Returns:
        steps: Array of step values
        time_hours: Array of approximate time values in hours
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get FPS data
    if 'time/fps' not in event_acc.Tags()['scalars']:
        return None, None
    
    fps_events = event_acc.Scalars('time/fps')
    
    steps = []
    cumulative_time = 0.0
    time_values = [0.0]  # Start at 0
    
    prev_step = 0
    for event in fps_events:
        current_step = event.step
        fps = event.value
        
        if fps > 0 and current_step > prev_step:
            # Calculate time elapsed since last logging point
            steps_elapsed = current_step - prev_step
            time_elapsed = steps_elapsed / fps  # seconds
            cumulative_time += time_elapsed
            
            steps.append(current_step)
            time_values.append(cumulative_time)
        
        prev_step = current_step
    
    if len(steps) == 0:
        return None, None
    
    # Convert to numpy arrays and time to hours
    steps_array = np.array(steps)
    time_hours = np.array(time_values[1:]) / 3600.0  # Convert to hours
    
    return steps_array, time_hours


def plot_training_time(base_dir, output_file='training_time_comparison.png', percentages_to_plot=None,
                       env_pattern='cartpole-swingup', title_suffix='Cartpole Swingup',
                       show_individual_runs=False, num_points=1000, std_scale=1.0):
    """
    Plot training time from multiple experiment directories across multiple seeds.
    Creates ribbon plots showing mean ± standard deviation/error across runs.
    
    Tries to load time/time_elapsed if available, otherwise calculates approximate
    time from FPS data.
    
    Args:
        base_dir: Base directory containing run subdirectories (e.g., 1st_run, 2nd_run, etc.)
        output_file: Output filename for the plot
        percentages_to_plot: List of percentages to plot (e.g., [0, 25, 50, 75, 100]).
                            If None, plots all available experiments.
        env_pattern: Pattern to match environment directories (e.g., 'cartpole-swingup', 'walker-walk')
        title_suffix: Suffix to add to plot title (e.g., 'Cartpole Swingup', 'Walker Walk')
        show_individual_runs: If True, plot individual runs as thin lines in addition to mean
        num_points: Number of points in the interpolation grid
        std_scale: Scale factor for standard deviation (1.0 for ±1σ, 2.0 for ±2σ, etc.)
    """
    # Find all run directories
    run_dirs = find_run_directories(base_dir)
    
    if not run_dirs:
        print(f"No run directories found in: {base_dir}")
        print("Looking for directories matching patterns: *_run, run*, seed*")
        return
    
    print(f"\nLoading training time data from {len(run_dirs)} run directories")
    
    # Collect data from all runs, organized by percentage
    percentage_data = {}
    time_source = None  # Track which method worked
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        
        # Find experiment directories in this run
        pattern = os.path.join(run_dir, f'{env_pattern}-*-percentage-*pct')
        exp_dirs = glob.glob(pattern)
        
        if not exp_dirs:
            print(f"Warning: No experiment directories found in {run_name}")
            continue
        
        for exp_dir in exp_dirs:
            # Extract percentage from directory name
            dirname = os.path.basename(exp_dir)
            percentage = extract_percentage(dirname)
            
            if percentage is None:
                print(f"Warning: Could not extract percentage from {dirname}")
                continue
            
            # Skip if not in the list of percentages to plot
            if percentages_to_plot is not None and percentage not in percentages_to_plot:
                continue
            
            # Find tensorboard directory - check for both SAC_1 and TD3_1
            tb_base = os.path.join(exp_dir, 'tensorboard')
            tb_dir = None
            
            # Try SAC_1 first, then TD3_1
            for subdir in ['SAC_1', 'TD3_1']:
                potential_dir = os.path.join(tb_base, subdir)
                if os.path.exists(potential_dir):
                    tb_dir = potential_dir
                    break
            
            if tb_dir is None:
                print(f"Warning: Tensorboard directory not found in: {tb_base}")
                continue
            
            # Try to load time/time_elapsed first
            steps, values = load_tensorboard_data(tb_dir, tag='time/time_elapsed')
            
            # If not available, calculate from FPS
            if steps is None or values is None:
                if time_source is None:
                    print("Note: 'time/time_elapsed' not found, calculating approximate time from FPS data...")
                    time_source = 'fps'
                steps, values = calculate_time_from_fps(tb_dir)
            elif time_source is None:
                time_source = 'direct'
                print("Note: Using 'time/time_elapsed' from TensorBoard logs")
            
            if steps is not None and values is not None:
                # Convert to hours if needed (fps calculation already returns hours)
                if time_source == 'direct':
                    values_hours = values / 3600.0
                else:
                    values_hours = values
                
                # Initialize list for this percentage if not exists
                if percentage not in percentage_data:
                    percentage_data[percentage] = []
                
                percentage_data[percentage].append({
                    'steps': steps,
                    'values': values_hours,
                    'run_name': run_name
                })
                print(f"Loaded time data for {percentage}% from {run_name} ({len(steps)} data points, max: {values_hours[-1]:.2f} hours)")
    
    if not percentage_data:
        print("No training time data loaded. Exiting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Sort by percentage for consistent ordering
    percentages = sorted(percentage_data.keys())
    
    # Use a colormap with more distinct colors
    if len(percentages) <= 10:
        cmap = plt.cm.tab10
    elif len(percentages) <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv
    
    colors = [cmap(i / len(percentages)) for i in range(len(percentages))]
    
    print(f"\nGenerating time plots for {len(percentages)} percentages...")
    
    for idx, percentage in enumerate(percentages):
        runs = percentage_data[percentage]
        num_runs = len(runs)
        
        print(f"\n{percentage}pct: {num_runs} run(s)")
        
        if num_runs == 0:
            continue
        
        # Interpolate all runs to a common grid
        common_steps, interpolated_values = interpolate_to_common_grid(runs, num_points=num_points)
        
        if common_steps is None:
            print(f"  Warning: Could not interpolate data for {percentage}%")
            continue
        
        # Convert to numpy array for easier computation
        interpolated_array = np.array(interpolated_values)
        
        # Calculate mean and standard deviation
        mean_values = np.mean(interpolated_array, axis=0)
        std_values = np.std(interpolated_array, axis=0)
        
        # Plot individual runs if requested
        if show_individual_runs and num_runs > 1:
            for i, run_vals in enumerate(interpolated_values):
                plt.plot(common_steps, run_vals,
                        color=colors[idx],
                        alpha=0.2,
                        linewidth=0.5)
        
        # Plot mean line
        line_label = f'{percentage}pct (n={num_runs})'
        plt.plot(common_steps, mean_values,
                label=line_label,
                color=colors[idx],
                linewidth=2.5,
                alpha=0.9)
        
        # Plot uncertainty band (only if we have more than one run)
        if num_runs > 1:
            plt.fill_between(common_steps,
                           mean_values - std_scale * std_values,
                           mean_values + std_scale * std_values,
                           color=colors[idx],
                           alpha=0.2)
            print(f"  Final mean time ± std: {mean_values[-1]:.2f} ± {std_values[-1]:.2f} hours")
        else:
            print(f"  Single run - no uncertainty band")
            print(f"  Final time: {mean_values[-1]:.2f} hours")
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Time (hours)', fontsize=12)
    
    # Add std_scale info to title if not default
    title_text = f'Training Time vs Training Steps - {title_suffix}'
    if time_source == 'fps':
        title_text += ' (Estimated from FPS'
        if std_scale != 1.0:
            title_text += f', ±{std_scale}σ)'
        else:
            title_text += ', ±1σ)'
    else:
        if std_scale != 1.0:
            title_text += f' (±{std_scale}σ)'
        else:
            title_text += ' (±1σ)'
    
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(base_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining time plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()


def plot_multiple_experiments(base_dir, output_file='experiment_comparison.png', percentages_to_plot=None, 
                             env_pattern='cartpole-swingup', title_suffix='Cartpole Swingup',
                             show_individual_runs=False, num_points=1000, std_scale=1.0):
    """
    Plot rollout/ep_rew_mean from multiple experiment directories across multiple seeds.
    Creates ribbon plots showing mean ± standard deviation/error across runs.
    
    Args:
        base_dir: Base directory containing run subdirectories (e.g., 1st_run, 2nd_run, etc.)
        output_file: Output filename for the plot
        percentages_to_plot: List of percentages to plot (e.g., [0, 25, 50, 75, 100]).
                            If None, plots all available experiments.
        env_pattern: Pattern to match environment directories (e.g., 'cartpole-swingup', 'walker-walk')
        title_suffix: Suffix to add to plot title (e.g., 'Cartpole Swingup', 'Walker Walk')
        show_individual_runs: If True, plot individual runs as thin lines in addition to mean
        num_points: Number of points in the interpolation grid
        std_scale: Scale factor for standard deviation (1.0 for ±1σ, 2.0 for ±2σ, etc.)
    """
    # Find all run directories
    run_dirs = find_run_directories(base_dir)
    
    if not run_dirs:
        print(f"No run directories found in: {base_dir}")
        print("Looking for directories matching patterns: *_run, run*, seed*")
        return
    
    print(f"Found {len(run_dirs)} run directories:")
    for run_dir in run_dirs:
        print(f"  - {os.path.basename(run_dir)}")
    
    # Collect data from all runs, organized by percentage
    # Structure: percentage_data[percentage][run_idx] = {'steps': ..., 'values': ...}
    percentage_data = {}
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        
        # Find experiment directories in this run
        pattern = os.path.join(run_dir, f'{env_pattern}-*-percentage-*pct')
        exp_dirs = glob.glob(pattern)
        
        if not exp_dirs:
            print(f"Warning: No experiment directories found in {run_name}")
            continue
        
        for exp_dir in exp_dirs:
            # Extract percentage from directory name
            dirname = os.path.basename(exp_dir)
            percentage = extract_percentage(dirname)
            
            if percentage is None:
                print(f"Warning: Could not extract percentage from {dirname}")
                continue
            
            # Skip if not in the list of percentages to plot
            if percentages_to_plot is not None and percentage not in percentages_to_plot:
                continue
            
            # Find tensorboard directory - check for both SAC_1 and TD3_1
            tb_base = os.path.join(exp_dir, 'tensorboard')
            tb_dir = None
            
            # Try SAC_1 first, then TD3_1
            for subdir in ['SAC_1', 'TD3_1']:
                potential_dir = os.path.join(tb_base, subdir)
                if os.path.exists(potential_dir):
                    tb_dir = potential_dir
                    break
            
            if tb_dir is None:
                print(f"Warning: Tensorboard directory not found in: {tb_base}")
                continue
            
            # Load data
            steps, values = load_tensorboard_data(tb_dir)
            
            if steps is not None and values is not None:
                # Initialize list for this percentage if not exists
                if percentage not in percentage_data:
                    percentage_data[percentage] = []
                
                percentage_data[percentage].append({
                    'steps': steps,
                    'values': values,
                    'run_name': run_name
                })
                print(f"Loaded data for {percentage}% from {run_name} ({len(steps)} data points)")
    
    if not percentage_data:
        print("No data loaded. Exiting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Sort by percentage for consistent ordering
    percentages = sorted(percentage_data.keys())
    
    # Use a colormap with more distinct colors
    if len(percentages) <= 10:
        cmap = plt.cm.tab10
    elif len(percentages) <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv
    
    colors = [cmap(i / len(percentages)) for i in range(len(percentages))]
    
    print(f"\nGenerating plots for {len(percentages)} percentages...")
    
    for idx, percentage in enumerate(percentages):
        runs = percentage_data[percentage]
        num_runs = len(runs)
        
        print(f"\n{percentage}pct: {num_runs} run(s)")
        
        if num_runs == 0:
            continue
        
        # Interpolate all runs to a common grid
        common_steps, interpolated_values = interpolate_to_common_grid(runs, num_points=num_points)
        
        if common_steps is None:
            print(f"  Warning: Could not interpolate data for {percentage}%")
            continue
        
        # Convert to numpy array for easier computation
        interpolated_array = np.array(interpolated_values)  # Shape: (num_runs, num_points)
        
        # Calculate mean and standard deviation
        mean_values = np.mean(interpolated_array, axis=0)
        std_values = np.std(interpolated_array, axis=0)
        
        # Plot individual runs if requested
        if show_individual_runs and num_runs > 1:
            for i, run_vals in enumerate(interpolated_values):
                plt.plot(common_steps, run_vals,
                        color=colors[idx],
                        alpha=0.2,
                        linewidth=0.5)
        
        # Plot mean line
        line_label = f'{percentage}pct (n={num_runs})'
        plt.plot(common_steps, mean_values,
                label=line_label,
                color=colors[idx],
                linewidth=2.5,
                alpha=0.9)
        
        # Plot uncertainty band (only if we have more than one run)
        if num_runs > 1:
            plt.fill_between(common_steps,
                           mean_values - std_scale * std_values,
                           mean_values + std_scale * std_values,
                           color=colors[idx],
                           alpha=0.2)
            print(f"  Mean reward range: [{mean_values.min():.2f}, {mean_values.max():.2f}]")
            print(f"  Final mean ± std: {mean_values[-1]:.2f} ± {std_values[-1]:.2f}")
        else:
            print(f"  Single run - no uncertainty band")
            print(f"  Reward range: [{mean_values.min():.2f}, {mean_values.max():.2f}]")
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Episode Reward Mean', fontsize=12)
    
    # Add std_scale info to title if not default
    title_text = f'Episode Reward Mean vs Training Steps - {title_suffix}'
    if std_scale != 1.0:
        title_text += f' (±{std_scale}σ)'
    else:
        title_text += ' (±1σ)'
    
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(base_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show the plot
    plt.show()


def main():
    """Main function to run the plotting script."""
    
    # ========== CONFIGURATION - CHANGE THESE AS NEEDED ==========
    
    # Select base directory containing run subdirectories (1st_run, 2nd_run, etc.)
    # This should point to the directory containing multiple seeds/runs
    # Uncomment the one you want to use:
    #base_dir = '../logs/SAC-MPC-walker-runs/'
    #base_dir = '../logs/TD3-MPC-walker-runs/'
    base_dir = '../logs/SAC-MPC-walker-velocity_only_reward/'
    
    # Select environment to plot
    # Option 1: Cartpole
    #env_pattern = 'cartpole-swingup'
    #title_suffix = 'Cartpole Swingup'
    #reward_output_file = 'cartpole_experiment_comparison_ribbon.png'
    #time_output_file = 'cartpole_training_time_comparison_ribbon.png'
    
    # Option 2: Walker (uncomment these 4 lines and comment out the cartpole lines above)
    env_pattern = 'walker-walk'
    title_suffix = 'Walker Walk - Velocity Only Reward'
    reward_output_file = 'walker_experiment_comparison_ribbon.png'
    time_output_file = 'walker_training_time_comparison_ribbon.png'
    
    # Select which percentages to plot
    # Option 1: Plot all available experiments (set to None)
    #percentages_to_plot = None
    
    # Option 2: Plot specific percentages
    percentages_to_plot = [0, 25, 50, 75, 100]
    #percentages_to_plot = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #percentages_to_plot = [0, 50, 100]
    
    # Select which plots to generate
    plot_rewards = True      # Plot episode reward mean
    plot_time = True         # Plot training time
    
    # Additional options for ribbon plots
    show_individual_runs = False  # Show individual runs as thin lines
    num_points = 1000             # Number of points in interpolation grid
    std_scale = 1.0               # Scale for std deviation (1.0 = ±1σ, 2.0 = ±2σ)
    
    # =============================================================
    
    # Convert to absolute path if relative
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_dir)
    
    print("="*70)
    print("MULTI-SEED RIBBON PLOT GENERATION")
    print("="*70)
    print(f"Base directory: {base_dir}")
    print(f"Environment pattern: {env_pattern}")
    print(f"Percentages to plot: {percentages_to_plot if percentages_to_plot else 'All available'}")
    print(f"Generating plots: Rewards={plot_rewards}, Time={plot_time}")
    print(f"Show individual runs: {show_individual_runs}")
    print(f"Standard deviation scale: ±{std_scale}σ")
    print("="*70)
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return
    
    # Generate the reward plot
    if plot_rewards:
        print("\n" + "="*60)
        print("GENERATING REWARD RIBBON PLOT")
        print("="*60)
        plot_multiple_experiments(base_dir, 
                                output_file=reward_output_file,
                                percentages_to_plot=percentages_to_plot,
                                env_pattern=env_pattern,
                                title_suffix=title_suffix,
                                show_individual_runs=show_individual_runs,
                                num_points=num_points,
                                std_scale=std_scale)
    
    # Generate the training time plot
    if plot_time:
        print("\n" + "="*60)
        print("GENERATING TRAINING TIME RIBBON PLOT")
        print("="*60)
        plot_training_time(base_dir,
                          output_file=time_output_file,
                          percentages_to_plot=percentages_to_plot,
                          env_pattern=env_pattern,
                          title_suffix=title_suffix,
                          show_individual_runs=show_individual_runs,
                          num_points=num_points,
                          std_scale=std_scale)


if __name__ == '__main__':
    main()
