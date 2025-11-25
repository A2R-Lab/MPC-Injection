#!/usr/bin/env python3
"""
Plotting script to visualize tensorboard logs from multiple experiment runs.
Reads rollout/ep_rew_mean from each experiment directory and plots them on a single graph.
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_percentage(dirname):
    """Extract the percentage value from directory name."""
    match = re.search(r'(\d+)pct', dirname)
    if match:
        return int(match.group(1))
    return None


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
                       env_pattern='cartpole-swingup', title_suffix='Cartpole Swingup'):
    """
    Plot training time from multiple experiment directories.
    
    Tries to load time/time_elapsed if available, otherwise calculates approximate
    time from FPS data.
    
    Args:
        base_dir: Base directory containing experiment subdirectories
        output_file: Output filename for the plot
        percentages_to_plot: List of percentages to plot (e.g., [0, 25, 50, 75, 100]).
                            If None, plots all available experiments.
        env_pattern: Pattern to match environment directories (e.g., 'cartpole-swingup', 'walker-walk')
        title_suffix: Suffix to add to plot title (e.g., 'Cartpole Swingup', 'Walker Walk')
    """
    # Find all experiment directories
    pattern = os.path.join(base_dir, f'{env_pattern}-*-percentage-*pct')
    exp_dirs = sorted(glob.glob(pattern))
    
    if not exp_dirs:
        print(f"No experiment directories found matching pattern: {pattern}")
        return
    
    print(f"\nLoading training time data from {len(exp_dirs)} experiment directories")
    
    # Collect data from all experiments
    data_dict = {}
    time_source = None  # Track which method worked
    
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
            
            data_dict[percentage] = {
                'steps': steps,
                'values': values_hours,
                'dirname': dirname
            }
            print(f"Loaded time data for {percentage}% ({len(steps)} data points, max: {values_hours[-1]:.2f} hours)")
    
    if not data_dict:
        print("No training time data loaded. Exiting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Sort by percentage for consistent ordering
    percentages = sorted(data_dict.keys())
    
    # Use a colormap with more distinct colors
    if len(percentages) <= 10:
        cmap = plt.cm.tab10
    elif len(percentages) <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv
    
    colors = [cmap(i / len(percentages)) for i in range(len(percentages))]
    
    for idx, percentage in enumerate(percentages):
        data = data_dict[percentage]
        plt.plot(data['steps'], data['values'], 
                label=f'{percentage}pct',
                color=colors[idx],
                linewidth=2.0,
                alpha=0.9)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Time (hours)', fontsize=12)
    title_text = f'Training Time vs Training Steps - {title_suffix}'
    if time_source == 'fps':
        title_text += ' (Estimated from FPS)'
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(base_dir), output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining time plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()


def plot_multiple_experiments(base_dir, output_file='experiment_comparison.png', percentages_to_plot=None, 
                             env_pattern='cartpole-swingup', title_suffix='Cartpole Swingup'):
    """
    Plot rollout/ep_rew_mean from multiple experiment directories.
    
    Args:
        base_dir: Base directory containing experiment subdirectories
        output_file: Output filename for the plot
        percentages_to_plot: List of percentages to plot (e.g., [0, 25, 50, 75, 100]).
                            If None, plots all available experiments.
        env_pattern: Pattern to match environment directories (e.g., 'cartpole-swingup', 'walker-walk')
        title_suffix: Suffix to add to plot title (e.g., 'Cartpole Swingup', 'Walker Walk')
    """
    # Find all experiment directories
    pattern = os.path.join(base_dir, f'{env_pattern}-*-percentage-*pct')
    exp_dirs = sorted(glob.glob(pattern))
    
    if not exp_dirs:
        print(f"No experiment directories found matching pattern: {pattern}")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Collect data from all experiments
    data_dict = {}
    
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
            data_dict[percentage] = {
                'steps': steps,
                'values': values,
                'dirname': dirname
            }
            print(f"Loaded data for {percentage}% ({len(steps)} data points)")
    
    if not data_dict:
        print("No data loaded. Exiting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Sort by percentage for consistent ordering
    percentages = sorted(data_dict.keys())
    
    # Use a colormap with more distinct colors
    # tab20 provides 20 distinct colors, good for multiple lines
    if len(percentages) <= 10:
        cmap = plt.cm.tab10
    elif len(percentages) <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv
    
    colors = [cmap(i / len(percentages)) for i in range(len(percentages))]
    
    for idx, percentage in enumerate(percentages):
        data = data_dict[percentage]
        plt.plot(data['steps'], data['values'], 
                label=f'{percentage}pct',
                color=colors[idx],
                linewidth=2.0,
                alpha=0.9)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Episode Reward Mean', fontsize=12)
    plt.title(f'Episode Reward Mean vs Training Steps - {title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(base_dir), output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show the plot
    plt.show()


def main():
    """Main function to run the plotting script."""
    
    # ========== CONFIGURATION - CHANGE THESE AS NEEDED ==========
    
    # Select base directory containing experiment runs
    # Uncomment the one you want to use:
    #base_dir = 'logs/SAC-MPC-walker-runs/1st_run/'
    base_dir = 'logs/TD3-MPC-walker-runs/5th_run/'
    
    # Select environment to plot
    # Option 1: Cartpole
    #env_pattern = 'cartpole-swingup'
    #title_suffix = 'Cartpole Swingup'
    #reward_output_file = 'cartpole_experiment_comparison.png'
    #time_output_file = 'cartpole_training_time_comparison.png'
    
    # Option 2: Walker (uncomment these 4 lines and comment out the cartpole lines above)
    env_pattern = 'walker-walk'
    title_suffix = 'Walker Walk'
    reward_output_file = 'walker_experiment_comparison.png'
    time_output_file = 'walker_training_time_comparison.png'
    
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
    
    # =============================================================
    
    # Convert to absolute path if relative
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_dir)
    
    print(f"Looking for experiments in: {base_dir}")
    print(f"Environment pattern: {env_pattern}")
    print(f"Percentages to plot: {percentages_to_plot if percentages_to_plot else 'All available'}")
    print(f"Generating plots: Rewards={plot_rewards}, Time={plot_time}")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return
    
    # Generate the reward plot
    if plot_rewards:
        print("\n" + "="*60)
        print("GENERATING REWARD PLOT")
        print("="*60)
        plot_multiple_experiments(base_dir, 
                                output_file=reward_output_file,
                                percentages_to_plot=percentages_to_plot,
                                env_pattern=env_pattern,
                                title_suffix=title_suffix)
    
    # Generate the training time plot
    if plot_time:
        print("\n" + "="*60)
        print("GENERATING TRAINING TIME PLOT")
        print("="*60)
        plot_training_time(base_dir,
                          output_file=time_output_file,
                          percentages_to_plot=percentages_to_plot,
                          env_pattern=env_pattern,
                          title_suffix=title_suffix)


if __name__ == '__main__':
    main()
