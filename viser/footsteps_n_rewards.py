"""
Script to plot the footsteps and rewards from the walker environment.
"""

import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

class FootStepAndRewardPlotter:
    """
    Visualize the footsteps of the walker and plot the rewards over time for a single
    trajectory.
    """
    def __init__(
            self,
        ):
        """
        Initializes the plotter
        """
        self.trajectory_data = None
    
    def load_trajectory(self, npz_path: str, time_range: str = None):
        """
        Loads trajectory data from .npz file
        
        Args:
            npz_path (str): Path to the .npz file containing trajectory data
            time_range (str): Optional range of timesteps to load (e.g., "199:600")
        """
        data = np.load(npz_path, allow_pickle=True)

        self.trajectory_data = data
        self.left_footsteps  = data['contact_left_foot']
        self.right_footsteps = data['contact_right_foot']
        self.rewards         = data['rewards']

        # Apply time range slicing if specified
        if time_range:
            start, end = self._parse_range(time_range)
            self.left_footsteps = self.left_footsteps[start:end]
            self.right_footsteps = self.right_footsteps[start:end]
            self.rewards = self.rewards[start:end]
            print(f"Loaded timesteps {start}:{end} (total: {len(self.rewards)} steps)")
        else:
            print(f"Loaded full trajectory (total: {len(self.rewards)} steps)")

        print(f"Available data: {list(data.keys())}")

        return data
    
    def _parse_range(self, range_str: str):
        """
        Parse range string like "199:600" into start and end indices
        
        Args:
            range_str (str): Range string in format "start:end"
            
        Returns:
            tuple: (start, end) indices
        """
        parts = range_str.split(':')
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if len(parts) > 1 and parts[1] else None
        return start, end
    
    def plot_footsteps_and_rewards(self):
        """
        Plots the footsteps and rewards as stacked raster plots
        """
        if self.trajectory_data is None:
            print("No trajectory data loaded. Please load trajectory first.")
            return
        
        num_steps = len(self.rewards)
        time_steps = np.arange(num_steps)
        
        # Create figure with 3 subplots stacked vertically
        # Reduced height from 6 to 3 to make plots more vertically compact
        # Also set gridspec_kw to control height ratios of subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 3), sharex=True, 
                                gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.15})
        
        # Plot Left Foot contact (LF)
        ax_lf = axes[0]
        for t in time_steps:
            if self.left_footsteps[t]:
                ax_lf.axvline(x=t, color='black', linewidth=0.8)
        ax_lf.set_ylabel('LF', fontsize=12, rotation=0, labelpad=20)
        ax_lf.set_ylim(0, 1)
        ax_lf.set_yticks([])
        ax_lf.spines['left'].set_visible(False)
        ax_lf.spines['right'].set_visible(False)
        ax_lf.spines['top'].set_visible(False)
        
        # Plot Right Foot contact (RF)
        ax_rf = axes[1]
        for t in time_steps:
            if self.right_footsteps[t]:
                ax_rf.axvline(x=t, color='black', linewidth=0.8)
        ax_rf.set_ylabel('RF', fontsize=12, rotation=0, labelpad=20)
        ax_rf.set_ylim(0, 1)
        ax_rf.set_yticks([])
        ax_rf.spines['left'].set_visible(False)
        ax_rf.spines['right'].set_visible(False)
        ax_rf.spines['top'].set_visible(False)
        
        # Plot Rewards (R)
        ax_r = axes[2]
        # Reshape rewards to 2D for imshow (1 row, N columns)
        rewards_2d = self.rewards.reshape(1, -1)
        # Set explicit vmin/vmax to ensure color gradient spans the actual reward range
        # This prevents the plot from being uniformly red or blue
        reward_min, reward_max = self.rewards.min(), self.rewards.max()
        im = ax_r.imshow(rewards_2d, aspect='auto', cmap='coolwarm', 
                         interpolation='nearest', extent=[0, num_steps, 0, 1],
                         vmin=reward_min, vmax=reward_max)
        ax_r.set_ylabel('Reward', fontsize=12, rotation=0, labelpad=20)
        print(f"Reward range: [{reward_min:.4f}, {reward_max:.4f}]")
        ax_r.set_yticks([])
        ax_r.set_xlabel('Timestep', fontsize=12)
        ax_r.spines['left'].set_visible(False)
        ax_r.spines['right'].set_visible(False)
        ax_r.spines['top'].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
        
        return fig
    

def main():
    parser = argparse.ArgumentParser(
        description="Plot footsteps and rewards from walker trajectory data",
        epilog="""Examples:
        python footsteps_n_rewards.py --trajectory path/to/trajectory_data.npz
        """
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        help="Path to trajectory .npz file",
        default="body_trajs/model_traj_data/walker-walk-SAC-MPC-20260107-113507-percentage-50pct/trajectories_step_500000.npz"
    )
    
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Range of timesteps to plot (e.g., '199:600' or ':500' or '100:')"
    )

    args = parser.parse_args()
    traj_path = args.trajectory
    time_range = args.range

    plotter = FootStepAndRewardPlotter()

    plotter.load_trajectory(traj_path, time_range=time_range)
    plotter.plot_footsteps_and_rewards()


if __name__ == "__main__":
    main()