"""
Script to plot the footsteps from the walker environment.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import re

FONT_SIZE = 18

class FootStepPlotter:
    """
    Visualize the footsteps of the walker over time for a single
    trajectory.
    """
    def __init__(
            self,
        ):
        """
        Initializes the plotter
        """
        self.trajectory_data = None
        self.last_fig = None
        self.last_trajectory_path = None
        self.left_footsteps = None
        self.right_footsteps = None
    
    def load_trajectory(self, npz_path: str, time_range: str = None):
        """
        Loads trajectory data from .npz file
        
        Args:
            npz_path (str): Path to the .npz file containing trajectory data
            time_range (str): Optional range of timesteps to load (e.g., "199:600")
        """
        data = np.load(npz_path, allow_pickle=True)

        self.last_trajectory_path = npz_path

        self.trajectory_data = data
        self.left_footsteps  = data['contact_left_foot']
        self.right_footsteps = data['contact_right_foot']

        # Apply time range slicing if specified
        if time_range:
            start, end = self._parse_range(time_range)
            self.left_footsteps = self.left_footsteps[start:end]
            self.right_footsteps = self.right_footsteps[start:end]
            print(f"Loaded timesteps {start}:{end} (total: {len(self.left_footsteps)} steps)")
        else:
            print(f"Loaded full trajectory (total: {len(self.left_footsteps)} steps)")

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
    
    def plot_footsteps(self):
        """
        Plots the footsteps as raster plots
        """
        if self.trajectory_data is None:
            print("No trajectory data loaded. Please load trajectory first.")
            return
        
        num_steps = min(len(self.left_footsteps), len(self.right_footsteps))
        time_steps = np.arange(num_steps)
        
        # Create figure with two stacked raster plots, one per foot.
        fig, axes = plt.subplots(
            2, 1,
            figsize=(12, 2.0),
            sharex=True,
            gridspec_kw={'height_ratios': [1, 1]},
            layout="constrained",
        )
        
        # Plot Left Foot contact (LF)
        ax_lf = axes[0]
        for t in time_steps:
            if self.left_footsteps[t]:
                ax_lf.axvline(x=t, color='black', linewidth=0.8)
        ax_lf.set_ylabel('LF', fontsize=FONT_SIZE, rotation=0, labelpad=20)
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
        ax_rf.set_ylabel('RF', fontsize=FONT_SIZE, rotation=0, labelpad=20)
        ax_rf.set_ylim(0, 1)
        ax_rf.set_yticks([])
        ax_rf.spines['left'].set_visible(False)
        ax_rf.spines['right'].set_visible(False)
        ax_rf.spines['top'].set_visible(False)
        ax_rf.set_xlabel('Timestep', fontsize=FONT_SIZE)
        ax_rf.set_xlim(time_steps[0], time_steps[-1])

        plt.show()
        
        self.last_fig = fig
        return fig
    
    def save_last_plot(
            self,
            trajectory_path: str,
            time_range: str = None,
            ext: str = "pdf",
            dpi: int = 300,
    ) -> Path:
        """
        Save the most recently created figure into the same directory this script lives in.

        Args:
            trajectory_path: The same path passed via --trajectory (used to derive filename).
            time_range: Optional range string (e.g. "199:600") to include in filename.
            ext: "pdf" or "png".
            dpi: Used for raster formats like png (ignored by pdf backends in most cases).

        Returns:
            Path to the saved file.
        """
        if self.last_fig is None:
            print("No figure to save. Please create a plot first.")
            return None

        script_dir = Path(__file__).resolve().parent
        
        traj_path = Path(trajectory_path)

        pct_match = re.search(r"percentage-(\d+pct)", traj_path.parent.name)
        pct_part = pct_match.group(1) if pct_match else "unknown"

        step_match = re.search(r"step_(\d+)", traj_path.stem)
        step_number = step_match.group(1) if step_match else "unknown"

        suffix = ""
        if time_range:
            saved_range = time_range.replace(":", "-")
            suffix = f"__range__{saved_range}"
        
        out_name = f"raster_ftstp_plots_sac_mpc_{pct_part}_{step_number}{suffix}.{ext.lstrip('.')}"
        out_path = script_dir / "viser_figs" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = dict(bbox_inches='tight', pad_inches=0.1)
        if ext.lower().lstrip(".") in ("png", "jpg", "jpeg", "tif", "tiff"):
            save_kwargs["dpi"] = dpi

        self.last_fig.savefig(out_path, **save_kwargs)
        print(f"Saved plot to: {out_path}")
        return out_path

    

def main():
    parser = argparse.ArgumentParser(
        description="Plot footsteps from walker trajectory data",
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

    plotter = FootStepPlotter()

    plotter.load_trajectory(traj_path, time_range=time_range)
    plotter.plot_footsteps()
    plotter.save_last_plot(args.trajectory, time_range=args.range, ext="png")


if __name__ == "__main__":
    main()
