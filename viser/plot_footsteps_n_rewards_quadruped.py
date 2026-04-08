"""
Script to plot the footsteps and rewards from the quadruped environment.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


FONT_SIZE = 18
DEFAULT_FOOT_NAMES = ["FL", "FR", "RL", "RR"]


class FootStepAndRewardPlotter:
    """
    Visualize the footsteps of the quadruped and plot the rewards over time for a
    single trajectory.
    """

    def __init__(self):
        """Initialize the plotter."""
        self.trajectory_data = None
        self.last_fig = None
        self.last_trajectory_path = None
        self.foot_contacts = None
        self.foot_names = None
        self.rewards = None

    def load_trajectory(self, npz_path: str, time_range: str = None):
        """
        Load trajectory data from an .npz file.

        Args:
            npz_path: Path to the .npz file containing trajectory data.
            time_range: Optional range of timesteps to load (e.g., "199:600").
        """
        data = np.load(npz_path, allow_pickle=True)

        self.last_trajectory_path = npz_path
        self.trajectory_data = data

        if "foot_contacts" not in data.files:
            raise KeyError(
                f"Expected 'foot_contacts' in {npz_path}, available keys: {list(data.files)}"
            )
        if "rewards" not in data.files:
            raise KeyError(
                f"Expected 'rewards' in {npz_path}, available keys: {list(data.files)}"
            )

        self.foot_contacts = np.asarray(data["foot_contacts"], dtype=bool)
        self.rewards = np.asarray(data["rewards"])

        if "foot_names" in data.files:
            self.foot_names = [str(name) for name in data["foot_names"]]
        else:
            self.foot_names = DEFAULT_FOOT_NAMES.copy()

        num_steps = min(len(self.rewards), len(self.foot_contacts))
        self.rewards = self.rewards[:num_steps]
        self.foot_contacts = self.foot_contacts[:num_steps]

        if time_range:
            start, end = self._parse_range(time_range)
            self.foot_contacts = self.foot_contacts[start:end]
            self.rewards = self.rewards[start:end]
            print(f"Loaded timesteps {start}:{end} (total: {len(self.rewards)} steps)")
        else:
            print(f"Loaded full trajectory (total: {len(self.rewards)} steps)")

        print(f"Available data: {list(data.keys())}")
        print(f"Using foot names: {self.foot_names}")

        return data

    def _parse_range(self, range_str: str):
        """
        Parse range string like "199:600" into start and end indices.

        Args:
            range_str: Range string in format "start:end".

        Returns:
            tuple: (start, end) indices
        """
        parts = range_str.split(":")
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if len(parts) > 1 and parts[1] else None
        return start, end

    def plot_footsteps_and_rewards(self):
        """Plot footsteps and rewards as stacked raster plots."""
        if self.trajectory_data is None:
            print("No trajectory data loaded. Please load trajectory first.")
            return None

        num_steps = len(self.rewards)
        if num_steps == 0:
            print("Trajectory is empty after preprocessing.")
            return None

        time_steps = np.arange(num_steps)
        num_feet = self.foot_contacts.shape[1]

        fig_height = 1.4 + 0.45 * num_feet
        fig, axes = plt.subplots(
            num_feet + 1,
            1,
            figsize=(12, fig_height),
            sharex=True,
            gridspec_kw={"height_ratios": [0.4] * num_feet + [1.2]},
            layout="constrained",
        )

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for foot_idx in range(num_feet):
            ax = axes[foot_idx]
            foot_name = self.foot_names[foot_idx] if foot_idx < len(self.foot_names) else f"F{foot_idx}"
            for t in time_steps:
                if self.foot_contacts[t, foot_idx]:
                    ax.axvline(x=t, color="black", linewidth=0.8)
            ax.set_ylabel(foot_name, fontsize=FONT_SIZE, rotation=0, labelpad=20)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        ax_r = axes[-1]
        reward_min, reward_max = self.rewards.min(), self.rewards.max()
        ax_r.plot(time_steps, self.rewards, linewidth=1.5, color="#1f77b4")
        ax_r.set_xlim(time_steps[0], time_steps[-1])
        ax_r.set_ylabel("Reward", fontsize=FONT_SIZE)
        ax_r.set_xlabel("Timestep", fontsize=FONT_SIZE)
        ax_r.grid(True, alpha=0.3, linestyle="--")
        print(f"Reward range: [{reward_min:.4f}, {reward_max:.4f}]")

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
        """
        if self.last_fig is None:
            print("No figure to save. Please create a plot first.")
            return None

        script_dir = Path(__file__).resolve().parent
        traj_path = Path(trajectory_path)

        parent_name = traj_path.parent.name
        pct_part = "unknown"
        if "percentage-" in parent_name:
            pct_start = parent_name.find("percentage-") + len("percentage-")
            pct_part = parent_name[pct_start:]

        filename = traj_path.stem
        step_number = "unknown"
        if "step_" in filename:
            step_start = filename.find("step_") + len("step_")
            step_number = filename[step_start:]

        suffix = ""
        if time_range:
            saved_range = time_range.replace(":", "-")
            suffix = f"__range__{saved_range}"

        out_name = (
            f"raster_n_reward_plots_sac_mpc_quadruped_{pct_part}_{step_number}"
            f"{suffix}.{ext.lstrip('.')}"
        )
        out_path = script_dir / "viser_figs" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.1}
        if ext.lower().lstrip(".") in ("png", "jpg", "jpeg", "tif", "tiff"):
            save_kwargs["dpi"] = dpi

        self.last_fig.savefig(out_path, **save_kwargs)
        print(f"Saved plot to: {out_path}")
        return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot footsteps and rewards from quadruped trajectory data",
        epilog="""
        Examples:
        python viser/plot_footsteps_n_rewards_quadruped.py --trajectory path/to/trajectory_data.npz
        """,
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        help="Path to trajectory .npz file",
        default=(
            "body_trajs/model_traj_data_quadruped/"
            "quadruped-velocity_tracking-SAC-MPC-20260405-015147-percentage-25pct-seed100/"
            "trajectories_step_500000.npz"
        ),
    )

    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Range of timesteps to plot (e.g., '199:600' or ':500' or '100:')",
    )

    args = parser.parse_args()

    plotter = FootStepAndRewardPlotter()
    plotter.load_trajectory(args.trajectory, time_range=args.range)
    plotter.plot_footsteps_and_rewards()
    plotter.save_last_plot(args.trajectory, time_range=args.range, ext="png")


if __name__ == "__main__":
    main()
