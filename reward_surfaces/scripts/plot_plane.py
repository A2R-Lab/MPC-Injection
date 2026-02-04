"""Plot reward surface from CSV results."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
script_dir = Path(__file__).parent
reward_surfaces_root = script_dir.parent
sys.path.insert(0, str(reward_surfaces_root))

from reward_surfaces.plotting import plot_surface


def main():
    parser = argparse.ArgumentParser(description='Plot reward surface')
    parser.add_argument('csv_path', type=str, help='Path to results.csv')
    parser.add_argument('--outname', type=str, required=True,
                       help='Output file base name (without extension)')
    parser.add_argument('--env-name', type=str, default="Environment",
                       help='Environment name for plot title')
    parser.add_argument('--key', type=str, default="episode_rewards",
                       help='Key to plot (e.g., episode_rewards, episode_std_rewards)')
    parser.add_argument('--type', type=str, default="mesh",
                       choices=['mesh', 'heat', 'contour', 'contourf', 'all'],
                       help='Plot type')
    parser.add_argument('--logscale', type=str, default="auto",
                       choices=['auto', 'on', 'off'],
                       help='Use log scale for rewards')
    parser.add_argument('--show', action='store_true',
                       help='Display plot instead of saving')
    parser.add_argument('--no-save-data', action='store_true',
                       help='Do not save plot data as .npz file')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Plotting reward surface from {csv_path}")
    
    output_path = plot_surface(
        csv_path=str(csv_path),
        output_name=args.outname,
        env_name=args.env_name,
        key_name=args.key,
        plot_type=args.type,
        logscale=args.logscale,
        show=args.show,
        save_data=not args.no_save_data,
    )
    
    if not args.show:
        print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
