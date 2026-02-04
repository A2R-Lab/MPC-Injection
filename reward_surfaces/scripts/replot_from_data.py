#!/usr/bin/env python3
"""
Reload and replot reward surface from saved .npz or CSV data.

This script allows you to customize plots without re-running evaluations.
You can either:
  1. Load from a previously saved .npz file (fastest, contains processed grids)
  2. Load from results.csv (original data, reprocessed)

Usage examples:
  # Replot from .npz file with custom settings
  python replot_from_data.py checkpoint_100000/surface_plot_data.npz --title "Custom Title"
  
  # Replot from CSV file
  python replot_from_data.py checkpoint_100000/results.csv --title "My Plot"
  
  # Show plot interactively instead of saving
  python replot_from_data.py checkpoint_100000/surface_plot_data.npz --show
  
  # Compare two surfaces and plot the difference
  python replot_from_data.py path/to/checkpoint_100000/results.csv --data-path2 path2/to/checkpoint_100000/results.csv --compare
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

FONT_SIZE = 20

def extract_checkpoint_number(path: str) -> int:
    """Extract checkpoint number from file path (e.g., 'checkpoint_100000' -> 100000)."""
    match = re.search(r'checkpoint_(\d+)', str(path))
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract checkpoint number from path: {path}")


def load_from_npz(npz_path: str) -> dict:
    """Load pre-processed grid data from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'X': data['X'],
        'Y': data['Y'],
        'Z': data['Z'],
        'dim0_values': data['dim0_values'],
        'dim1_values': data['dim1_values'],
        'magnitude': float(data['magnitude']),
        'grid_size': int(data['grid_size']),
        'env_name': str(data['env_name']),
        'key_name': str(data['key_name']),
    }


def load_from_csv(csv_path: str, key_name: str = 'episode_rewards') -> dict:
    """Load and process grid data from CSV file."""
    df = pd.read_csv(csv_path)
    
    dim0_values = sorted(df['dim0'].unique())
    dim1_values = sorted(df['dim1'].unique())
    grid_size = len(dim0_values)
    
    X = np.zeros((grid_size, grid_size))
    Y = np.zeros((grid_size, grid_size))
    Z = np.zeros((grid_size, grid_size))
    
    for idx, row in df.iterrows():
        i = dim0_values.index(row['dim0'])
        j = dim1_values.index(row['dim1'])
        X[j, i] = row['dim0']
        Y[j, i] = row['dim1']
        Z[j, i] = row[key_name]
    
    magnitude = df['magnitude'].iloc[0] if 'magnitude' in df.columns else 1.0
    
    return {
        'X': X,
        'Y': Y,
        'Z': Z,
        'dim0_values': np.array(dim0_values),
        'dim1_values': np.array(dim1_values),
        'magnitude': magnitude,
        'grid_size': grid_size,
        'env_name': 'Environment',
        'key_name': key_name,
    }


def plot_2d_heatmap(data: dict, title: str = None, output_path: str = None,
                    show: bool = False, cmap: str = 'viridis', 
                    figsize: tuple = (10, 8), dpi: int = 300,
                    xlabel: str = "Direction 1", ylabel: str = "Direction 2",
                    vmin: float = None, vmax: float = None):
    """
    Plot 2D heatmap from loaded data.
    
    Args:
        data: Dictionary with X, Y, Z arrays and metadata
        title: Custom title (defaults to env_name | key_name)
        output_path: Path to save figure (if None, auto-generates)
        show: Display plot interactively
        cmap: Matplotlib colormap name
        figsize: Figure size tuple
        dpi: DPI for saved figure
        xlabel: X-axis label
        ylabel: Y-axis label
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    """
    sns.set_theme(font="Serif")
    fig, ax = plt.subplots(figsize=figsize)
    
    Z = data['Z']
    dim0_values = data['dim0_values']
    dim1_values = data['dim1_values']
    
    # Create labels
    labels_d1 = [f"{x:.1f}" for x in dim0_values]
    labels_d2 = [f"{y:.1f}" for y in dim1_values]
    
    # Set title
    if title is None:
        key_name = data.get('key_name', 'episode_rewards')
        env_name = data.get('env_name', 'Environment')
        title = f"{env_name} | {key_name.replace('_', ' ').title()}"
    
    sns_plot = sns.heatmap(Z, cmap=cmap, cbar=True,
                          ax=ax, vmin=vmin, vmax=vmax)
    sns_plot.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE+2)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.set_label('Reward', fontsize=FONT_SIZE)
        cbar.ax.tick_params(labelsize=FONT_SIZE-4)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_3d_surface(data: dict, title: str = None, output_path: str = None,
                    show: bool = False, cmap: str = 'coolwarm',
                    figsize: tuple = (12, 9), dpi: int = 300,
                    xlabel: str = "Direction 1", ylabel: str = "Direction 2",
                    zlabel: str = None, elev: float = 30, azim: float = -60):
    """
    Plot 3D surface from loaded data.
    
    Args:
        data: Dictionary with X, Y, Z arrays and metadata
        title: Custom title
        output_path: Path to save figure
        show: Display plot interactively
        cmap: Matplotlib colormap name
        figsize: Figure size tuple
        dpi: DPI for saved figure
        xlabel: X-axis label
        ylabel: Y-axis label
        zlabel: Z-axis label (defaults to key_name)
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    X = data['X']
    Y = data['Y']
    Z = data['Z']
    magnitude = data.get('magnitude', 1.0)
    
    # Scale by magnitude
    X_scaled = magnitude * X
    Y_scaled = magnitude * Y
    
    # Set title
    if title is None:
        key_name = data.get('key_name', 'episode_rewards')
        env_name = data.get('env_name', 'Environment')
        title = f"{env_name} | {key_name.replace('_', ' ').title()}"
    
    fig.suptitle(title, fontsize=FONT_SIZE+2)
    
    # Plot surface
    surf = ax.plot_surface(X_scaled, Y_scaled, Z, cmap=cmap,
                          linewidth=0, antialiased=False, edgecolor='none', alpha=0.9)
    
    # Plot center marker
    grid_size = data.get('grid_size', Z.shape[0])
    center_idx = grid_size // 2
    Z_range = abs(np.max(Z) - np.min(Z))
    zline = np.linspace(Z[center_idx, center_idx], np.max(Z) + (Z_range * 0.1), 4)
    xline = np.zeros_like(zline)
    yline = np.zeros_like(zline)
    ax.plot3D(xline, yline, zline, 'black', zorder=10)
    
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.05)
    cbar.ax.tick_params(labelsize=FONT_SIZE-4)
    
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_zlabel(zlabel or data.get('key_name', 'Reward').replace('_', ' ').title(), fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE-4)
    ax.view_init(elev=elev, azim=azim)
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_contour(data: dict, title: str = None, output_path: str = None,
                 show: bool = False, cmap: str = 'summer', filled: bool = True,
                 figsize: tuple = (10, 8), dpi: int = 300, levels: int = 20,
                 xlabel: str = "Direction 1", ylabel: str = "Direction 2"):
    """
    Plot 2D contour from loaded data.
    
    Args:
        data: Dictionary with X, Y, Z arrays and metadata
        title: Custom title
        output_path: Path to save figure
        show: Display plot interactively
        cmap: Matplotlib colormap name
        filled: Use filled contour (contourf) vs line contour
        figsize: Figure size tuple
        dpi: DPI for saved figure
        levels: Number of contour levels
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    X = data['X']
    Y = data['Y']
    Z = data['Z']
    
    # Set title
    if title is None:
        key_name = data.get('key_name', 'episode_rewards')
        env_name = data.get('env_name', 'Environment')
        title = f"{env_name} | {key_name.replace('_', ' ').title()}"
    
    if filled:
        CS = ax.contourf(X, Y, Z, cmap=cmap, levels=levels)
    else:
        CS = ax.contour(X, Y, Z, cmap=cmap, levels=levels)
        ax.clabel(CS, inline=1, fontsize=FONT_SIZE-6)
    
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE+2)
    ax.tick_params(axis='both', labelsize=FONT_SIZE-4)
    cbar = plt.colorbar(CS, ax=ax)
    cbar.ax.tick_params(labelsize=FONT_SIZE-4)
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Reload and replot reward surface from saved data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('data_path', type=str,
                       help='Path to .npz or .csv file')
    parser.add_argument('--data-path2', type=str, default=None,
                       help='Path to second .npz or .csv file for comparison mode')
    parser.add_argument('--compare', action='store_true',
                       help='Compare two surfaces: show both, then show and save their difference')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (auto-generated if not specified)')
    parser.add_argument('--title', type=str,
                       help='Custom plot title')
    parser.add_argument('--title2', type=str,
                       help='Custom plot title for second surface (comparison mode)')
    parser.add_argument('--type', type=str, default='heat',
                       choices=['heat', 'surface', 'contour', 'contourf', 'all'],
                       help='Plot type')
    parser.add_argument('--cmap', type=str, default=None,
                       help='Colormap (default: viridis for heat, coolwarm for 3D)')
    parser.add_argument('--key', type=str, default='episode_rewards',
                       help='Key to plot (only used when loading from CSV)')
    parser.add_argument('--show', action='store_true',
                       help='Display plot interactively')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # =========================================================================
    # Comparison mode: load two surfaces, show both, subtract, save difference
    # =========================================================================
    if args.compare:
        if not args.data_path2:
            raise ValueError("Comparison mode requires --data-path2 to be specified")
        
        data_path2 = Path(args.data_path2)
        if not data_path2.exists():
            raise FileNotFoundError(f"Second data file not found: {data_path2}")
        
        # Extract and validate checkpoint numbers
        checkpoint1 = extract_checkpoint_number(str(data_path))
        checkpoint2 = extract_checkpoint_number(str(data_path2))
        
        if checkpoint1 != checkpoint2:
            raise ValueError(
                f"Checkpoint numbers do not match!\n"
                f"  File 1: checkpoint_{checkpoint1} (from {data_path})\n"
                f"  File 2: checkpoint_{checkpoint2} (from {data_path2})\n"
                f"Both files must be from the same checkpoint number."
            )
        
        print(f"Checkpoint number: {checkpoint1}")
        
        # Load first dataset
        print(f"\n=== Loading first surface ===")
        if data_path.suffix == '.npz':
            print(f"Loading from .npz file: {data_path}")
            data1 = load_from_npz(str(data_path))
        elif data_path.suffix == '.csv':
            print(f"Loading from CSV file: {data_path}")
            data1 = load_from_csv(str(data_path), key_name=args.key)
        else:
            raise ValueError(f"Unsupported file type: {data_path.suffix}")
        
        print(f"  Grid size: {data1['grid_size']}x{data1['grid_size']}")
        print(f"  Z range: [{data1['Z'].min():.2f}, {data1['Z'].max():.2f}]")
        
        # Load second dataset
        print(f"\n=== Loading second surface ===")
        if data_path2.suffix == '.npz':
            print(f"Loading from .npz file: {data_path2}")
            data2 = load_from_npz(str(data_path2))
        elif data_path2.suffix == '.csv':
            print(f"Loading from CSV file: {data_path2}")
            data2 = load_from_csv(str(data_path2), key_name=args.key)
        else:
            raise ValueError(f"Unsupported file type: {data_path2.suffix}")
        
        print(f"  Grid size: {data2['grid_size']}x{data2['grid_size']}")
        print(f"  Z range: [{data2['Z'].min():.2f}, {data2['Z'].max():.2f}]")
        
        # Validate grid sizes match
        if data1['grid_size'] != data2['grid_size']:
            raise ValueError(
                f"Grid sizes do not match: {data1['grid_size']} vs {data2['grid_size']}"
            )
        
        # Show first surface
        print(f"\n=== Showing first surface (close window to continue) ===")
        title1 = args.title or f"Surface 1: {data_path.parent.parent.name}"
        plot_2d_heatmap(data1, title=title1, output_path=None,
                       show=True, cmap=args.cmap or 'viridis', dpi=args.dpi)
        
        # Show second surface
        print(f"\n=== Showing second surface (close window to continue) ===")
        title2 = args.title2 or f"Surface 2: {data_path2.parent.parent.name}"
        plot_2d_heatmap(data2, title=title2, output_path=None,
                       show=True, cmap=args.cmap or 'viridis', dpi=args.dpi)
        
        # Compute difference: data1 - data2
        print(f"\n=== Computing difference (Surface 1 - Surface 2) ===")
        diff_data = {
            'X': data1['X'].copy(),
            'Y': data1['Y'].copy(),
            'Z': np.abs(data1['Z'] - data2['Z']),
            'dim0_values': data1['dim0_values'],
            'dim1_values': data1['dim1_values'],
            'magnitude': data1.get('magnitude', 1.0),
            'grid_size': data1['grid_size'],
            'env_name': 'Difference',
            'key_name': 'reward_difference',
        }
        
        print(f"  Difference Z range: [{diff_data['Z'].min():.2f}, {diff_data['Z'].max():.2f}]")
        
        # Show difference surface
        print(f"\n=== Showing difference surface (close window to save) ===")
        diff_title = f"Difference (checkpoint {checkpoint1}): Surface 1 - Surface 2"
        plot_2d_heatmap(diff_data, title=diff_title, output_path=None,
                       show=True, cmap='viridis', dpi=args.dpi)
        
        # Save difference surface
        output_dir = args.output or str(data_path.parent.parent)
        output_path = f"{output_dir}/surface_2dheat_{checkpoint1}_subtracted.png"
        print(f"\n=== Saving difference surface ===")
        plot_2d_heatmap(diff_data, title=diff_title, output_path=output_path,
                       show=False, cmap='viridis', dpi=args.dpi)
        
        print(f"\nDone! Difference plot saved to: {output_path}")
        return
    
    # =========================================================================
    # Single surface mode (original behavior)
    # =========================================================================
    
    # Load data
    if data_path.suffix == '.npz':
        print(f"Loading from .npz file: {data_path}")
        data = load_from_npz(str(data_path))
    elif data_path.suffix == '.csv':
        print(f"Loading from CSV file: {data_path}")
        data = load_from_csv(str(data_path), key_name=args.key)
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix}")
    
    print(f"  Grid size: {data['grid_size']}x{data['grid_size']}")
    print(f"  Z range: [{data['Z'].min():.2f}, {data['Z'].max():.2f}]")
    
    # Generate output path if not specified
    base_output = args.output or str(data_path.parent / data_path.stem.replace('_plot_data', ''))
    
    # Plot
    if args.type in ['all', 'heat']:
        output_path = f"{base_output}_2dheat.png" if args.type == 'all' else f"{base_output}.png"
        cmap = args.cmap or 'viridis'
        plot_2d_heatmap(data, title=args.title, output_path=output_path,
                       show=args.show, cmap=cmap, dpi=args.dpi)
    
    if args.type in ['all', 'surface']:
        output_path = f"{base_output}_3dsurface.png" if args.type == 'all' else f"{base_output}.png"
        cmap = args.cmap or 'coolwarm'
        plot_3d_surface(data, title=args.title, output_path=output_path,
                       show=args.show, cmap=cmap, dpi=args.dpi)
    
    if args.type in ['all', 'contour', 'contourf']:
        suffix = '_2dcontourf.png' if args.type in ['all', 'contourf'] else '_2dcontour.png'
        output_path = f"{base_output}{suffix}"
        cmap = args.cmap or 'summer'
        filled = args.type in ['all', 'contourf']
        plot_contour(data, title=args.title, output_path=output_path,
                    show=args.show, cmap=cmap, filled=filled, dpi=args.dpi)


if __name__ == "__main__":
    """
    Example usage:

    python reward_surfaces/scripts/replot_from_data.py \
  plots/sac-mpc_walker_0pct_training_surfaces/checkpoint_100000/results.csv \
  --data-path2 plots/sac-mpc_walker_50pct_training_surfaces/checkpoint_100000/results.csv \
  --compare
    """
    main()

