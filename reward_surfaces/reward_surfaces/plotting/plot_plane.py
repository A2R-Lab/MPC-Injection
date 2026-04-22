"""Plotting functions for reward surfaces."""

import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import math
import warnings
from typing import Optional

FONT_SIZE = 18

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


def plot_surface(
    csv_path: str,
    output_name: str,
    env_name: str = "Environment",
    key_name: str = "episode_rewards",
    plot_type: str = "mesh",
    logscale: str = "auto",
    show: bool = False,
    save_data: bool = False,
) -> str:
    """
    Plot reward surface from CSV results.
    
    Args:
        csv_path: Path to results.csv file
        output_name: Base name for output files (without extension)
        env_name: Environment name for plot title
        key_name: Column name to plot (e.g., 'episode_rewards')
        plot_type: One of 'mesh', 'heat', 'contour', 'contourf', 'all'
        logscale: 'auto', 'on', or 'off'
        show: Whether to display the plot
        save_data: Whether to save the processed grid data as .npz file
    
    Returns:
        Path to saved plot file
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract grid dimensions
    dim0_values = sorted(df['dim0'].unique())
    dim1_values = sorted(df['dim1'].unique())
    grid_size = len(dim0_values)
    
    # Create meshgrid
    X = np.zeros((grid_size, grid_size))
    Y = np.zeros((grid_size, grid_size))
    Z = np.zeros((grid_size, grid_size))
    
    for idx, row in df.iterrows():
        i = dim0_values.index(row['dim0'])
        j = dim1_values.index(row['dim1'])
        X[j, i] = row['dim0']
        Y[j, i] = row['dim1']
        Z[j, i] = row[key_name]
    
    # Get magnitude from metadata
    magnitude = df['magnitude'].iloc[0] if 'magnitude' in df.columns else 1.0
    
    # Save processed grid data for easy reloading
    if save_data:
        data_path = f"{output_name}_plot_data.npz"
        np.savez(
            data_path,
            X=X,
            Y=Y,
            Z=Z,
            dim0_values=np.array(dim0_values),
            dim1_values=np.array(dim1_values),
            magnitude=magnitude,
            grid_size=grid_size,
            env_name=env_name,
            key_name=key_name,
        )
        print(f"Saved plot data: {data_path}")
    
    # Determine logscale
    use_logscale = False
    if logscale == "auto":
        if np.max(Z) - np.min(Z) > 10000:
            use_logscale = True
    elif logscale == "on":
        use_logscale = True
    
    # Create title
    title = f"{env_name} | {key_name.replace('_', ' ').title()}"
    
    output_path = None
    
    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    if plot_type in ['all', 'mesh']:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title, fontsize=FONT_SIZE + 2, fontweight='bold')
        
        if np.min(Z) < -1e9 and not use_logscale:
            print(f"Warning: Data includes extremely large negative rewards ({np.min(Z):.3E}). "
                  "Consider setting logscale='on'")
        
        # Scale X and Y by magnitude
        X_scaled = magnitude * X
        Y_scaled = magnitude * Y
        
        Z_plot = Z.copy()
        # Apply log scale if needed
        if use_logscale:
            Z_neg = Z[Z < 0]
            Z_pos = Z[Z >= 0]
            if len(Z_neg) > 0:
                Z_plot[Z < 0] = -np.log10(1 - Z_neg)
            if len(Z_pos) > 0:
                Z_plot[Z >= 0] = np.log10(1 + Z_pos)
        
        # Plot surface
        surf = ax.plot_surface(X_scaled, Y_scaled, Z_plot, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False, edgecolor='none', alpha=0.9)
        
        # Plot center line
        center_idx = grid_size // 2
        Z_range = abs(np.max(Z_plot) - np.min(Z_plot))
        zline = np.linspace(Z_plot[center_idx, center_idx], 
                           np.max(Z_plot) + (Z_range * 0.1), 4)
        xline = np.zeros_like(zline)
        yline = np.zeros_like(zline)
        ax.plot3D(xline, yline, zline, 'black', zorder=10)
        
        # Colorbar
        if use_logscale:
            # Create log scale colorbar
            max_Z = np.max(Z)
            min_Z = np.min(Z)
            max_mag = math.floor(math.log10(max(abs(max_Z), abs(min_Z), 1)))
            min_mag = math.floor(math.log10(max(abs(min_Z), 1)))
            
            ticks = np.round(np.linspace(min_mag, max_mag, 8, endpoint=True))
            cbar = fig.colorbar(surf, shrink=0.5, aspect=5, ticks=ticks, pad=0.1)
            
            labels = []
            for label in ticks:
                if abs(label) > 2:
                    label_str = f"$-10^{{{int(-label)}}}$" if label < 0 else f"$10^{{{int(label)}}}$"
                else:
                    val = -10.0**(-label) if label < 0 else 10.0**label
                    label_str = f"${val:.2f}$"
                labels.append(label_str)
            cbar.ax.set_yticklabels(labels)
            cbar.ax.tick_params(labelsize=FONT_SIZE - 2)
        else:
            cbar = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.05)
            cbar.ax.tick_params(labelsize=FONT_SIZE - 2)
        
        ax.set_xlabel('Direction 1', fontsize=FONT_SIZE)
        ax.set_ylabel('Direction 2', fontsize=FONT_SIZE)
        ax.set_zlabel(key_name.replace('_', ' ').title(), fontsize=FONT_SIZE)
        ax.tick_params(axis='x', labelsize=FONT_SIZE - 2)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 2)
        ax.tick_params(axis='z', labelsize=FONT_SIZE - 2)
        
        output_path = f"{output_name}_3dsurface.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        if not show:
            plt.close(fig)
    
    # --------------------------------------------------------------------
    # Plot 2D heatmap
    # --------------------------------------------------------------------
    if plot_type in ['all', 'heat']:
        sns.set_theme(
            font="Serif",
            rc={
                "axes.labelsize": FONT_SIZE,
                "axes.titlesize": FONT_SIZE + 2,
                "xtick.labelsize": FONT_SIZE - 2,
                "ytick.labelsize": FONT_SIZE - 2,
            },
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create labels
        labels_d1 = [f"{x:.1f}" for x in dim0_values]
        labels_d2 = [f"{y:.1f}" for y in dim1_values]
        
        sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True,
                              xticklabels=labels_d1, yticklabels=labels_d2,
                              ax=ax)
        sns_plot.invert_yaxis()
        sns_plot.set(xlabel="Direction 1", ylabel="Direction 2")
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight='bold')
        ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
        cbar = sns_plot.collections[0].colorbar
        cbar.ax.tick_params(labelsize=FONT_SIZE - 2)
        
        output_path = f"{output_name}_2dheat.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        if not show:
            plt.close(fig)
    
    # --------------------------------------------------------------------
    # Plot 2D contour
    # --------------------------------------------------------------------
    if plot_type in ['all', 'contour', 'contourf']:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_type in ['all', 'contourf']:
            CS = ax.contourf(X, Y, Z, cmap='summer', levels=20)
        else:
            CS = ax.contour(X, Y, Z, cmap='summer', levels=20)
            ax.clabel(CS, inline=1, fontsize=FONT_SIZE - 4)
        
        ax.set_xlabel('Direction 1', fontsize=FONT_SIZE)
        ax.set_ylabel('Direction 2', fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight='bold')
        ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
        cbar = plt.colorbar(CS, ax=ax)
        cbar.ax.tick_params(labelsize=FONT_SIZE - 2)
        
        suffix = '_2dcontourf.png' if plot_type in ['all', 'contourf'] else '_2dcontour.png'
        output_path = f"{output_name}{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        if not show:
            plt.close(fig)
    
    if show:
        plt.show()
    
    return output_path
