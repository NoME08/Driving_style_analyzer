import matplotlib.pyplot as plt
import numpy as np


def plot_main_analysis(df, trips_df=None, total_distance=None, figsize=(15, 10),
                      save_path=None, title_suffix=None):
    """
    Create main analysis visualization with 3 subplots.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: time_sec, speed_kmh, acceleration, driving_mode
    trips_df : pandas.DataFrame, optional
        DataFrame with trip statistics (for total distance if not provided)
    total_distance : float, optional
        Total distance in km (calculated from df if not provided)
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure (e.g., 'analysis.png')
    title_suffix : str, optional
        Suffix for the figure title (e.g., filename)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    title = 'Driving Cycle Analysis'
    if title_suffix:
        title += f' — {title_suffix}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Calculate total distance if not provided
    if total_distance is None:
        if trips_df is not None and 'distance_km' in trips_df.columns:
            total_distance = trips_df['distance_km'].sum()
        else:
            # Calculate from speed data
            total_distance = np.trapezoid(df['speed_kmh'], df['time_sec']) / 3600

    # 1. Speed with driving modes
    ax1 = plt.subplot(3, 1, 1)

    # Define colors for driving modes
    colors = {
        'stop': 'gray',
        'cruise': 'blue',
        'accel': 'red',
        'decel': 'orange'
    }

    # Plot each driving mode with different colors
    for mode, color in colors.items():
        mask = df['driving_mode'] == mode
        if mask.any():
            ax1.scatter(df.loc[mask, 'time_sec'] / 60,  # Convert to minutes
                        df.loc[mask, 'speed_kmh'],
                        c=color,
                        label=mode.replace('_', ' ').title(),
                        s=1, alpha=0.5)

    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('Speed with Driving Modes')
    ax1.legend(markerscale=5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, df['time_sec'].max() / 60)

    # 2. Acceleration profile
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df['time_sec'] / 60, df['acceleration'], 'b-', linewidth=0.8, alpha=0.5)

    # Add threshold lines
    ax2.axhline(y=3, color='r', linestyle='--', alpha=0.8,
                label='Hard accel threshold (+3 km/h/s)')
    ax2.axhline(y=-3, color='orange', linestyle='--', alpha=0.8,
                label='Hard brake threshold (-3 km/h/s)')

    # Fill areas beyond thresholds
    ax2.fill_between(df['time_sec'] / 60, 3, df['acceleration'],
                     where=(df['acceleration'] > 3),
                     color='red', alpha=0.3, interpolate=True)
    ax2.fill_between(df['time_sec'] / 60, -3, df['acceleration'],
                     where=(df['acceleration'] < -3),
                     color='orange', alpha=0.3, interpolate=True)

    ax2.set_ylabel('Acceleration (km/h/s)')
    ax2.set_title('Acceleration Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, df['time_sec'].max() / 60)

    # 3. Cumulative distance
    ax3 = plt.subplot(3, 1, 3)
    cumulative_distance = np.cumsum(df['speed_kmh'] * np.gradient(df['time_sec'])) / 3600
    ax3.plot(df['time_sec'] / 60, cumulative_distance, 'g-', linewidth=2)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Cumulative Distance (km)')
    ax3.set_title(f'Cumulative Distance ({total_distance:.2f} km total)')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(df['time_sec'] / 60, 0, cumulative_distance,
                     alpha=0.2, color='green')
    ax3.set_xlim(0, df['time_sec'].max() / 60)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    return fig


def plot_detailed_statistics(df, trips_df, figsize=(14, 10), save_path=None,
                            title_suffix=None):
    """
    Create detailed statistics visualization with 2x2 subplots.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: speed_kmh, acceleration
    trips_df : pandas.DataFrame
        DataFrame with trip statistics
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure
    title_suffix : str, optional
        Suffix for the figure title (e.g., filename)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    title = 'Driving Cycle Detailed Statistics'
    if title_suffix:
        title += f' — {title_suffix}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Speed distribution histogram
    ax = axes[0, 0]
    ax.hist(df['speed_kmh'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=df['speed_kmh'].mean(), color='red', linestyle='--',
               label=f"Mean: {df['speed_kmh'].mean():.1f} km/h")
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('Frequency (seconds)')
    ax.set_title('Speed Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Acceleration distribution histogram
    ax = axes[0, 1]
    ax.hist(df['acceleration'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=3, color='red', linestyle='--', label='Hard accel threshold')
    ax.axvline(x=-3, color='orange', linestyle='--', label='Hard brake threshold')
    ax.set_xlabel('Acceleration (km/h/s)')
    ax.set_ylabel('Frequency (seconds)')
    ax.set_title('Acceleration Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Trip durations
    ax = axes[1, 0]
    if len(trips_df) > 0:
        bars = ax.bar(range(1, len(trips_df) + 1), trips_df['duration_min'],
                      color='purple', alpha=0.7)
        ax.set_xlabel('Trip Number')
        ax.set_ylabel('Duration (minutes)')
        ax.set_title('Trip Durations')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No trips identified',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Trip Durations')

    # 4. Trip distances
    ax = axes[1, 1]
    if len(trips_df) > 0:
        bars = ax.bar(range(1, len(trips_df) + 1), trips_df['distance_km'],
                      color='orange', alpha=0.7)
        ax.set_xlabel('Trip Number')
        ax.set_ylabel('Distance (km)')
        ax.set_title('Trip Distances')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No trips identified',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Trip Distances')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    return fig


