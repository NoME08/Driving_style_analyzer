import pandas as pd
import numpy as np


def detect_driving_modes(df, accel_threshold=2.0, brake_threshold=-2.0, stop_threshold=1.5):
    """
    Detect driving modes based on acceleration and speed thresholds.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'speed_kmh', 'acceleration'
    accel_threshold : float
        Acceleration threshold for 'accel' mode (km/h/s)
    brake_threshold : float
        Deceleration threshold for 'decel' mode (km/h/s)
    stop_threshold : float
        Speed threshold for 'stop' mode (km/h)

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with added 'driving_mode' column
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Initialize all as 'cruise'
    df['driving_mode'] = 'cruise'

    # Apply thresholds
    df.loc[df['acceleration'] > accel_threshold, 'driving_mode'] = 'accel'
    df.loc[df['acceleration'] < brake_threshold, 'driving_mode'] = 'decel'
    df.loc[df['speed_kmh'] < stop_threshold, 'driving_mode'] = 'stop'

    return df


def get_mode_statistics(df):
    """
    Calculate statistics for driving modes.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'driving_mode' column

    Returns:
    --------
    pandas.Series
        Mode counts with percentages
    dict
        Dictionary with detailed statistics
    """
    mode_counts = df['driving_mode'].value_counts()

    # Calculate percentages
    total = len(df)
    mode_stats = pd.DataFrame({
        'count': mode_counts,
        'percentage': (mode_counts / total * 100).round(1)
    })

    # Create detailed statistics dictionary
    stats = {
        'total_points': total,
        'mode_counts': mode_counts.to_dict(),
        'mode_percentages': (mode_counts / total * 100).round(1).to_dict(),
        'modes_detected': list(mode_counts.index)
    }

    return mode_stats, stats


def print_mode_statistics(df):
    """
    Print driving mode statistics in a readable format.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'driving_mode' column
    """
    mode_stats, stats = get_mode_statistics(df)

    print("\n" + "=" * 50)
    print("Driving Mode Statistics")
    print("=" * 50)
    print(f"Total data points: {stats['total_points']}")

    # Print in a consistent order
    for mode in ['stop', 'accel', 'decel', 'cruise']:
        if mode in mode_stats.index:
            count = mode_stats.loc[mode, 'count']
            percentage = mode_stats.loc[mode, 'percentage']
            print(f"{mode:8s}: {count:6d} points ({percentage:5.1f}%)")
        else:
            print(f"{mode:8s}:      0 points (  0.0%)")

    print("=" * 50)


def get_mode_durations(df, time_col='time_sec'):
    """
    Calculate total duration for each driving mode.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'driving_mode' column and time column
    time_col : str
        Name of the time column (should be in seconds)

    Returns:
    --------
    pandas.Series
        Total duration in seconds for each mode
    """
    # Calculate time differences between consecutive points
    time_diff = np.gradient(df[time_col])

    # Group by driving mode and sum time differences
    mode_durations = df.groupby('driving_mode').apply(
        lambda group: np.sum(time_diff[group.index])
    )

    return mode_durations