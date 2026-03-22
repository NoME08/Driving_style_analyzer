import pandas as pd
import numpy as np


def identify_trips(df, stop_mode='stop'):
    """
    Identify individual trips based on stop points.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'driving_mode' column
    stop_mode : str
        Driving mode that indicates a stop (default: 'stop')

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with added 'trip_id' column (-1 for stop points)
    """
    df = df.copy()

    # Mark stop points
    df['is_stopped'] = df['driving_mode'] == stop_mode

    # Create trip IDs based on changes in stop status
    # Each time we transition from stopped to moving or vice versa, increment trip_id
    df['trip_id'] = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()

    # Set trip_id to -1 for stop points
    df.loc[df['is_stopped'], 'trip_id'] = -1

    return df


def calculate_trip_statistics(df, time_col='time_sec', speed_col='speed_kmh'):
    """
    Calculate statistics for each identified trip.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'trip_id' column (positive values for trips)
    time_col : str
        Name of the time column (seconds)
    speed_col : str
        Name of the speed column (km/h)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with trip statistics
    dict
        Summary statistics for all trips
    """
    # Get unique trip IDs (excluding -1 for stops)
    trip_ids = df[df['trip_id'] > 0]['trip_id'].unique()

    trips = []
    for trip_id in trip_ids:
        trip_data = df[df['trip_id'] == trip_id]

        if len(trip_data) > 0:
            # Calculate distance using trapezoidal integration
            # Distance (km) = integral of speed (km/h) over time (h)
            # time in seconds, convert to hours for integration
            distance = np.trapezoid(trip_data[speed_col], trip_data[time_col]) / 3600

            # Duration in seconds (number of data points * time interval)
            # Use time difference between first and last point
            duration_sec = trip_data[time_col].iloc[-1] - trip_data[time_col].iloc[0]

            trips.append({
                'trip_id': len(trips) + 1,  # Re-index starting from 1
                'original_trip_id': trip_id,
                'duration_sec': duration_sec,
                'duration_min': duration_sec / 60,
                'max_speed_kmh': trip_data[speed_col].max(),
                'min_speed_kmh': trip_data[speed_col].min(),
                'avg_speed_kmh': trip_data[speed_col].mean(),
                'distance_km': distance,
                'data_points': len(trip_data)
            })

    trips_df = pd.DataFrame(trips)

    # Calculate summary statistics
    if len(trips_df) > 0:
        summary = {
            'total_trips': len(trips_df),
            'total_distance_km': trips_df['distance_km'].sum(),
            'total_duration_sec': trips_df['duration_sec'].sum(),
            'avg_duration_sec': trips_df['duration_sec'].mean(),
            'avg_distance_km': trips_df['distance_km'].mean(),
            'max_duration_sec': trips_df['duration_sec'].max(),
            'min_duration_sec': trips_df['duration_sec'].min(),
            'max_distance_km': trips_df['distance_km'].max(),
            'min_distance_km': trips_df['distance_km'].min(),
            'avg_speed_kmh': trips_df['avg_speed_kmh'].mean(),
        }
    else:
        summary = {
            'total_trips': 0,
            'total_distance_km': 0,
            'total_duration_sec': 0,
            'avg_duration_sec': 0,
            'avg_distance_km': 0,
            'max_duration_sec': 0,
            'min_duration_sec': 0,
            'max_distance_km': 0,
            'min_distance_km': 0,
            'avg_speed_kmh': 0,
        }

    return trips_df, summary


def print_trip_summary(trips_df, summary):
    """
    Print trip statistics in a readable format.

    Parameters:
    -----------
    trips_df : pandas.DataFrame
        DataFrame with trip statistics
    summary : dict
        Summary statistics dictionary
    """
    print("\n" + "=" * 50)
    print("Trip Analysis Summary")
    print("=" * 50)
    print(f"Total trips identified: {summary['total_trips']}")
    print(f"Total distance:        {summary['total_distance_km']:.2f} km")
    print(f"Total duration:        {summary['total_duration_sec']:.0f} seconds "
          f"({summary['total_duration_sec'] / 60:.1f} minutes)")

    if summary['total_trips'] > 0:
        print(f"\nPer trip statistics:")
        print(f"  Average duration:    {summary['avg_duration_sec']:.0f} seconds "
              f"({summary['avg_duration_sec'] / 60:.1f} minutes)")
        print(f"  Longest trip:        {summary['max_duration_sec']:.0f} seconds "
              f"({summary['max_duration_sec'] / 60:.1f} minutes)")
        print(f"  Shortest trip:       {summary['min_duration_sec']:.0f} seconds "
              f"({summary['min_duration_sec'] / 60:.1f} minutes)")
        print(f"  Average distance:    {summary['avg_distance_km']:.2f} km")
        print(f"  Longest distance:    {summary['max_distance_km']:.2f} km")
        print(f"  Shortest distance:   {summary['min_distance_km']:.2f} km")
        print(f"  Average speed:       {summary['avg_speed_kmh']:.1f} km/h")

    print("=" * 50)


def get_detailed_trip_report(trips_df):
    """
    Generate a detailed report of all trips.

    Parameters:
    -----------
    trips_df : pandas.DataFrame
        DataFrame with trip statistics

    Returns:
    --------
    str
        Formatted string with detailed trip information
    """
    report_lines = []
    report_lines.append("\nDetailed Trip Report")
    report_lines.append("=" * 60)

    if len(trips_df) == 0:
        report_lines.append("No trips identified.")
        return "\n".join(report_lines)

    for _, trip in trips_df.iterrows():
        report_lines.append(
            f"Trip {int(trip['trip_id']):2d}: "
            f"{trip['duration_sec']:5.0f} sec ({trip['duration_min']:5.1f} min), "
            f"Distance: {trip['distance_km']:5.2f} km, "
            f"Avg speed: {trip['avg_speed_kmh']:5.1f} km/h, "
            f"Max speed: {trip['max_speed_kmh']:5.1f} km/h"
        )

    report_lines.append("=" * 60)
    return "\n".join(report_lines)


# Example usage
if __name__ == "__main__":
    # Test the functions
    from data_loader import load_data
    from mode_detector import detect_driving_modes

    print("Testing trip_analyzer module...")
    df = load_data()
    df = detect_driving_modes(df)
    df = identify_trips(df)
    trips_df, summary = calculate_trip_statistics(df)
    print_trip_summary(trips_df, summary)
    print(get_detailed_trip_report(trips_df))