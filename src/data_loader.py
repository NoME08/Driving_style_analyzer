import pandas as pd
import numpy as np


def load_data(file_path="../data/udds.csv"):
    """
    Load UDDS driving data from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing columns 'time_sec' and 'speed_mph'

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: time_sec, speed_mph, speed_kmh, acceleration
    """
    print(f"Loading driving data from {file_path}...")

    # Read data
    df = pd.read_csv(file_path)

    # Ensure numeric columns
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df["speed_mph"] = pd.to_numeric(df["speed_mph"], errors="coerce")

    # Drop any rows with missing values
    df = df.dropna(subset=["time_sec", "speed_mph"])

    # Convert mph to km/h
    df["speed_kmh"] = df["speed_mph"] * 1.609344

    # Calculate acceleration (km/h per second)
    df["acceleration"] = np.gradient(df["speed_kmh"], df["time_sec"])

    print(f"Loaded {len(df)} data points")
    print(f"Time range: {df['time_sec'].min():.1f} - {df['time_sec'].max():.1f} seconds")
    print(f"Speed range: {df['speed_kmh'].min():.1f} - {df['speed_kmh'].max():.1f} km/h")

    return df


def get_summary(df):
    """
    Generate summary statistics for the driving data.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: time_sec, speed_kmh, acceleration

    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    summary = {
        "total_time_sec": df["time_sec"].max() - df["time_sec"].min(),
        "total_points": len(df),
        "max_speed_kmh": df["speed_kmh"].max(),
        "min_speed_kmh": df["speed_kmh"].min(),
        "mean_speed_kmh": df["speed_kmh"].mean(),
        "max_acceleration": df["acceleration"].max(),
        "min_acceleration": df["acceleration"].min(),
        "mean_acceleration": df["acceleration"].mean(),
    }
    return summary


def print_summary(df):
    """
    Print summary statistics for the driving data.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: time_sec, speed_kmh, acceleration
    """
    summary = get_summary(df)

    print("\n" + "=" * 50)
    print("Driving Data Summary")
    print("=" * 50)
    print(f"Total duration:      {summary['total_time_sec']:.1f} seconds")
    print(f"Data points:         {summary['total_points']}")
    print(f"Max speed:           {summary['max_speed_kmh']:.1f} km/h")
    print(f"Min speed:           {summary['min_speed_kmh']:.1f} km/h")
    print(f"Mean speed:          {summary['mean_speed_kmh']:.1f} km/h")
    print(f"Max acceleration:    {summary['max_acceleration']:.2f} km/h/s")
    print(f"Min acceleration:    {summary['min_acceleration']:.2f} km/h/s")
    print(f"Mean acceleration:   {summary['mean_acceleration']:.2f} km/h/s")
    print("=" * 50)


# For backward compatibility
def get_data():
    """
    Legacy function for backward compatibility.
    Loads data and returns DataFrame.
    """
    df = load_data()
    return df
