import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load driving data from CSV file. Auto-detects format:
      - UDDS format: columns 'time_sec' and 'speed_mph' (comma-delimited)
      - Car Scanner OBD format: columns SECONDS;PID;VALUE;UNITS;... (semicolon-delimited)

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Standardized DataFrame with columns:
            time_sec, speed_kmh, acceleration [, lat, lon, rpm, throttle_pct]
    """
    print(f"Loading driving data from {file_path}...")

    # --- Detect format by reading first line ---
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline().strip()

    if first_line.startswith('"SECONDS"') or first_line.startswith('SECONDS'):
        return _load_car_scanner(file_path)
    else:
        return _load_udds(file_path)


def _load_udds(file_path):
    """Load UDDS format CSV (comma-delimited, columns: time_sec, speed_mph)."""
    df = pd.read_csv(file_path)

    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df["speed_mph"] = pd.to_numeric(df["speed_mph"], errors="coerce")
    df = df.dropna(subset=["time_sec", "speed_mph"])

    df["speed_kmh"] = df["speed_mph"] * 1.609344
    df["acceleration"] = np.gradient(df["speed_kmh"], df["time_sec"])

    print(f"Loaded {len(df)} data points (UDDS format)")
    print(f"Time range: {df['time_sec'].min():.1f} - {df['time_sec'].max():.1f} seconds")
    print(f"Speed range: {df['speed_kmh'].min():.1f} - {df['speed_kmh'].max():.1f} km/h")
    return df


def _load_car_scanner(file_path):
    """
    Load Car Scanner OBD CSV format.

    Car Scanner exports data in "long" format:
        SECONDS;PID;VALUE;UNITS;LATITUDE;LONGTITUDE
    where each row is one PID reading at one timestamp.

    We pivot on GPS speed as the primary time axis and merge
    other PIDs (RPM, throttle) when timestamps align closely.
    """
    df = pd.read_csv(file_path, sep=';', quotechar='"')

    # Strip quotes from column names (Car Scanner sometimes wraps them)
    df.columns = df.columns.str.strip('"').str.strip()

    # --- Identify available PIDs ---
    pids_available = df['PID'].unique()
    print(f"Detected Car Scanner OBD format, {len(df)} rows, {len(pids_available)} PIDs")

    # --- Priority order for speed source ---
    speed_pid = None
    for candidate in ['车速', '速度 (GPS)', 'GPS Speed', 'Speed (OBD)']:
        if candidate in pids_available:
            rows = df[df['PID'] == candidate]
            vals = pd.to_numeric(rows['VALUE'], errors='coerce')
            if vals.max() > 0:
                speed_pid = candidate
                break

    if speed_pid is None:
        raise ValueError(
            "No usable speed PID found in CSV. "
            f"Available PIDs: {list(pids_available)}"
        )

    print(f"Using speed source: '{speed_pid}'")

    # --- Build primary DataFrame from speed PID ---
    # Select columns that exist (GPS may be missing)
    base_cols = ['SECONDS', 'VALUE']
    extra_cols = [c for c in ['LATITUDE', 'LONGTITUDE'] if c in df.columns]
    speed_df = df[df['PID'] == speed_pid][base_cols + extra_cols].copy()

    speed_df['VALUE'] = pd.to_numeric(speed_df['VALUE'], errors='coerce')
    for c in extra_cols:
        speed_df[c] = pd.to_numeric(speed_df[c], errors='coerce')
    speed_df = speed_df.sort_values('SECONDS').reset_index(drop=True)

    col_names = ['seconds_abs', 'speed_kmh'] + [c.lower() for c in extra_cols]
    speed_df.columns = col_names

    # --- Merge auxiliary PIDs ---
    aux_pids = {
        '发动机转速': 'rpm',
        '节气门位置': 'throttle_pct',
        '计算的瞬时燃油率': 'fuel_rate_lph',
        '进气温度': 'intake_temp_c',
        '冷却液温度': 'coolant_temp_c',
    }

    for pid_name, col_name in aux_pids.items():
        if pid_name in pids_available:
            aux = df[df['PID'] == pid_name][['SECONDS', 'VALUE']].copy()
            aux['VALUE'] = pd.to_numeric(aux['VALUE'], errors='coerce')
            aux = aux.sort_values('SECONDS')
            aux.columns = ['seconds_abs', col_name]

            speed_df = pd.merge_asof(
                speed_df, aux,
                on='seconds_abs',
                direction='nearest',
                tolerance=3.0
            )

    # --- Normalize time to start from 0 ---
    t0 = speed_df['seconds_abs'].iloc[0]
    speed_df['time_sec'] = speed_df['seconds_abs'] - t0

    # --- Compute acceleration from speed ---
    speed_df['acceleration'] = np.gradient(
        speed_df['speed_kmh'], speed_df['time_sec']
    )

    # Drop rows with no speed
    speed_df = speed_df.dropna(subset=['speed_kmh']).reset_index(drop=True)

    print(f"Loaded {len(speed_df)} data points (Car Scanner OBD format)")
    print(f"Duration: {speed_df['time_sec'].iloc[-1]:.1f} seconds "
          f"({speed_df['time_sec'].iloc[-1] / 60:.1f} minutes)")
    print(f"Speed range: {speed_df['speed_kmh'].min():.1f} - "
          f"{speed_df['speed_kmh'].max():.1f} km/h")
    print(f"Avg speed: {speed_df['speed_kmh'].mean():.1f} km/h")

    # Report merged PIDs
    extra_cols = [c for c in aux_pids.values() if c in speed_df.columns]
    if extra_cols:
        available = [c for c in extra_cols if speed_df[c].notna().any()]
        missing = [c for c in extra_cols if not speed_df[c].notna().any()]
        if available:
            print(f"Merged OBD data: {', '.join(available)}")
        if missing:
            print(f"Note: PIDs not aligned with speed timestamps: {', '.join(missing)}")

    return speed_df


def get_summary(df):
    """
    Generate summary statistics for the driving data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns: time_sec, speed_kmh, acceleration

    Returns
    -------
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

    Parameters
    ----------
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
