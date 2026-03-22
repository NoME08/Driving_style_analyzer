#!/usr/bin/env python3
"""
UDDS Urban Driving Cycle Analysis - Refactored Version

This script analyzes UDDS driving data using modular components in the src/ directory.
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, print_summary
from src.mode_detector import detect_driving_modes, print_mode_statistics
from src.trip_analyzer import identify_trips, calculate_trip_statistics, print_trip_summary, get_detailed_trip_report
from src.visualizer import plot_main_analysis, plot_detailed_statistics

import matplotlib.pyplot as plt


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("UDDS Urban Driving Cycle Analysis (Refactored)")
    print("=" * 70)

    # 1. Load and preprocess data
    print("\n[1] Loading data...")
    # Build path to data file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "udds.csv")
    df = load_data(data_path)
    print_summary(df)

    # 2. Detect driving modes
    print("\n[2] Detecting driving modes...")
    df = detect_driving_modes(df)
    print_mode_statistics(df)

    # 3. Identify trips
    print("\n[3] Identifying trips...")
    df = identify_trips(df)
    trips_df, trips_summary = calculate_trip_statistics(df)
    print_trip_summary(trips_df, trips_summary)

    # 4. Print detailed trip report
    print(get_detailed_trip_report(trips_df))

    # 5. Create visualizations
    print("\n[4] Creating visualizations...")

    # Build save paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_plot_path = os.path.join(script_dir, '..', 'output', 'udds_metric_analysis.png')
    detailed_plot_path = os.path.join(script_dir, '..', 'output', 'udds_detailed_statistics.png')

    # Main analysis plot
    plot_main_analysis(df, trips_df, save_path=main_plot_path)

    # Detailed statistics plot
    plot_detailed_statistics(df, trips_df, save_path=detailed_plot_path)

    # 6. Print key metrics
    print("\n" + "=" * 70)
    print("UDDS Urban Driving Cycle Key Metrics")
    print("=" * 70)
    print(f"Total duration:        {df['time_sec'].max():.0f} seconds ({df['time_sec'].max() / 60:.1f} minutes)")
    print(f"Total distance:        {trips_summary['total_distance_km']:.2f} km")
    print(f"Maximum speed:         {df['speed_kmh'].max():.1f} km/h")
    print(f"Average speed:         {df['speed_kmh'].mean():.1f} km/h")
    print(f"Maximum acceleration:  {df['acceleration'].max():.2f} km/h/s")
    print(f"Maximum deceleration:  {df['acceleration'].min():.2f} km/h/s")
    print(f"Number of trips:       {trips_summary['total_trips']}")
    print(f"Average trip length:   {trips_summary['avg_duration_sec']:.0f} seconds "
          f"({trips_summary['avg_duration_sec'] / 60:.1f} minutes)")
    print(f"Longest trip:          {trips_summary['max_duration_sec']:.0f} seconds "
          f"({trips_summary['max_duration_sec'] / 60:.1f} minutes)")
    print("=" * 70)

    print("\nAnalysis complete!")
    print("Visualizations saved as:")
    print("  - Day_02/udds_metric_analysis.png")
    print("  - Day_02/udds_detailed_statistics.png")

    # Show plots (this will block until plots are closed)
    print("\nDisplaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
