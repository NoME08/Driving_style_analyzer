import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Start analyzing UDDS data...")

# read data
df = pd.read_csv("udds.csv")
print(f"Load {len(df)} points")

# Calculate acceleration
df['speed_kmh'] = df['speed_mph'] * 1.609344
df['acceleration'] = np.gradient(df['speed_kmh'], df['time_sec'])
print(f"Speed range: {df['speed_kmh'].min(): .1f} - {df['speed_kmh'].max(): .1f} km/h")

# Recognise driving mode
df['driving_mode'] = 'cruise'

# Threshold setting
ACCEL_THRESHOLD = 3.0
BRAKE_THRESHOLD = -3.0
STOP_THRESHOLD = 0.8

df.loc[(df['acceleration'] > ACCEL_THRESHOLD), 'driving_mode'] = 'accel'
df.loc[(df['acceleration'] < BRAKE_THRESHOLD), 'driving_mode'] = 'decel'
df.loc[(df['speed_kmh'] < STOP_THRESHOLD), 'driving_mode'] = 'stop'

# Mode Statistics
print(f"Driving mode counts: {df['driving_mode'].value_counts()}")
mode_stats = df['driving_mode'].value_counts()
for mode in ['stop', 'accel', 'decel', 'cruise']:
    if mode in mode_stats:
        count = mode_stats[mode]
        percentage = 100 * count / len(df)
        print(f"{mode}: {count} ({percentage:.1f}%)")

# Identify trips
df['is_stopped'] = df['driving_mode'] == 'stop'
df['trip_id'] = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()
df.loc[df['is_stopped'], 'trip_id'] = -1

# Calculate trip statistics
trips = []
for trip_id in df[df['trip_id'] > 0]['trip_id'].unique():
    trip_data = df[df['trip_id'] == trip_id]
    if len(trip_data) > 0:
        # Calculate distance (km) = speed(km/h) * time(h)
        # 使用 numpy.trapezoid 替代废弃的 numpy.trapz
        distance = np.trapezoid(trip_data['speed_kmh'], trip_data['time_sec']) / 3600

        trips.append({
            'trip_id': len(trips) + 1,
            'duration_sec': len(trip_data),
            'duration_min': len(trip_data) / 60,
            'max_speed_kmh': trip_data['speed_kmh'].max(),
            'avg_speed_kmh': trip_data['speed_kmh'].mean(),
            'distance_km': distance
        })

trips_df = pd.DataFrame(trips)
print(f"\n Identified {len(trips_df)} individual trips")
print(
    f"  Longest trip: {trips_df['duration_sec'].max():.0f} seconds ({trips_df['duration_sec'].max() / 60:.1f} minutes)")
print(f"  Shortest trip: {trips_df['duration_sec'].min():.0f} seconds")
print(f"  Average trip: {trips_df['duration_sec'].mean():.0f} seconds")

# Total distance
total_distance = trips_df['distance_km'].sum()
print(f"\n Total distance: {total_distance:.1f} km")

# Visualization
fig = plt.figure(figsize=(15, 10))  # issue_1 the height was 0
fig.suptitle('UDDS Urban Driving Cycle Analysis (Metric Units)', fontsize=14, fontweight='bold')

ax1 = plt.subplot(3, 1, 1)
colors = {'stop': 'gray', 'cruise': 'blue', 'accel': 'red', 'decel': 'orange'}  # 修正颜色字典的键名
for mode, color in colors.items():
    mask = df['driving_mode'] == mode
    if mask.any():
        ax1.scatter(df.loc[mask, 'time_sec'] / 60, df.loc[mask, 'speed_kmh'],
                    c=color, label=mode.replace('_', ' ').title(), s=1, alpha=0.5)

ax1.set_ylabel('Speed (km/h)')
ax1.set_title('Speed with Driving Modes')
ax1.legend(markerscale=5)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, df['time_sec'].max() / 60)

# Visual acceleration
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df['time_sec'] / 60, df['acceleration'], 'b-', linewidth=0.8, alpha=0.5)
ax2.axhline(y=3, color='r', linestyle='--', alpha=0.8, label='Hard accel threshold (+3 km/h/s)')
ax2.axhline(y=-3, color='orange', linestyle='--', alpha=0.8, label='Hard brake threshold (-3 km/h/s)')
ax2.fill_between(df['time_sec'] / 60, 3, df['acceleration'],
                 where=(df['acceleration'] > 3), color='red', alpha=0.3, interpolate=True)
ax2.fill_between(df['time_sec'] / 60, -3, df['acceleration'],
                 where=(df['acceleration'] < -3), color='orange', alpha=0.3, interpolate=True)
ax2.set_ylabel('Acceleration (km/h/s)')
ax2.set_title('Acceleration Profile')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, df['time_sec'].max() / 60)

# Visual distance
ax3 = plt.subplot(3, 1, 3)
cumulative_distance = np.cumsum(df['speed_kmh'] * np.gradient(df['time_sec'])) / 3600
ax3.plot(df['time_sec'] / 60, cumulative_distance, 'g-', linewidth=2)
ax3.set_xlabel('Time (minutes)')
ax3.set_ylabel('Cumulative Distance (km)')
ax3.set_title(f'Cumulative Distance ({total_distance:.2f} km total)')
ax3.grid(True, alpha=0.3)
ax3.fill_between(df['time_sec'] / 60, 0, cumulative_distance, alpha=0.2, color='green')
ax3.set_xlim(0, df['time_sec'].max() / 60)

plt.tight_layout()
plt.savefig('udds_metric_analysis.png', dpi=300)
plt.show()

# Addition plot
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('UDDS Driving Cycle Detailed Statistics', fontsize=14, fontweight='bold')

# histogram
ax = axes[0, 0]
ax.hist(df['speed_kmh'], bins=50, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(x=df['speed_kmh'].mean(), color='red', linestyle='--',
           label=f"Mean: {df['speed_kmh'].mean():.1f} km/h")
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Frequency (seconds)')
ax.set_title('Speed Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(df['acceleration'], bins=50, color='green', alpha=0.7, edgecolor='black')
ax.axvline(x=3, color='red', linestyle='--', label='Hard accel threshold')
ax.axvline(x=-3, color='orange', linestyle='--', label='Hard brake threshold')
ax.set_xlabel('Acceleration (km/h/s)')
ax.set_ylabel('Frequency (seconds)')
ax.set_title('Acceleration Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
bars = ax.bar(range(1, len(trips_df) + 1), trips_df['duration_min'], color='purple', alpha=0.7)
ax.set_xlabel('Trip Number')
ax.set_ylabel('Duration (minutes)')
ax.set_title('Trip Durations')
ax.grid(True, alpha=0.3, axis='y')

# Distance distribution
ax = axes[1, 1]
bars = ax.bar(range(1, len(trips_df) + 1), trips_df['distance_km'], color='orange', alpha=0.7)
ax.set_xlabel('Trip Number')
ax.set_ylabel('Distance (km)')
ax.set_title('Trip Distances')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Output
print("\n" + "=" * 70)
print("UDDS Urban Driving Cycle Key Metrics")
print("=" * 70)
print(f"Total duration:        {df['time_sec'].max():.0f} seconds ({df['time_sec'].max() / 60:.1f} minutes)")
print(f"Total distance:        {total_distance:.2f} km")
print(f"Maximum speed:         {df['speed_kmh'].max():.1f} km/h")
print(f"Average speed:         {df['speed_kmh'].mean():.1f} km/h")
print(f"Maximum acceleration:  {df['acceleration'].max():.2f} km/h/s")
print(f"Maximum deceleration:  {df['acceleration'].min():.2f} km/h/s")
print(f"Number of stops:       {len(trips_df)}")
print(
    f"Average trip length:   {trips_df['duration_sec'].mean():.0f} seconds ({trips_df['duration_sec'].mean() / 60:.1f} minutes)")
print(
    f"Longest trip:          {trips_df['duration_sec'].max():.0f} seconds ({trips_df['duration_sec'].max() / 60:.1f} minutes)")
print("=" * 70)
