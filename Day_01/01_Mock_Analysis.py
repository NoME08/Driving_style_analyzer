import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(" Driving Data Analysis Lab Started Successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

# Generate simulated driving data (like a real car driving)
# Assume recording every second for 5 minutes (300 seconds)

# Set random seed for reproducibility
np.random.seed(42)

# Create time axis (seconds)
time = np.arange(0, 300)

# Generate simulated speed data (km/h)
# Basic pattern: accelerate - cruise - decelerate - accelerate - cruise - decelerate...
speed = np.zeros(300)

# 0-60 seconds: accelerate from 0 to 80 km/h
speed[0:60] = np.linspace(0, 80, 60) + np.random.normal(0, 1, 60)
# 60-120 seconds: cruise at 80 km/h with some fluctuations
speed[60:120] = 80 + np.random.normal(0, 2, 60)
# 120-180 seconds: decelerate to 0 (stop at red light)
speed[120:180] = np.linspace(80, 0, 60) + np.random.normal(0, 1, 60)
# 180-240 seconds: accelerate again to 60 km/h
speed[180:240] = np.linspace(0, 60, 60) + np.random.normal(0, 1, 60)
# 240-300 seconds: cruise at 60 km/h
speed[240:300] = 60 + np.random.normal(0, 1, 60)

# Ensure speed is not negative
speed = np.maximum(speed, 0)

print(f"Generated {len(time)} seconds of driving data")
print(f"Maximum speed: {speed.max():.1f} km/h")
print(f"Average speed: {speed.mean():.1f} km/h")

# Calculate acceleration (rate of change of speed)
acceleration = np.gradient(speed)

# Find hard acceleration and hard braking
hard_acceleration = acceleration[acceleration > 3]  # acceleration > 3 km/h/s
hard_braking = acceleration[acceleration < -3]      # deceleration < -3 km/h/s

print(f"Number of hard accelerations: {len(hard_acceleration)}")
print(f"Number of hard braking events: {len(hard_braking)}")

# Calculate driving aggressiveness score
aggressive_score = (len(hard_acceleration) + len(hard_braking)) / len(time) * 100
print(f"Driving aggressiveness score: {aggressive_score:.2f}%")
print(f"(Higher score indicates more aggressive driving)")

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Subplot 1: Speed vs Time
ax1.plot(time, speed, color='blue', linewidth=1.5)
ax1.set_title('Speed vs Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Speed (km/h)')
ax1.grid(True, alpha=0.3)
ax1.fill_between(time, 0, speed, alpha=0.1, color='blue')

# Subplot 2: Acceleration vs Time
ax2.plot(time, acceleration, color='red', linewidth=1)
ax2.axhline(y=3, color='green', linestyle='--', alpha=0.5, label='Hard accel threshold')
ax2.axhline(y=-3, color='orange', linestyle='--', alpha=0.5, label='Hard brake threshold')
ax2.fill_between(time, 3, acceleration, where=(acceleration>3),
                  color='green', alpha=0.3, label='Hard acceleration')
ax2.fill_between(time, -3, acceleration, where=(acceleration<-3),
                  color='orange', alpha=0.3, label='Hard braking')
ax2.set_title('Acceleration vs Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Acceleration (km/h/s)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Scatter plot with color representing acceleration
sc = ax3.scatter(time, speed, c=acceleration, cmap='RdYlGn', s=10, alpha=0.6)
ax3.set_title('Driving Style Scatter (Color = Acceleration)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Speed (km/h)')
plt.colorbar(sc, ax=ax3, label='Acceleration (km/h/s)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Congratulations! You've just completed your first driving data analysis!")
print("These charts show your simulated driving session:")
print("- Blue curve: Speed changes")
print("- Red curve: Acceleration changes")
print("- Green/Orange areas: Hard acceleration/braking locations")