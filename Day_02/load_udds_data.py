import pandas as pd
import matplotlib.pyplot as plt

print("Loading UDDS data...")

# read the data from udds file
df = pd.read_csv("uddscol.txt", skiprows=1, header=None, sep='\t', names=["time_sec", "speed_mph"])
df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
df["speed_mph"] = pd.to_numeric(df["speed_mph"], errors="coerce")

df = df.dropna()

print("Data loaded")

# convert mph to kmh
df["speed_kmh"] = df["speed_mph"] * 1.609344

print(f"\n Data information:")
print(f"Total time: {df['time_sec'].max(): .1f} seconds")
print(f"Data counts: {len(df)}")
print(f"Max speed: {df['speed_kmh'].max(): .1f} km/h / {df['speed_mph'].max(): .1f} mph")
print(f"Min speed: {df['speed_kmh'].min(): .1f} km/h / {df['speed_mph'].min(): .1f} mph")
print(f"Mean speed: {df['speed_kmh'].mean(): .1f} km/h / {df['speed_mph'].mean(): .1f} mph")

df.to_csv("udds.csv", index=False)
print("Data saved to udds.csv")

# visualized
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(df["time_sec"], df["speed_mph"], label="mph", linewidth=2)
ax1.set_title("UUDS Driving(mph)")
ax1.set_xlabel("Time(sec)")
ax1.set_ylabel("Speed(mph)")
ax1.legend()
ax1.grid(True, alpha=1)

ax2.plot(df["time_sec"], df["speed_kmh"], label="kmh", linewidth=2)
ax2.set_title("UUDS Driving(kmh)")
ax2.set_xlabel("Time(sec)")
ax2.set_ylabel("Speed(kmh)")
ax2.grid(True, alpha=0.5)
ax2.legend()
plt.show()
