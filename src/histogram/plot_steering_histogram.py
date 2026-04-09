import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

forwards_csv  = os.path.join(base_dir, "..", "..", "training", "training_data_forwards", "driving_log.csv")
backwards_csv = os.path.join(base_dir, "..", "..", "training", "training_data_backwards", "driving_log.csv")
unstable_csv  = os.path.join(base_dir, "..", "..", "training", "training_data_forwards_backwards_unstable", "driving_log.csv")

cols = ["center", "steering", "throttle", "brake", "speed"]

df_fwd      = pd.read_csv(forwards_csv,  header=None, names=cols)
df_bwd      = pd.read_csv(backwards_csv, header=None, names=cols)
df_unstable = pd.read_csv(unstable_csv,  header=None, names=cols)

df_all = pd.concat([df_fwd, df_bwd, df_unstable], ignore_index=True)

# steering values are already normalized to [-1, 1] by the sim
steering = df_all["steering"].astype(float)

print(f"Total samples     : {len(steering)}")
print(f"  Forwards        : {len(df_fwd)}")
print(f"  Backwards       : {len(df_bwd)}")
print(f"  Unstable        : {len(df_unstable)}")
print(f"Steering angle range (normalized): [{steering.min():.4f}, {steering.max():.4f}]")
print(f"Mean  : {steering.mean():.4f}")
print(f"Std   : {steering.std():.4f}")


BALANCE_THRESHOLD = 1000
NUM_BINS = 25

fig, ax = plt.subplots(figsize=(8, 5))

n, bins, patches = ax.hist(
    steering,
    bins=NUM_BINS,
    range=(-1.0, 1.0),
    color="steelblue",
    edgecolor="white",
    linewidth=0.4,
)

ax.axhline(BALANCE_THRESHOLD, color="steelblue", linestyle="-", linewidth=1.5)

ax.set_xlim(-1.0, 1.0)
ax.set_xlabel("Steering Angle", fontsize=11)
ax.set_ylabel("Number of Samples", fontsize=11)
ax.set_title("Steering Angle Distribution", fontsize=13, fontweight="bold")

plt.tight_layout()
output_path = os.path.join(base_dir, "..", "..", "docs", "steering_histogram.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nHistogram saved to: {output_path}")
plt.show()
