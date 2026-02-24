import sys
import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# Project Root Setup
# ------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.raft_mbh import compute_raft_flow, compute_mbh
from src.force_sfm import ForceSFM

# ------------------------------------
# GPU AUTO DETECTION
# ------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------------
# VIDEO ARGUMENT SUPPORT
# ------------------------------------
VIDEO_ARG = sys.argv[1] if len(sys.argv) > 1 else "snatching_1.mp4"
VIDEO_PATH = os.path.join(ROOT_DIR, "videos", VIDEO_ARG)

# ------------------------------------
# Load Video
# ------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

force_model = ForceSFM()

ret, prev = cap.read()
if not ret:
    print("❌ Cannot read first frame")
    exit()

prev = cv2.resize(prev, (320, 320))
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)

print("🚀 Processing video...")

# ------------------------------------
# Frame Processing Loop
# ------------------------------------
while True:
    ret, curr = cap.read()
    if not ret:
        break

    curr_resized = cv2.resize(curr, (320, 320))
    curr_rgb = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2RGB)

    # Optical Flow (RAFT)
    flow = compute_raft_flow(prev, curr_rgb)

    # Optional MBH extraction (can be used later)
    _ = compute_mbh(flow)

    # Force-SfM modeling
    force_model.compute_frame(flow, curr_rgb)

    prev = curr_rgb

cap.release()

print("✅ Video processing complete")

# ------------------------------------
# Final Force Modeling
# ------------------------------------
hand_acc, neck_disp, dist_change, force_indices, flags = force_model.finalize()

if len(force_indices) == 0:
    print("No motion detected.")
    exit()

fi_array = np.array(force_indices)
flag_indices = [i for i, f in enumerate(flags) if f == 1]

# ------------------------------------
# Metrics Calculation
# ------------------------------------
mean_force = np.mean(fi_array)
max_force = np.max(fi_array)
spike_ratio = max_force / (mean_force + 1e-6)
flag_count = len(flag_indices)

clustered_flags = 0
if flag_count >= 2:
    for i in range(flag_count - 1):
        if flag_indices[i + 1] - flag_indices[i] <= 15:
            clustered_flags += 1

motion_density = np.sum(
    fi_array > np.percentile(fi_array, 80)
) / len(fi_array)

duration = 0
if flag_count >= 2:
    duration = flag_indices[-1] - flag_indices[0]

# ------------------------------------
# FINAL HIGH-ACCURACY DETECTION LOGIC
# ------------------------------------
event_detected = 0

if (
    spike_ratio > 35 and        # strong sudden spike
    3 <= flag_count <= 8 and    # avoid long fights
    clustered_flags >= 2 and    # short burst cluster
    motion_density < 0.20 and   # not sustained motion
    duration < 40               # short duration event
):
    event_detected = 1

# ------------------------------------
# Print Debug Results
# ------------------------------------
print("\n-----------------------------")
print(f"Max force_index: {max_force:.3f}")
print(f"Mean force_index: {mean_force:.3f}")
print(f"Spike Ratio: {spike_ratio:.1f}")
print(f"Flag Count: {flag_count}")
print(f"Clustered Flags: {clustered_flags}")
print(f"Motion Density: {motion_density:.3f}")
print(f"Duration: {duration}")
print("-----------------------------")

print("\n🚨 Snatching Event Detected:", event_detected)

if event_detected:
    print("Event Frames:", flag_indices[:10])

# ------------------------------------
# Output Table (Optional)
# ------------------------------------
df = pd.DataFrame({
    "frame_id": range(len(force_indices)),
    "force_index": np.round(force_indices, 3),
    "flag": flags
})

print("\n🔎 Sample Output:")
print(df.head(20))

# ------------------------------------
# Plot Force Index
# ------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(force_indices)
plt.axhline(y=np.percentile(force_indices, 96), color='r', linestyle='--')
plt.title("Force Index Over Time")
plt.xlabel("Frame")
plt.ylabel("Force Index")
plt.show()