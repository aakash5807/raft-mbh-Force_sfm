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
# VIDEO PATH
# ------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "..",
    "videos",
    "Chain_Snatching170.mp4"
)

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

prev = cv2.resize(prev, (256,256))
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)

mbh_features = []

print("🚀 Processing video...")

frame_count = 0

while True:
    ret, curr = cap.read()
    if not ret:
        break

    prev = cv2.resize(curr, (320,320))
    curr = cv2.resize(curr, (320,320))
    curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)

    # Optical Flow
    flow = compute_raft_flow(prev, curr_rgb)

    # MBH
    mbh_vec = compute_mbh(flow)
    mbh_features.append(mbh_vec)

    # Force-SfM
    force_model.compute_frame(flow, curr_rgb)

    prev = curr_rgb
    frame_count += 1

cap.release()

print("✅ Video processing complete")

# ------------------------------------
# Final Modeling
# ------------------------------------
hand_acc, neck_disp, dist_change, force_indices, flags = force_model.finalize()

# ------------------------------------
# Output Table
# ------------------------------------
df = pd.DataFrame({
    "frame_id": range(len(force_indices)),
    "hand_acc": np.round(hand_acc, 3),
    "neck_disp": np.round(neck_disp, 3),
    "distance_change": np.round(dist_change, 3),
    "force_index": np.round(force_indices, 3),
    "force_flag": flags
})

print("\n🔎 Sample Output:")
print(df.head(20))

# ------------------------------------
# Improved Event Detection (CCTV tuned)
# ------------------------------------
event_detected = 0
event_frames = []

for i in range(len(flags)):
    if flags[i] == 1:
        event_detected = 1
        event_frames.append(i)

print("\n🚨 Snatching Event Detected:", event_detected)

if event_detected:
    print("Event Frames:", event_frames[:10])

# ------------------------------------
# Plot
# ------------------------------------
plt.figure(figsize=(12,4))
plt.plot(force_indices)
plt.title("Force Index Over Time")
plt.xlabel("Frame")
plt.ylabel("Force Index")
plt.axhline(y=0.30, color='r', linestyle='--')
plt.show()