import sys
import os
import torch
import numpy as np
import cv2
from collections import OrderedDict

from raft.core.raft import RAFT
from raft.core.utils.utils import InputPadder

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -----------------------------
# RAFT Model Loading
# -----------------------------
class Args:
    def __init__(self):
        self.small = False
        self.mixed_precision = False
        self.alternate_corr = False
        self.dropout = 0

    def __contains__(self, key):
        return hasattr(self, key)

args = Args()
model = RAFT(args)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "raft-sintel.pth"
)

state_dict = torch.load(MODEL_PATH, map_location=device)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.to(device).eval()

print("RAFT loaded successfully")

# -----------------------------
# Optical Flow
# -----------------------------
def compute_raft_flow(frame1, frame2):

    frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()[None].to(device)
    frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()[None].to(device)

    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)

    with torch.no_grad():
        _, flow = model(frame1, frame2, iters=20, test_mode=True)

    flow = padder.unpad(flow)
    flow = flow[0].permute(1, 2, 0).cpu().numpy()

    return flow


# -----------------------------
# Orientation Histogram (MBH)
# -----------------------------
def orientation_histogram(dx, dy, bins=8, mag_thresh=0.01):

    mag = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx) + np.pi

    mask = mag > mag_thresh
    ang = ang[mask]
    mag = mag[mask]

    hist, _ = np.histogram(
        ang,
        bins=bins,
        range=(0, 2*np.pi),
        weights=mag
    )

    hist = hist / (np.linalg.norm(hist) + 1e-6)

    return hist


# -----------------------------
# MBH Feature
# -----------------------------
def compute_mbh(flow, grid=(2,2), bins=8):

    u = flow[..., 0]
    v = flow[..., 1]

    du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    du_dy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    dv_dx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)

    h, w = u.shape
    gh, gw = grid
    cell_h, cell_w = h // gh, w // gw

    mbh_feature = []

    for i in range(gh):
        for j in range(gw):

            y1, y2 = i * cell_h, (i+1) * cell_h
            x1, x2 = j * cell_w, (j+1) * cell_w

            hist_u = orientation_histogram(
                du_dx[y1:y2, x1:x2],
                du_dy[y1:y2, x1:x2],
                bins
            )

            hist_v = orientation_histogram(
                dv_dx[y1:y2, x1:x2],
                dv_dy[y1:y2, x1:x2],
                bins
            )

            mbh_feature.extend(hist_u)
            mbh_feature.extend(hist_v)

    return np.array(mbh_feature)


# -----------------------------
# Temporal MBH Aggregation
# -----------------------------
FRAME_STRIDE = 2
WINDOW_SIZE = 10

def extract_video_mbh(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return []

    ret, prev = cap.read()
    if not ret:
        print("Cannot read first frame")
        return []

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(prev, (320,320))

    window_buffer = []
    all_descriptors = []
    frame_count = 0

    while True:

        ret, curr = cap.read()
        if not ret:
            break

        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        curr = cv2.resize(curr, (320,320))

        frame_count += 1

        if frame_count % FRAME_STRIDE != 0:
            prev = curr
            continue

        flow = compute_raft_flow(prev, curr)

        mbh_vec = compute_mbh(flow)

        window_buffer.append(mbh_vec)

        if len(window_buffer) == WINDOW_SIZE:
            descriptor = np.mean(window_buffer, axis=0)
            all_descriptors.append(descriptor)
            window_buffer = []

        prev = curr

    cap.release()

    return np.array(all_descriptors)
