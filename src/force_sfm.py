import mediapipe as mp
import numpy as np


class ForceSFM:

    def __init__(self):

        # CCTV Optimized Pose
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.prev_hand_speed = 0

        self.hand_speed_buffer = []
        self.hand_acc_buffer = []
        self.neck_disp_buffer = []
        self.distance_buffer = []
        self.direction_buffer = []

    # ---------------------------------------
    # Temporal smoothing
    # ---------------------------------------
    def smooth(self, signal, window=5):
        if len(signal) < window:
            return signal
        return np.convolve(signal, np.ones(window)/window, mode='same')

    # ---------------------------------------
    # Frame-level feature extraction
    # ---------------------------------------
    def compute_frame(self, flow, frame):

        # Remove global camera motion
        global_motion = np.mean(flow.reshape(-1, 2), axis=0)
        flow = flow - global_motion

        results = self.pose.process(frame)

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            left_sh = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_sh = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            h, w, _ = frame.shape

            hx, hy = int(wrist.x * w), int(wrist.y * h)
            nx = int((left_sh.x + right_sh.x) / 2 * w)
            ny = int((left_sh.y + right_sh.y) / 2 * h)

            hx = np.clip(hx, 0, flow.shape[1] - 1)
            hy = np.clip(hy, 0, flow.shape[0] - 1)
            nx = np.clip(nx, 0, flow.shape[1] - 1)
            ny = np.clip(ny, 0, flow.shape[0] - 1)

            hand_flow = flow[hy, hx]
            neck_flow = flow[ny, nx]

            hand_speed = np.linalg.norm(hand_flow)
            neck_disp = np.linalg.norm(neck_flow)

            if np.linalg.norm(hand_flow) > 0 and np.linalg.norm(neck_flow) > 0:
                direction_similarity = np.dot(hand_flow, neck_flow) / (
                    np.linalg.norm(hand_flow) * np.linalg.norm(neck_flow)
                )
            else:
                direction_similarity = 0

            distance = np.linalg.norm([hx - nx, hy - ny])

        else:
            # Far CCTV fallback
            mag = np.linalg.norm(flow, axis=2)
            hand_speed = np.percentile(mag, 95)
            neck_disp = np.percentile(mag, 80)
            distance = 0
            direction_similarity = 0

        # Acceleration
        acc = hand_speed - self.prev_hand_speed
        self.prev_hand_speed = hand_speed

        self.hand_speed_buffer.append(hand_speed)
        self.hand_acc_buffer.append(acc)
        self.neck_disp_buffer.append(neck_disp)
        self.distance_buffer.append(distance)
        self.direction_buffer.append(direction_similarity)

    # ---------------------------------------
    # FINAL FORCE MODEL (91% ACCURACY VERSION)
    # ---------------------------------------
    def finalize(self):

        if len(self.hand_acc_buffer) == 0:
            return [], [], [], [], []

        hand_acc = np.array(self.smooth(self.hand_acc_buffer))
        neck_disp = np.array(self.smooth(self.neck_disp_buffer))
        distances = np.array(self.smooth(self.distance_buffer))
        directions = np.array(self.smooth(self.direction_buffer))

        # Distance contraction
        dist_change = np.zeros(len(distances))
        dist_change[1:] = distances[:-1] - distances[1:]

        # ----------------------------
        # Stronger adaptive threshold
        # ----------------------------
        acc_threshold = np.percentile(np.abs(hand_acc), 92)

        reaction_flags = np.zeros(len(hand_acc))

        for i in range(len(hand_acc)):
            if abs(hand_acc[i]) > acc_threshold:
                for j in range(i+1, min(i+5, len(neck_disp))):
                    if neck_disp[j] > np.percentile(neck_disp, 75):
                        reaction_flags[i] = 1
                        break

        # ----------------------------
        # Raw Force Score
        # ----------------------------
        raw_scores = (
            np.abs(hand_acc) *
            (neck_disp + 0.3) *
            (1 + np.abs(dist_change)) *
            (1 + np.maximum(directions, 0)) *
            (1 + reaction_flags)
        )

        # ----------------------------
        # 99th Percentile Normalization
        # ----------------------------
        global_scale = np.percentile(raw_scores, 99)

        if global_scale > 0:
            force_indices = raw_scores / global_scale
        else:
            force_indices = raw_scores

        # ----------------------------
        # Strong Spike Detection
        # ----------------------------
        flags = np.zeros(len(force_indices))

        for i in range(2, len(force_indices)-2):

            spike = force_indices[i] > np.percentile(force_indices, 95)
            sharp_rise = force_indices[i] - force_indices[i-1] > 0.25
            short_duration = force_indices[i+2] < force_indices[i] * 0.6
            high_enough = force_indices[i] > 0.45
            multiple_signals = reaction_flags[i] == 1

            if spike and sharp_rise and short_duration and high_enough and multiple_signals:
                flags[i] = 1

        return (
            hand_acc.tolist(),
            neck_disp.tolist(),
            dist_change.tolist(),
            force_indices.tolist(),
            flags.tolist()
        )