"""
SafeTurn AI — Phase 1: Predictive Free-Left Turn Safety System
==============================================================
HackSETU 2025 | Theme 3: Intelligent Free-Left Turn Management

"Predictive, not reactive — we act before danger arrives."

Pipeline: CCTV Input → YOLOv8 Detection → Centroid Tracking →
          Trajectory Prediction (3s) → Conflict Engine → Decision

Usage:
    python safeturn_phase1.py --video traffic.mp4
    python safeturn_phase1.py --mock              # No video / no YOLO needed
    python safeturn_phase1.py --mock --frames 60  # Run 60 frames then exit

Controls:
    Q — Quit    S — Screenshot
"""

import argparse
import math
import time
import sys
import collections
import numpy as np

# ─── OpenCV Import ──────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found. Install: pip install opencv-python")
    sys.exit(1)

# ─── YOLOv8 Import (graceful fallback) ──────────────────────────
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not found — running in MOCK detection mode.")
    print("       Install: pip install ultralytics")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Detection settings
CONFIDENCE_THRESHOLD = 0.35
# COCO class IDs we care about at a junction
TARGET_CLASSES = {
    0: "person",      # Pedestrian — our primary protected class
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
# Which classes are "vehicles" (everything except person)
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

# Tracking settings
MAX_DISAPPEARED_FRAMES = 30   # Deregister object after this many missed frames
HISTORY_LENGTH = 5            # Frames of position history for velocity calc

# Prediction settings
LOOKAHEAD_SECONDS = 3.0       # Predict 3 seconds into the future

# Conflict engine settings
CONFLICT_DISTANCE_PX = 80     # Predicted distance < this → high conflict
CONFLICT_DISTANCE_MAX = 250   # Beyond this → zero conflict contribution

# Decision thresholds
HOLD_THRESHOLD = 0.70         # 🔴 HOLD  — restrict the turn
CAUTION_THRESHOLD = 0.40      # 🟡 CAUTION — warn
                              # 🟢 ALLOW  — below caution threshold

# Colors (BGR for OpenCV)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_GREEN  = (0, 200, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (255, 255, 0)
COLOR_ORANGE = (0, 140, 255)

# Class colors for bounding boxes
CLASS_COLORS = {
    0: (255, 100, 100),   # person — blue-ish
    1: (100, 255, 100),   # bicycle — green
    2: (100, 100, 255),   # car — red
    3: (0, 200, 255),     # motorcycle — yellow
    5: (255, 0, 200),     # bus — pink
    7: (200, 150, 0),     # truck — teal
}


# ═══════════════════════════════════════════════════════════════════
# CENTROID TRACKER
# ═══════════════════════════════════════════════════════════════════
# Custom tracker — no external deps. Assigns persistent IDs to
# detected objects using centroid distance matching. Maintains
# position history per object for velocity computation.
# ═══════════════════════════════════════════════════════════════════

class CentroidTracker:
    """
    Tracks objects across frames by matching centroids.
    Each tracked object gets:
      - Unique ID
      - Class label
      - Bounding box
      - Position history (deque of last N centroids)
      - Disappeared counter
    """

    def __init__(self, max_disappeared=MAX_DISAPPEARED_FRAMES):
        self.next_id = 0
        self.objects = {}          # id → centroid (x, y)
        self.bboxes = {}           # id → (x1, y1, x2, y2)
        self.classes = {}          # id → class_id
        self.disappeared = {}     # id → frames since last seen
        self.histories = {}       # id → deque of (x, y)
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox, class_id):
        """Register a new object with a fresh ID."""
        oid = self.next_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.classes[oid] = class_id
        self.disappeared[oid] = 0
        self.histories[oid] = collections.deque(maxlen=HISTORY_LENGTH)
        self.histories[oid].append(centroid)
        self.next_id += 1
        return oid

    def deregister(self, oid):
        """Remove an object that has disappeared for too long."""
        for d in [self.objects, self.bboxes, self.classes,
                  self.disappeared, self.histories]:
            d.pop(oid, None)

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list of (centroid, bbox, class_id)
                centroid = (cx, cy)
                bbox = (x1, y1, x2, y2)
                class_id = int

        Returns:
            dict of {object_id: (centroid, bbox, class_id)}
        """
        # No detections this frame → increment disappeared for all
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self._build_result()

        # If we have no existing objects, register everything
        if len(self.objects) == 0:
            for centroid, bbox, class_id in detections:
                self.register(centroid, bbox, class_id)
            return self._build_result()

        # ─── Match detections to existing objects ───
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        det_centroids = [d[0] for d in detections]

        # Compute distance matrix between existing objects and new detections
        D = np.zeros((len(object_centroids), len(det_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, dc in enumerate(det_centroids):
                D[i, j] = math.dist(oc, dc)

        # Greedy matching: closest pairs first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            # Only match if distance is reasonable (< 150px)
            if D[row, col] > 150:
                continue

            oid = object_ids[row]
            centroid, bbox, class_id = detections[col]
            self.objects[oid] = centroid
            self.bboxes[oid] = bbox
            self.classes[oid] = class_id
            self.disappeared[oid] = 0
            self.histories[oid].append(centroid)

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        for row in range(len(object_centroids)):
            if row not in used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        # Handle unmatched detections (new objects)
        for col in range(len(det_centroids)):
            if col not in used_cols:
                centroid, bbox, class_id = detections[col]
                self.register(centroid, bbox, class_id)

        return self._build_result()

    def _build_result(self):
        """Build result dict: {id: (centroid, bbox, class_id)}"""
        result = {}
        for oid in self.objects:
            result[oid] = (
                self.objects[oid],
                self.bboxes[oid],
                self.classes[oid],
            )
        return result


# ═══════════════════════════════════════════════════════════════════
# TRAJECTORY PREDICTION
# ═══════════════════════════════════════════════════════════════════
# Uses linear extrapolation from the last N positions.
# Velocity = average displacement per frame.
# Predicted position = current pos + velocity * frames_ahead.
# ═══════════════════════════════════════════════════════════════════

def predict_position(history, fps):
    """
    Predict where an object will be in LOOKAHEAD_SECONDS.

    Args:
        history: deque of (x, y) positions (most recent last)
        fps: current processing FPS

    Returns:
        (pred_x, pred_y) — predicted position at T+3s
        (vx, vy) — velocity vector (pixels per frame)
        None, None if insufficient history
    """
    if len(history) < 2:
        return None, None

    positions = list(history)

    # Compute velocity as average displacement across history
    # This smooths out jitter from detection noise
    dx_total = positions[-1][0] - positions[0][0]
    dy_total = positions[-1][1] - positions[0][1]
    n_steps = len(positions) - 1

    # Velocity in pixels per frame
    vx = dx_total / n_steps
    vy = dy_total / n_steps

    # How many frames into the future = lookahead_seconds * fps
    fps = max(fps, 1)  # Guard against zero
    frames_ahead = LOOKAHEAD_SECONDS * fps

    # Predicted position
    pred_x = positions[-1][0] + vx * frames_ahead
    pred_y = positions[-1][1] + vy * frames_ahead

    return (pred_x, pred_y), (vx, vy)


# ═══════════════════════════════════════════════════════════════════
# CONFLICT ENGINE
# ═══════════════════════════════════════════════════════════════════
# For every pedestrian-vehicle pair:
#   1. Compute distance between PREDICTED positions (not current)
#   2. Factor in closing speed (are they converging?)
#   3. Factor in trajectory convergence angle
# Output: conflict_probability 0.0 to 1.0
# ═══════════════════════════════════════════════════════════════════

def compute_conflict_probability(ped_pred, ped_vel, veh_pred, veh_vel):
    """
    Compute the probability that a pedestrian and vehicle will
    conflict at their predicted positions in 3 seconds.

    Args:
        ped_pred: (x, y) predicted pedestrian position
        ped_vel: (vx, vy) pedestrian velocity vector
        veh_pred: (x, y) predicted vehicle position
        veh_vel: (vx, vy) vehicle velocity vector

    Returns:
        float: conflict probability 0.0–1.0
    """
    if ped_pred is None or veh_pred is None:
        return 0.0

    # ── Distance factor ──
    # Core signal: how close will they be at T+3s?
    dist = math.dist(ped_pred, veh_pred)
    if dist < CONFLICT_DISTANCE_PX:
        dist_factor = 1.0
    elif dist > CONFLICT_DISTANCE_MAX:
        dist_factor = 0.0
    else:
        # Linear interpolation between thresholds
        dist_factor = 1.0 - (dist - CONFLICT_DISTANCE_PX) / (CONFLICT_DISTANCE_MAX - CONFLICT_DISTANCE_PX)

    # ── Closing speed factor ──
    # Are they moving TOWARD each other? Relative velocity projected
    # onto the line connecting their current directions.
    if ped_vel is not None and veh_vel is not None:
        # Relative velocity of vehicle w.r.t. pedestrian
        rel_vx = veh_vel[0] - ped_vel[0]
        rel_vy = veh_vel[1] - ped_vel[1]
        closing_speed = math.sqrt(rel_vx**2 + rel_vy**2)
        # Normalize: higher closing speed → higher risk
        speed_factor = min(closing_speed / 8.0, 1.0)
    else:
        speed_factor = 0.5  # Unknown — assume moderate

    # ── Trajectory convergence factor ──
    # If both velocity vectors point toward the other's current
    # position, trajectories are converging → high risk.
    if ped_vel is not None and veh_vel is not None:
        ped_speed = math.sqrt(ped_vel[0]**2 + ped_vel[1]**2)
        veh_speed = math.sqrt(veh_vel[0]**2 + veh_vel[1]**2)
        if ped_speed > 0.1 and veh_speed > 0.1:
            # Direction vectors
            ped_dir = (ped_vel[0] / ped_speed, ped_vel[1] / ped_speed)
            veh_dir = (veh_vel[0] / veh_speed, veh_vel[1] / veh_speed)
            # Dot product of directions — if both point same-ish way, lower risk
            # If they point at each other (opposite), higher risk
            dot = ped_dir[0] * veh_dir[0] + ped_dir[1] * veh_dir[1]
            # dot = -1 means head-on (max risk), +1 means same direction (min risk)
            convergence_factor = (1.0 - dot) / 2.0  # Maps [-1,1] → [1,0]
        else:
            convergence_factor = 0.5
    else:
        convergence_factor = 0.5

    # ── Combine factors ──
    # Distance is the primary signal (60%), speed and convergence are secondary
    probability = (0.60 * dist_factor +
                   0.25 * speed_factor +
                   0.15 * convergence_factor)

    return round(min(max(probability, 0.0), 1.0), 3)


# ═══════════════════════════════════════════════════════════════════
# DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════

def get_decision(conflict_prob):
    """
    Convert conflict probability to actionable decision.

    Returns:
        (decision_str, color_bgr)
    """
    if conflict_prob > HOLD_THRESHOLD:
        return "HOLD", COLOR_RED
    elif conflict_prob > CAUTION_THRESHOLD:
        return "CAUTION", COLOR_YELLOW
    else:
        return "ALLOW", COLOR_GREEN


# ═══════════════════════════════════════════════════════════════════
# MOCK DETECTION GENERATOR
# ═══════════════════════════════════════════════════════════════════
# Synthetic objects that move across the frame so we can demo
# the full pipeline without a real video or YOLOv8 installed.
# ═══════════════════════════════════════════════════════════════════

class MockDetectionGenerator:
    """
    Generates synthetic detections simulating a junction scenario:
    - A pedestrian crossing from left to right
    - A car approaching from bottom-right making a free left turn
    This guarantees the conflict engine fires and the HOLD decision triggers.
    """

    def __init__(self, width=960, height=540):
        self.w = width
        self.h = height
        self.frame_count = 0

        # Pedestrian: walks right across the crossing
        self.ped_x = 100.0
        self.ped_y = self.h * 0.45  # Mid-height
        self.ped_vx = 2.5           # Steady walking speed
        self.ped_vy = 0.3           # Slight vertical drift

        # Vehicle: approaches from bottom-right, curves left (free-left turn)
        self.veh_x = self.w * 0.75
        self.veh_y = self.h * 0.85
        self.veh_vx = -3.0
        self.veh_vy = -2.5

    def generate(self):
        """Generate detections for one frame."""
        self.frame_count += 1

        # Update positions
        self.ped_x += self.ped_vx + np.random.normal(0, 0.3)
        self.ped_y += self.ped_vy + np.random.normal(0, 0.2)

        self.veh_x += self.veh_vx + np.random.normal(0, 0.5)
        self.veh_y += self.veh_vy + np.random.normal(0, 0.3)

        # Reset positions when they go off screen (loop the scenario)
        if self.ped_x > self.w + 50:
            self.ped_x = -30
            self.ped_y = self.h * 0.45 + np.random.uniform(-20, 20)

        if self.veh_x < -80 or self.veh_y < -80:
            self.veh_x = self.w * 0.75 + np.random.uniform(-30, 30)
            self.veh_y = self.h * 0.85 + np.random.uniform(-20, 20)

        detections = []

        # Pedestrian detection (class 0)
        px, py = int(self.ped_x), int(self.ped_y)
        pw, ph = 30, 70  # Person bbox size
        ped_bbox = (px - pw//2, py - ph//2, px + pw//2, py + ph//2)
        detections.append(((px, py), ped_bbox, 0))

        # Vehicle detection (class 2 = car)
        vx, vy = int(self.veh_x), int(self.veh_y)
        vw, vh = 100, 60  # Car bbox size
        veh_bbox = (vx - vw//2, vy - vh//2, vx + vw//2, vy + vh//2)
        detections.append(((vx, vy), veh_bbox, 2))

        # Add a second pedestrian near the crossing (sometimes)
        if self.frame_count % 120 < 80:
            p2x = int(self.ped_x + 60 + np.random.normal(0, 1))
            p2y = int(self.ped_y - 30 + np.random.normal(0, 1))
            p2_bbox = (p2x - pw//2, p2y - ph//2, p2x + pw//2, p2y + ph//2)
            detections.append(((p2x, p2y), p2_bbox, 0))

        # Add a motorcycle occasionally
        if self.frame_count % 200 < 100:
            mx = int(self.w * 0.55 + (self.frame_count % 200) * 2.5)
            my = int(self.h * 0.60 + np.random.normal(0, 1))
            m_bbox = (mx - 25, my - 20, mx + 25, my + 20)
            detections.append(((mx, my), m_bbox, 3))

        return detections

    def generate_frame(self):
        """Generate a visual background frame (gray road with lane markings)."""
        frame = np.full((self.h, self.w, 3), (60, 60, 60), dtype=np.uint8)

        # Draw road surface (darker gray)
        cv2.rectangle(frame, (0, int(self.h*0.3)),
                      (self.w, int(self.h*0.7)), (50, 50, 50), -1)

        # Crosswalk stripes
        stripe_y1 = int(self.h * 0.38)
        stripe_y2 = int(self.h * 0.52)
        for x in range(50, self.w - 50, 60):
            cv2.rectangle(frame, (x, stripe_y1),
                          (x + 30, stripe_y2), (200, 200, 200), -1)

        # "FREE LEFT" text on road
        cv2.putText(frame, "FREE LEFT", (self.w//2 - 80, int(self.h*0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # Junction label
        cv2.putText(frame, "JUNCTION CAM", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        return frame


# ═══════════════════════════════════════════════════════════════════
# OVERLAY RENDERER
# ═══════════════════════════════════════════════════════════════════

def draw_overlay(frame, tracked_objects, predictions, conflicts,
                 decision, decision_color, max_conflict_prob, fps, stats):
    """
    Draw the full SafeTurn overlay on the frame:
    - Bounding boxes with class labels and IDs
    - Predicted position markers
    - Conflict zone indicators
    - Decision banner (HOLD / CAUTION / ALLOW)
    - Probability bar
    - Stats panel
    """
    h, w = frame.shape[:2]

    # ── Bounding boxes + labels ──
    for oid, (centroid, bbox, class_id) in tracked_objects.items():
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(class_id, COLOR_WHITE)
        label = TARGET_CLASSES.get(class_id, "?")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label_text = f"#{oid} {label}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1)

        # Centroid dot
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, color, -1)

    # ── Predicted positions ──
    for oid, (pred_pos, vel) in predictions.items():
        if pred_pos is None:
            continue
        px, py = int(pred_pos[0]), int(pred_pos[1])
        class_id = tracked_objects[oid][2] if oid in tracked_objects else 0
        color = CLASS_COLORS.get(class_id, COLOR_WHITE)

        # Predicted position — dashed circle
        cv2.circle(frame, (px, py), 12, color, 1)
        cv2.circle(frame, (px, py), 3, color, -1)

        # Line from current to predicted (trajectory arrow)
        if oid in tracked_objects:
            cx, cy = tracked_objects[oid][0]
            cv2.arrowedLine(frame, (int(cx), int(cy)), (px, py),
                            color, 1, tipLength=0.15)

    # ── Conflict zones ──
    for (pid, vid), prob in conflicts.items():
        if prob < CAUTION_THRESHOLD:
            continue
        # Draw circle at midpoint of predicted positions
        if pid in predictions and vid in predictions:
            p1 = predictions[pid][0]
            p2 = predictions[vid][0]
            if p1 is not None and p2 is not None:
                mid_x = int((p1[0] + p2[0]) / 2)
                mid_y = int((p1[1] + p2[1]) / 2)
                radius = int(30 + prob * 40)
                zone_color = COLOR_RED if prob > HOLD_THRESHOLD else COLOR_YELLOW
                cv2.circle(frame, (mid_x, mid_y), radius, zone_color, 2)
                cv2.putText(frame, f"{prob:.0%}", (mid_x - 15, mid_y - radius - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

    # ── Decision banner (top center) ──
    banner_h = 60
    # Semi-transparent banner background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), decision_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Decision text
    cv2.putText(frame, f"SafeTurn: {decision}",
                (w // 2 - 130, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)

    # ── Probability bar (below banner) ──
    bar_y = banner_h + 5
    bar_h = 18
    bar_w = int(w * 0.4)
    bar_x = (w - bar_w) // 2
    fill_w = int(bar_w * max_conflict_prob)

    # Bar background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    # Bar fill
    if fill_w > 0:
        bar_color = COLOR_RED if max_conflict_prob > 0.7 else (
            COLOR_YELLOW if max_conflict_prob > 0.4 else COLOR_GREEN)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      bar_color, -1)
    # Bar border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  COLOR_WHITE, 1)
    # Probability text
    cv2.putText(frame, f"Conflict: {max_conflict_prob:.1%}",
                (bar_x + bar_w + 10, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # ── Stats panel (bottom-left) ──
    panel_y = h - 90
    cv2.rectangle(frame, (5, panel_y), (260, h - 5), (30, 30, 30), -1)
    cv2.rectangle(frame, (5, panel_y), (260, h - 5), (100, 100, 100), 1)

    cv2.putText(frame, f"Conflicts Prevented: {stats['holds']}",
                (15, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)
    cv2.putText(frame, f"Cautions Issued:     {stats['cautions']}",
                (15, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
    cv2.putText(frame, f"Turns Allowed:       {stats['allows']}",
                (15, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

    # ── FPS counter (bottom-right) ──
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    # ── SafeTurn branding (top-right) ──
    cv2.putText(frame, "SafeTurn AI", (w - 150, banner_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 2)
    cv2.putText(frame, "3s Predictive | HackSETU 2025", (w - 280, banner_h + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════
# YOLO DETECTION WRAPPER
# ═══════════════════════════════════════════════════════════════════

def run_yolo_detection(model, frame):
    """
    Run YOLOv8 on a frame, filter for target classes.

    Returns:
        list of (centroid, bbox, class_id)
    """
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append(
                ((cx, cy), (int(x1), int(y1), int(x2), int(y2)), cls)
            )

    return detections


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SafeTurn AI — Predictive Free-Left Turn Safety System"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file")
    parser.add_argument("--mock", action="store_true",
                        help="Run with synthetic mock detections (no video/YOLO needed)")
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after N frames (0 = run forever)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model path (default: yolov8n.pt)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display window (for testing)")
    args = parser.parse_args()

    # ── Mode selection ──
    use_mock = args.mock or (args.video is None and not YOLO_AVAILABLE)

    if use_mock:
        print("=" * 55)
        print("  SafeTurn AI — MOCK MODE")
        print("  Synthetic detections — no video/YOLO required")
        print("=" * 55)
        mock_gen = MockDetectionGenerator()
        cap = None
    else:
        if args.video is None:
            print("[ERROR] No video file specified. Use --video <path> or --mock.")
            sys.exit(1)

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {args.video}")
            print("        Falling back to MOCK mode.")
            use_mock = True
            mock_gen = MockDetectionGenerator()
            cap = None
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            print("=" * 55)
            print(f"  SafeTurn AI — VIDEO MODE")
            print(f"  Source: {args.video}")
            print(f"  Frames: {total_frames} | FPS: {vid_fps:.1f}")
            print("=" * 55)

    # ── Load YOLOv8 model ──
    model = None
    if not use_mock and YOLO_AVAILABLE:
        print(f"[INFO] Loading YOLOv8 model: {args.model}")
        try:
            model = YOLO(args.model)
            print("[INFO] YOLOv8 loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load YOLO: {e}")
            print("       Falling back to MOCK mode.")
            use_mock = True
            mock_gen = MockDetectionGenerator()

    # ── Initialize tracker and stats ──
    tracker = CentroidTracker()
    stats = {"holds": 0, "cautions": 0, "allows": 0}
    fps = 10.0  # Initial estimate
    frame_count = 0
    prev_decision = None

    print("\n[READY] Press Q to quit, S to screenshot.\n")

    # ═══════════════════════════════════
    # MAIN PROCESSING LOOP
    # ═══════════════════════════════════
    while True:
        loop_start = time.time()

        # ── Get frame ──
        if use_mock:
            frame = mock_gen.generate_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

        # ── Detection ──
        if use_mock:
            detections = mock_gen.generate()
        elif model is not None:
            detections = run_yolo_detection(model, frame)
        else:
            detections = []

        # ── Tracking ──
        tracked = tracker.update(detections)

        # ── Trajectory Prediction ──
        predictions = {}
        for oid in tracked:
            if oid in tracker.histories:
                pred_pos, vel = predict_position(tracker.histories[oid], fps)
                predictions[oid] = (pred_pos, vel)

        # ── Conflict Engine ──
        # Find all pedestrian IDs and vehicle IDs
        pedestrian_ids = [oid for oid, (_, _, cls) in tracked.items() if cls == 0]
        vehicle_ids = [oid for oid, (_, _, cls) in tracked.items() if cls in VEHICLE_CLASSES]

        conflicts = {}
        max_conflict_prob = 0.0

        for pid in pedestrian_ids:
            for vid in vehicle_ids:
                if pid in predictions and vid in predictions:
                    ped_pred, ped_vel = predictions[pid]
                    veh_pred, veh_vel = predictions[vid]
                    prob = compute_conflict_probability(
                        ped_pred, ped_vel, veh_pred, veh_vel
                    )
                    conflicts[(pid, vid)] = prob
                    max_conflict_prob = max(max_conflict_prob, prob)

        # ── Decision ──
        decision, decision_color = get_decision(max_conflict_prob)

        # Track stats (count transitions to avoid duplicate counting)
        if decision != prev_decision:
            if decision == "HOLD":
                stats["holds"] += 1
            elif decision == "CAUTION":
                stats["cautions"] += 1
            elif decision == "ALLOW":
                stats["allows"] += 1
            prev_decision = decision

        # ── Draw overlay ──
        frame = draw_overlay(
            frame, tracked, predictions, conflicts,
            decision, decision_color, max_conflict_prob, fps, stats
        )

        # ── Display ──
        if not args.headless:
            cv2.imshow("SafeTurn AI — Predictive Junction Safety", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[EXIT] User quit.")
                break
            elif key == ord('s') or key == ord('S'):
                filename = f"safeturn_screenshot_{frame_count:05d}.png"
                cv2.imwrite(filename, frame)
                print(f"[SCREENSHOT] Saved: {filename}")

        # ── FPS calculation ──
        elapsed = time.time() - loop_start
        fps = 1.0 / max(elapsed, 0.001)

        frame_count += 1

        # ── Frame limit check ──
        if args.frames > 0 and frame_count >= args.frames:
            print(f"\n[DONE] Processed {frame_count} frames.")
            break

    # ── Cleanup ──
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # ── Final stats ──
    print("\n" + "=" * 45)
    print("  SafeTurn AI — Session Summary")
    print("=" * 45)
    print(f"  Total Frames:        {frame_count}")
    print(f"  Conflicts Prevented: {stats['holds']}")
    print(f"  Cautions Issued:     {stats['cautions']}")
    print(f"  Turns Allowed:       {stats['allows']}")
    print(f"  Final FPS:           {fps:.1f}")
    print("=" * 45)


if __name__ == "__main__":
    main()
