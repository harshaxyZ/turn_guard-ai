"""
SafeTurn AI — Phase 2: Enhanced Tracking + Visualization
=========================================================
HackSETU 2025 | Theme 3: Intelligent Free-Left Turn Management

Upgrades over Phase 1:
  ✓ DeepSORT tracking (auto-fallback to CentroidTracker)
  ✓ Per-object trajectory trail lines
  ✓ Conflict zone circles at predicted intersection points
  ✓ Enhanced statistics overlay with decision history
  ✓ Heatmap-style danger zones

Usage:
    python safeturn_phase2.py --video traffic.mp4
    python safeturn_phase2.py --mock
    python safeturn_phase2.py --mock --frames 120

Controls:
    Q — Quit    S — Screenshot    T — Toggle trajectory trails
    H — Toggle heatmap    D — Toggle debug info
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

# ─── DeepSORT Import (graceful fallback to CentroidTracker) ─────
DEEPSORT_AVAILABLE = False
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
    print("[INFO] DeepSORT loaded — using advanced tracking.")
except ImportError:
    print("[INFO] DeepSORT not available — using built-in CentroidTracker.")
    print("       Install: pip install deep-sort-realtime")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.35
TARGET_CLASSES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

# Tracking
MAX_DISAPPEARED_FRAMES = 30
HISTORY_LENGTH = 10  # Increased from 5 for smoother trails
TRAIL_LENGTH = 30    # Max trail points to draw

# Prediction
LOOKAHEAD_SECONDS = 3.0

# Conflict engine
CONFLICT_DISTANCE_PX = 80
CONFLICT_DISTANCE_MAX = 250

# Decision thresholds
HOLD_THRESHOLD = 0.70
CAUTION_THRESHOLD = 0.40

# Colors (BGR)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_GREEN  = (0, 200, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (255, 255, 0)
COLOR_ORANGE = (0, 140, 255)
COLOR_PURPLE = (200, 50, 200)

CLASS_COLORS = {
    0: (255, 100, 100),   # person — blue
    1: (100, 255, 100),   # bicycle — green
    2: (100, 100, 255),   # car — red
    3: (0, 200, 255),     # motorcycle — yellow
    5: (255, 0, 200),     # bus — pink
    7: (200, 150, 0),     # truck — teal
}


# ═══════════════════════════════════════════════════════════════════
# CENTROID TRACKER (Fallback — same as Phase 1)
# ═══════════════════════════════════════════════════════════════════

class CentroidTracker:
    """Persistent object tracking via centroid distance matching."""

    def __init__(self, max_disappeared=MAX_DISAPPEARED_FRAMES):
        self.next_id = 0
        self.objects = {}
        self.bboxes = {}
        self.classes = {}
        self.disappeared = {}
        self.histories = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox, class_id):
        oid = self.next_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.classes[oid] = class_id
        self.disappeared[oid] = 0
        self.histories[oid] = collections.deque(maxlen=TRAIL_LENGTH)
        self.histories[oid].append(centroid)
        self.next_id += 1
        return oid

    def deregister(self, oid):
        for d in [self.objects, self.bboxes, self.classes,
                  self.disappeared, self.histories]:
            d.pop(oid, None)

    def update(self, detections):
        """
        detections: list of (centroid, bbox, class_id)
        Returns: dict {id: (centroid, bbox, class_id)}
        """
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self._build_result()

        if len(self.objects) == 0:
            for centroid, bbox, class_id in detections:
                self.register(centroid, bbox, class_id)
            return self._build_result()

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        det_centroids = [d[0] for d in detections]

        D = np.zeros((len(object_centroids), len(det_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, dc in enumerate(det_centroids):
                D[i, j] = math.dist(oc, dc)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
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

        for row in range(len(object_centroids)):
            if row not in used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for col in range(len(det_centroids)):
            if col not in used_cols:
                centroid, bbox, class_id = detections[col]
                self.register(centroid, bbox, class_id)

        return self._build_result()

    def _build_result(self):
        return {oid: (self.objects[oid], self.bboxes[oid], self.classes[oid])
                for oid in self.objects}


# ═══════════════════════════════════════════════════════════════════
# DEEPSORT TRACKER WRAPPER
# ═══════════════════════════════════════════════════════════════════
# Wraps deep_sort_realtime to match our CentroidTracker interface.
# If DeepSORT isn't installed, the system auto-falls back.
# ═══════════════════════════════════════════════════════════════════

class DeepSortTracker:
    """
    Wraps DeepSORT to provide the same interface as CentroidTracker.
    DeepSORT gives us better re-identification across occlusions.
    """

    def __init__(self):
        self.tracker = DeepSort(
            max_age=MAX_DISAPPEARED_FRAMES,
            n_init=3,                  # Confirm track after 3 detections
            max_iou_distance=0.7,
            embedder="mobilenet",      # Lightweight for CPU
            half=False,                # No FP16 on CPU
            embedder_gpu=False,        # Force CPU
        )
        self.histories = {}           # track_id → deque of (x, y)
        self.track_classes = {}       # track_id → class_id

    def update(self, detections, frame=None):
        """
        detections: list of (centroid, bbox, class_id)
        frame: the current video frame (needed for DeepSORT embeddings)
        Returns: dict {track_id: (centroid, bbox, class_id)}
        """
        if frame is None or len(detections) == 0:
            # Run DeepSORT with empty to age out tracks
            self.tracker.update_tracks([], frame=frame)
            return {}

        # Convert to DeepSORT format: [[x1, y1, w, h, conf], ...]
        ds_detections = []
        det_classes = []
        for centroid, bbox, class_id in detections:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(([x1, y1, w, h], 0.9, class_id))
            det_classes.append(class_id)

        # Update DeepSORT
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        result = {}
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Get class from detection or cached
            if hasattr(track, 'det_class') and track.det_class is not None:
                class_id = int(track.det_class)
            elif track_id in self.track_classes:
                class_id = self.track_classes[track_id]
            else:
                class_id = 2  # Default to car

            self.track_classes[track_id] = class_id

            # Maintain position history for trajectory prediction
            if track_id not in self.histories:
                self.histories[track_id] = collections.deque(maxlen=TRAIL_LENGTH)
            self.histories[track_id].append((cx, cy))

            result[track_id] = ((cx, cy), (x1, y1, x2, y2), class_id)

        # Cleanup histories for dead tracks
        active_ids = set(r for r in result)
        for tid in list(self.histories.keys()):
            if tid not in active_ids:
                # Keep for a bit in case track resumes
                pass

        return result


# ═══════════════════════════════════════════════════════════════════
# TRAJECTORY PREDICTION (Enhanced)
# ═══════════════════════════════════════════════════════════════════

def predict_position(history, fps):
    """
    Predict position at T+3s using linear extrapolation
    with weighted average (recent positions weighted more).
    """
    if len(history) < 2:
        return None, None

    positions = list(history)
    n = len(positions)

    # Weighted velocity: more recent displacements matter more
    total_vx, total_vy = 0.0, 0.0
    total_weight = 0.0
    for i in range(1, n):
        weight = i / n  # Later positions → higher weight
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        total_vx += dx * weight
        total_vy += dy * weight
        total_weight += weight

    if total_weight < 0.01:
        return None, None

    vx = total_vx / total_weight
    vy = total_vy / total_weight

    fps = max(fps, 1)
    frames_ahead = LOOKAHEAD_SECONDS * fps

    pred_x = positions[-1][0] + vx * frames_ahead
    pred_y = positions[-1][1] + vy * frames_ahead

    return (pred_x, pred_y), (vx, vy)


# ═══════════════════════════════════════════════════════════════════
# CONFLICT ENGINE (Same proven logic from Phase 1)
# ═══════════════════════════════════════════════════════════════════

def compute_conflict_probability(ped_pred, ped_vel, veh_pred, veh_vel):
    """Multi-factor conflict probability: distance + speed + convergence."""
    if ped_pred is None or veh_pred is None:
        return 0.0

    # Distance factor (60%)
    dist = math.dist(ped_pred, veh_pred)
    if dist < CONFLICT_DISTANCE_PX:
        dist_factor = 1.0
    elif dist > CONFLICT_DISTANCE_MAX:
        dist_factor = 0.0
    else:
        dist_factor = 1.0 - (dist - CONFLICT_DISTANCE_PX) / (CONFLICT_DISTANCE_MAX - CONFLICT_DISTANCE_PX)

    # Closing speed factor (25%)
    if ped_vel and veh_vel:
        rel_vx = veh_vel[0] - ped_vel[0]
        rel_vy = veh_vel[1] - ped_vel[1]
        closing_speed = math.sqrt(rel_vx**2 + rel_vy**2)
        speed_factor = min(closing_speed / 8.0, 1.0)
    else:
        speed_factor = 0.5

    # Convergence factor (15%)
    if ped_vel and veh_vel:
        ps = math.sqrt(ped_vel[0]**2 + ped_vel[1]**2)
        vs = math.sqrt(veh_vel[0]**2 + veh_vel[1]**2)
        if ps > 0.1 and vs > 0.1:
            dot = (ped_vel[0]/ps * veh_vel[0]/vs +
                   ped_vel[1]/ps * veh_vel[1]/vs)
            convergence_factor = (1.0 - dot) / 2.0
        else:
            convergence_factor = 0.5
    else:
        convergence_factor = 0.5

    probability = 0.60 * dist_factor + 0.25 * speed_factor + 0.15 * convergence_factor
    return round(min(max(probability, 0.0), 1.0), 3)


def get_decision(conflict_prob):
    """Convert conflict probability to HOLD/CAUTION/ALLOW decision."""
    if conflict_prob > HOLD_THRESHOLD:
        return "HOLD", COLOR_RED
    elif conflict_prob > CAUTION_THRESHOLD:
        return "CAUTION", COLOR_YELLOW
    else:
        return "ALLOW", COLOR_GREEN


# ═══════════════════════════════════════════════════════════════════
# MOCK DETECTION GENERATOR (Enhanced with more objects)
# ═══════════════════════════════════════════════════════════════════

class MockDetectionGenerator:
    """
    Enhanced mock: simulates a realistic junction with multiple
    pedestrians, cars, and a motorcycle — all with varying behaviors.
    """

    def __init__(self, width=960, height=540):
        self.w = width
        self.h = height
        self.frame_count = 0

        # Pedestrian 1: crosses left → right
        self.ped1 = {"x": 80.0, "y": self.h * 0.45, "vx": 2.2, "vy": 0.3}
        # Pedestrian 2: crosses right → left (delayed)
        self.ped2 = {"x": self.w * 0.7, "y": self.h * 0.48, "vx": -1.8, "vy": -0.2}
        # Car: free-left turn approach
        self.car1 = {"x": self.w * 0.78, "y": self.h * 0.88, "vx": -3.2, "vy": -2.8}
        # Car 2: waiting / slow approach
        self.car2 = {"x": self.w * 0.85, "y": self.h * 0.78, "vx": -1.0, "vy": -0.5}
        # Motorcycle: fast approach
        self.moto = {"x": self.w * 0.6, "y": self.h * 0.9, "vx": -2.5, "vy": -3.5}

    def _move(self, obj, noise_x=0.3, noise_y=0.2):
        obj["x"] += obj["vx"] + np.random.normal(0, noise_x)
        obj["y"] += obj["vy"] + np.random.normal(0, noise_y)

    def _reset_if_offscreen(self, obj, start_x, start_y, randomness=30):
        if (obj["x"] < -80 or obj["x"] > self.w + 80 or
                obj["y"] < -80 or obj["y"] > self.h + 80):
            obj["x"] = start_x + np.random.uniform(-randomness, randomness)
            obj["y"] = start_y + np.random.uniform(-randomness, randomness)

    def generate(self):
        self.frame_count += 1
        detections = []

        # Pedestrian 1 — always present
        self._move(self.ped1)
        self._reset_if_offscreen(self.ped1, 80, self.h * 0.45)
        px, py = int(self.ped1["x"]), int(self.ped1["y"])
        detections.append(((px, py), (px-15, py-35, px+15, py+35), 0))

        # Pedestrian 2 — appears after frame 40
        if self.frame_count > 40:
            self._move(self.ped2)
            self._reset_if_offscreen(self.ped2, self.w * 0.7, self.h * 0.48)
            px2, py2 = int(self.ped2["x"]), int(self.ped2["y"])
            detections.append(((px2, py2), (px2-15, py2-35, px2+15, py2+35), 0))

        # Car 1 — free-left turn vehicle
        self._move(self.car1, 0.5, 0.3)
        self._reset_if_offscreen(self.car1, self.w * 0.78, self.h * 0.88)
        cx, cy = int(self.car1["x"]), int(self.car1["y"])
        detections.append(((cx, cy), (cx-50, cy-30, cx+50, cy+30), 2))

        # Car 2 — slower approach
        if self.frame_count > 60:
            self._move(self.car2, 0.3, 0.2)
            self._reset_if_offscreen(self.car2, self.w * 0.85, self.h * 0.78)
            cx2, cy2 = int(self.car2["x"]), int(self.car2["y"])
            detections.append(((cx2, cy2), (cx2-50, cy2-30, cx2+50, cy2+30), 2))

        # Motorcycle — appears periodically
        if self.frame_count % 250 < 150:
            self._move(self.moto, 0.4, 0.4)
            self._reset_if_offscreen(self.moto, self.w * 0.6, self.h * 0.9)
            mx, my = int(self.moto["x"]), int(self.moto["y"])
            detections.append(((mx, my), (mx-20, my-18, mx+20, my+18), 3))

        return detections

    def generate_frame(self):
        """Generate visual junction background with road markings."""
        frame = np.full((self.h, self.w, 3), (55, 55, 55), dtype=np.uint8)

        # Main road (horizontal)
        cv2.rectangle(frame, (0, int(self.h*0.30)),
                      (self.w, int(self.h*0.70)), (45, 45, 45), -1)

        # Side road (vertical, right side — free-left approach)
        cv2.rectangle(frame, (int(self.w*0.55), int(self.h*0.55)),
                      (int(self.w*0.80), self.h), (45, 45, 45), -1)

        # Crosswalk stripes (white zebra crossing)
        for x in range(50, self.w - 100, 55):
            cv2.rectangle(frame, (x, int(self.h*0.36)),
                          (x + 28, int(self.h*0.50)), (190, 190, 190), -1)

        # Center lane dashes
        for x in range(0, self.w, 40):
            cv2.line(frame, (x, int(self.h*0.50)),
                     (x+20, int(self.h*0.50)), (80, 80, 80), 1)

        # Free-left turn arrow
        arrow_x, arrow_y = int(self.w*0.65), int(self.h*0.72)
        cv2.arrowedLine(frame, (arrow_x, arrow_y + 30),
                        (arrow_x - 40, arrow_y - 10), (80, 100, 80), 2, tipLength=0.3)
        cv2.putText(frame, "FREE", (arrow_x - 35, arrow_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 100, 80), 1)
        cv2.putText(frame, "LEFT", (arrow_x - 35, arrow_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 100, 80), 1)

        # Junction label
        cv2.putText(frame, "JUNCTION CAM 01", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (self.w - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return frame


# ═══════════════════════════════════════════════════════════════════
# OVERLAY RENDERER (Phase 2 — Enhanced Visualization)
# ═══════════════════════════════════════════════════════════════════

def draw_trajectory_trails(frame, histories, tracked_objects, class_colors):
    """
    Draw trajectory trail lines for each tracked object.
    Color matches the object class. Fades from transparent to solid.
    """
    for oid, history in histories.items():
        if len(history) < 2:
            continue

        # Get color for this object
        class_id = tracked_objects.get(oid, (None, None, 2))[2]
        color = class_colors.get(class_id, COLOR_WHITE)

        points = list(history)
        for i in range(1, len(points)):
            # Fade effect: older points = more transparent
            alpha = i / len(points)
            thickness = max(1, int(alpha * 3))
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            # Blend color with black for fade effect
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, faded_color, thickness)


def draw_conflict_zones(frame, conflicts, predictions):
    """
    Draw animated conflict zone circles where predicted paths intersect.
    Pulsing effect on high-probability zones.
    """
    for (pid, vid), prob in conflicts.items():
        if prob < CAUTION_THRESHOLD:
            continue

        p1 = predictions.get(pid, (None, None))[0]
        p2 = predictions.get(vid, (None, None))[0]
        if p1 is None or p2 is None:
            continue

        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)

        # Pulsing radius based on time
        pulse = math.sin(time.time() * 6) * 5  # ±5px pulse
        base_radius = int(25 + prob * 50)
        radius = max(10, base_radius + int(pulse))

        if prob > HOLD_THRESHOLD:
            zone_color = COLOR_RED
            # Draw danger zone with cross-hatching effect
            cv2.circle(frame, (mid_x, mid_y), radius, zone_color, 2)
            cv2.circle(frame, (mid_x, mid_y), radius + 8, zone_color, 1)
            # Inner fill (semi-transparent)
            overlay = frame.copy()
            cv2.circle(overlay, (mid_x, mid_y), radius, zone_color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        else:
            zone_color = COLOR_YELLOW
            cv2.circle(frame, (mid_x, mid_y), radius, zone_color, 2)

        # Probability label
        label = f"{prob:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, label, (mid_x - tw//2, mid_y - radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

        # "DANGER ZONE" text for HOLD zones
        if prob > HOLD_THRESHOLD:
            cv2.putText(frame, "DANGER ZONE",
                        (mid_x - 50, mid_y + radius + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)


def draw_overlay(frame, tracked_objects, predictions, conflicts,
                 decision, decision_color, max_conflict_prob, fps,
                 stats, histories, show_trails=True, show_debug=False,
                 tracker_type="CentroidTracker"):
    """
    Full SafeTurn Phase 2 overlay:
    - Bounding boxes with class labels and IDs
    - Trajectory trails (NEW)
    - Predicted position markers with trajectory arrows
    - Conflict zone circles with pulsing animation (NEW)
    - Decision banner with gradient
    - Probability bar
    - Enhanced stats panel with decision history (NEW)
    - Tracker type indicator (NEW)
    """
    h, w = frame.shape[:2]

    # ── Trajectory trails ──
    if show_trails:
        draw_trajectory_trails(frame, histories, tracked_objects, CLASS_COLORS)

    # ── Conflict zones ──
    draw_conflict_zones(frame, conflicts, predictions)

    # ── Bounding boxes + labels ──
    for oid, (centroid, bbox, class_id) in tracked_objects.items():
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(class_id, COLOR_WHITE)
        label = TARGET_CLASSES.get(class_id, "?")

        # Rounded-corner feel: thicker box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with background
        label_text = f"#{oid} {label}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1)

        # Centroid dot
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, color, -1)

    # ── Predicted positions + trajectory arrows ──
    for oid, (pred_pos, vel) in predictions.items():
        if pred_pos is None:
            continue
        px, py = int(pred_pos[0]), int(pred_pos[1])
        class_id = tracked_objects[oid][2] if oid in tracked_objects else 0
        color = CLASS_COLORS.get(class_id, COLOR_WHITE)

        # Predicted position: crosshair marker
        cv2.drawMarker(frame, (px, py), color, cv2.MARKER_CROSS, 16, 1)
        cv2.circle(frame, (px, py), 10, color, 1)

        # Dashed trajectory arrow from current to predicted
        if oid in tracked_objects:
            cx, cy = tracked_objects[oid][0]
            # Draw dashed line
            _draw_dashed_line(frame, (int(cx), int(cy)), (px, py), color, 1, 8)
            # Arrowhead at predicted position
            _draw_arrowhead(frame, (int(cx), int(cy)), (px, py), color)

    # ── Decision banner (top center) — gradient style ──
    banner_h = 65
    overlay = frame.copy()
    # Gradient-like banner
    for y_off in range(banner_h):
        alpha = 1.0 - (y_off / banner_h) * 0.3
        color_row = tuple(int(c * alpha) for c in decision_color)
        cv2.line(overlay, (0, y_off), (w, y_off), color_row, 1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Decision text with shadow
    text = f"SafeTurn: {decision}"
    cv2.putText(frame, text, (w//2 - 132, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_BLACK, 4)
    cv2.putText(frame, text, (w//2 - 130, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_WHITE, 3)

    # ── Probability bar (below banner) ──
    bar_y = banner_h + 8
    bar_h = 20
    bar_w = int(w * 0.40)
    bar_x = (w - bar_w) // 2
    fill_w = int(bar_w * max_conflict_prob)

    # Bar with rounded ends
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    if fill_w > 0:
        bar_color = COLOR_RED if max_conflict_prob > 0.7 else (
            COLOR_YELLOW if max_conflict_prob > 0.4 else COLOR_GREEN)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  COLOR_WHITE, 1)

    # Threshold markers on the bar
    t40_x = bar_x + int(bar_w * 0.4)
    t70_x = bar_x + int(bar_w * 0.7)
    cv2.line(frame, (t40_x, bar_y), (t40_x, bar_y + bar_h), COLOR_YELLOW, 1)
    cv2.line(frame, (t70_x, bar_y), (t70_x, bar_y + bar_h), COLOR_RED, 1)

    cv2.putText(frame, f"Conflict: {max_conflict_prob:.1%}",
                (bar_x + bar_w + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # ── Enhanced stats panel (bottom-left) ──
    panel_w = 280
    panel_h = 120
    panel_x = 5
    panel_y = h - panel_h - 5

    # Panel background with slight transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

    # Stats title
    cv2.putText(frame, "SESSION STATS", (panel_x + 10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1)

    cy_offset = panel_y + 42
    cv2.putText(frame, f"Conflicts Prevented: {stats['holds']}",
                (panel_x + 15, cy_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_RED, 1)
    cv2.putText(frame, f"Cautions Issued:     {stats['cautions']}",
                (panel_x + 15, cy_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1)
    cv2.putText(frame, f"Turns Allowed:       {stats['allows']}",
                (panel_x + 15, cy_offset + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)
    cv2.putText(frame, f"Objects Tracked:     {len(tracked_objects)}",
                (panel_x + 15, cy_offset + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # ── FPS + tracker info (bottom-right) ──
    info_y = h - 15
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    cv2.putText(frame, f"Tracker: {tracker_type}", (w - 240, info_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ── Branding (top-right) ──
    cv2.putText(frame, "SafeTurn AI v2", (w - 170, banner_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 2)
    cv2.putText(frame, "3s Predictive | HackSETU 2025", (w - 280, banner_h + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # ── Debug info (toggle with D key) ──
    if show_debug:
        debug_y = banner_h + 80
        cv2.putText(frame, f"Tracked: {len(tracked_objects)} | "
                    f"Peds: {sum(1 for _,(_,_,c) in tracked_objects.items() if c==0)} | "
                    f"Vehicles: {sum(1 for _,(_,_,c) in tracked_objects.items() if c in VEHICLE_CLASSES)}",
                    (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(frame, f"Active conflicts: {sum(1 for p in conflicts.values() if p > CAUTION_THRESHOLD)}",
                    (10, debug_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return frame


def _draw_dashed_line(frame, pt1, pt2, color, thickness=1, dash_len=8):
    """Draw a dashed line between two points."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1:
        return
    dashes = int(dist / (dash_len * 2))
    for i in range(dashes):
        t1 = (i * 2 * dash_len) / dist
        t2 = ((i * 2 + 1) * dash_len) / dist
        if t2 > 1:
            t2 = 1
        start = (int(pt1[0] + dx * t1), int(pt1[1] + dy * t1))
        end = (int(pt1[0] + dx * t2), int(pt1[1] + dy * t2))
        cv2.line(frame, start, end, color, thickness)


def _draw_arrowhead(frame, pt1, pt2, color, size=10):
    """Draw an arrowhead at pt2 pointing away from pt1."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1:
        return
    dx /= dist
    dy /= dist

    # Perpendicular
    px, py = -dy, dx

    tip = pt2
    left = (int(tip[0] - size*(dx - 0.5*px)),
            int(tip[1] - size*(dy - 0.5*py)))
    right = (int(tip[0] - size*(dx + 0.5*px)),
             int(tip[1] - size*(dy + 0.5*py)))

    cv2.fillConvexPoly(frame, np.array([tip, left, right]), color)


# ═══════════════════════════════════════════════════════════════════
# YOLO DETECTION WRAPPER
# ═══════════════════════════════════════════════════════════════════

def run_yolo_detection(model, frame):
    """Run YOLOv8 on a frame, filter for target classes."""
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
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append(
                ((cx, cy), (int(x1), int(y1), int(x2), int(y2)), cls)
            )
    return detections


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SafeTurn AI v2 — Enhanced Tracking + Visualization"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file")
    parser.add_argument("--mock", action="store_true",
                        help="Run with synthetic mock detections")
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after N frames (0 = forever)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model path")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display window")
    parser.add_argument("--no-deepsort", action="store_true",
                        help="Force CentroidTracker even if DeepSORT available")
    args = parser.parse_args()

    # ── Mode selection ──
    use_mock = args.mock or (args.video is None and not YOLO_AVAILABLE)

    if use_mock:
        print("=" * 55)
        print("  SafeTurn AI v2 — MOCK MODE (Enhanced)")
        print("  Synthetic junction scenario with multiple objects")
        print("=" * 55)
        mock_gen = MockDetectionGenerator()
        cap = None
    else:
        if args.video is None:
            print("[ERROR] No video. Use --video <path> or --mock")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {args.video} — falling back to MOCK")
            use_mock = True
            mock_gen = MockDetectionGenerator()
            cap = None
        else:
            print(f"  SafeTurn AI v2 — VIDEO: {args.video}")

    # ── Load YOLO ──
    model = None
    if not use_mock and YOLO_AVAILABLE:
        try:
            model = YOLO(args.model)
            print(f"[INFO] YOLOv8 loaded: {args.model}")
        except Exception as e:
            print(f"[WARN] YOLO failed: {e} — using MOCK")
            use_mock = True
            mock_gen = MockDetectionGenerator()

    # ── Select tracker ──
    use_deepsort = DEEPSORT_AVAILABLE and not args.no_deepsort
    if use_deepsort:
        tracker = DeepSortTracker()
        tracker_type = "DeepSORT"
    else:
        tracker = CentroidTracker()
        tracker_type = "CentroidTracker"
    print(f"[INFO] Tracker: {tracker_type}")

    # ── State ──
    stats = {"holds": 0, "cautions": 0, "allows": 0}
    decision_history = collections.deque(maxlen=100)
    fps = 10.0
    frame_count = 0
    prev_decision = None
    show_trails = True
    show_debug = False

    print("\n[READY] Q=quit  S=screenshot  T=toggle trails  D=debug\n")

    while True:
        loop_start = time.time()

        # ── Get frame ──
        if use_mock:
            frame = mock_gen.generate_frame()
        else:
            ret, frame = cap.read()
            if not ret:
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
        if use_deepsort:
            tracked = tracker.update(detections, frame=frame)
            histories = tracker.histories
        else:
            tracked = tracker.update(detections)
            histories = tracker.histories

        # ── Trajectory Prediction ──
        predictions = {}
        for oid in tracked:
            hist = histories.get(oid)
            if hist and len(hist) >= 2:
                pred_pos, vel = predict_position(hist, fps)
                predictions[oid] = (pred_pos, vel)

        # ── Conflict Engine ──
        pedestrian_ids = [oid for oid, (_, _, cls) in tracked.items() if cls == 0]
        vehicle_ids = [oid for oid, (_, _, cls) in tracked.items() if cls in VEHICLE_CLASSES]

        conflicts = {}
        max_conflict_prob = 0.0
        for pid in pedestrian_ids:
            for vid in vehicle_ids:
                if pid in predictions and vid in predictions:
                    prob = compute_conflict_probability(
                        predictions[pid][0], predictions[pid][1],
                        predictions[vid][0], predictions[vid][1]
                    )
                    conflicts[(pid, vid)] = prob
                    max_conflict_prob = max(max_conflict_prob, prob)

        # ── Decision ──
        decision, decision_color = get_decision(max_conflict_prob)

        if decision != prev_decision:
            if decision == "HOLD":
                stats["holds"] += 1
            elif decision == "CAUTION":
                stats["cautions"] += 1
            else:
                stats["allows"] += 1
            prev_decision = decision

        decision_history.append((frame_count, decision, max_conflict_prob))

        # ── Draw overlay ──
        frame = draw_overlay(
            frame, tracked, predictions, conflicts,
            decision, decision_color, max_conflict_prob, fps,
            stats, histories, show_trails, show_debug, tracker_type
        )

        # ── Display ──
        if not args.headless:
            cv2.imshow("SafeTurn AI v2 — Predictive Junction Safety", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                fn = f"safeturn_v2_{frame_count:05d}.png"
                cv2.imwrite(fn, frame)
                print(f"[SCREENSHOT] {fn}")
            elif key == ord('t') or key == ord('T'):
                show_trails = not show_trails
                print(f"[TOGGLE] Trails: {'ON' if show_trails else 'OFF'}")
            elif key == ord('d') or key == ord('D'):
                show_debug = not show_debug
                print(f"[TOGGLE] Debug: {'ON' if show_debug else 'OFF'}")

        # ── FPS ──
        elapsed = time.time() - loop_start
        fps = 1.0 / max(elapsed, 0.001)
        frame_count += 1

        if args.frames > 0 and frame_count >= args.frames:
            break

    # ── Cleanup ──
    if cap:
        cap.release()
    cv2.destroyAllWindows()

    # ── Final report ──
    print("\n" + "=" * 50)
    print("  SafeTurn AI v2 — Session Summary")
    print("=" * 50)
    print(f"  Tracker:             {tracker_type}")
    print(f"  Total Frames:        {frame_count}")
    print(f"  Conflicts Prevented: {stats['holds']}")
    print(f"  Cautions Issued:     {stats['cautions']}")
    print(f"  Turns Allowed:       {stats['allows']}")
    print(f"  Final FPS:           {fps:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
