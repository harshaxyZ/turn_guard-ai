"""
SafeTurn AI — FINAL: Complete Predictive + Post-Accident System
================================================================
HackSETU 2025 | Theme 3: Intelligent Free-Left Turn Management

"Predictive, not reactive — we act before danger arrives."

EVERYTHING IN ONE FILE — beginner-friendly, demo-ready.

FEATURES:
  ═══ CORE (Predictive) ═══
  ✓ YOLOv8 detection (pedestrians, vehicles) with mock fallback
  ✓ Centroid tracking with persistent IDs
  ✓ 3-second trajectory prediction
  ✓ Conflict probability engine
  ✓ HOLD / CAUTION / ALLOW decision

  ═══ EXTENDED (Post-Accident) ═══
  ✓ Accident detection (overlap + speed drop)
  ✓ Traffic signal simulation (ALL SIGNALS RED)
  ✓ Severity estimation (HIGH / MEDIUM / LOW)
  ✓ Ambulance alert display
  ✓ Crowd density detection

Usage:
    python safeturn_final.py --video traffic.mp4
    python safeturn_final.py --mock
    python safeturn_final.py --mock --frames 120

Controls:
    Q — Quit    S — Screenshot    A — Simulate accident (test)
"""

import argparse
import math
import time
import sys
import collections
import numpy as np

# ─── OpenCV ─────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found. Install: pip install opencv-python")
    sys.exit(1)

# ─── YOLOv8 (graceful fallback) ─────────────────────────────────
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not found — using MOCK detection mode.")
    print("       Install: pip install ultralytics")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — All tunable parameters in one place
# ═══════════════════════════════════════════════════════════════════

# Detection
CONFIDENCE_THRESHOLD = 0.35
TARGET_CLASSES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

# Tracking
MAX_DISAPPEARED = 30       # Frames before object is removed
TRAIL_LENGTH = 20          # Trail history length

# Prediction
LOOKAHEAD_SECONDS = 3.0    # Predict this far ahead
HISTORY_FOR_VELOCITY = 5   # Frames used for velocity calc

# Conflict thresholds
CONFLICT_CLOSE_PX = 80     # Below this = high conflict
CONFLICT_FAR_PX = 250      # Above this = no conflict
HOLD_THRESHOLD = 0.70      # > 70% → 🔴 HOLD
CAUTION_THRESHOLD = 0.40   # > 40% → 🟡 CAUTION

# ═══ EXTENDED FEATURE CONFIG ═══
# Accident detection
ACCIDENT_OVERLAP_PX = 40   # Bbox overlap threshold for collision
ACCIDENT_SPEED_DROP = 0.5  # Speed must drop by 50% to confirm accident
ACCIDENT_COOLDOWN = 90     # Frames to keep showing accident alert

# Crowd density
CROWD_HIGH_THRESHOLD = 5   # >= this many people = HIGH density
CROWD_MED_THRESHOLD = 3    # >= this many people = MEDIUM density

# Colors (BGR)
RED    = (0, 0, 255)
YELLOW = (0, 220, 255)
GREEN  = (0, 200, 0)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
CYAN   = (255, 255, 0)
ORANGE = (0, 140, 255)
PURPLE = (200, 50, 200)
DARK_RED = (0, 0, 180)

CLASS_COLORS = {
    0: (255, 100, 100),   # person
    1: (100, 255, 100),   # bicycle
    2: (100, 100, 255),   # car
    3: (0, 200, 255),     # motorcycle
    5: (255, 0, 200),     # bus
    7: (200, 150, 0),     # truck
}


# ═══════════════════════════════════════════════════════════════════
# CENTROID TRACKER — Assigns persistent IDs to detected objects
# ═══════════════════════════════════════════════════════════════════

class CentroidTracker:
    """
    Simple but effective tracker:
    1. New detection? Assign new ID
    2. Existing detection moved? Match by closest centroid
    3. Object disappeared? Remove after N frames
    """

    def __init__(self):
        self.next_id = 0
        self.objects = {}       # id → (cx, cy)
        self.bboxes = {}        # id → (x1, y1, x2, y2)
        self.classes = {}       # id → class_id
        self.disappeared = {}   # id → frame count
        self.histories = {}     # id → deque of (cx, cy)
        self.speeds = {}        # id → current speed (pixels/frame)

    def register(self, centroid, bbox, class_id):
        oid = self.next_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.classes[oid] = class_id
        self.disappeared[oid] = 0
        self.histories[oid] = collections.deque(maxlen=TRAIL_LENGTH)
        self.histories[oid].append(centroid)
        self.speeds[oid] = 0.0
        self.next_id += 1
        return oid

    def deregister(self, oid):
        for d in [self.objects, self.bboxes, self.classes,
                  self.disappeared, self.histories, self.speeds]:
            d.pop(oid, None)

    def _compute_speed(self, oid):
        """Compute speed as average displacement over recent frames."""
        h = list(self.histories[oid])
        if len(h) < 2:
            return 0.0
        dx = h[-1][0] - h[-2][0]
        dy = h[-1][1] - h[-2][1]
        return math.sqrt(dx*dx + dy*dy)

    def update(self, detections):
        """
        Args: detections = [(centroid, bbox, class_id), ...]
        Returns: {id: (centroid, bbox, class_id)}
        """
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > MAX_DISAPPEARED:
                    self.deregister(oid)
            return self._result()

        if len(self.objects) == 0:
            for c, b, cl in detections:
                self.register(c, b, cl)
            return self._result()

        # Match by closest centroid
        oids = list(self.objects.keys())
        old_c = list(self.objects.values())
        new_c = [d[0] for d in detections]

        D = np.zeros((len(old_c), len(new_c)))
        for i, oc in enumerate(old_c):
            for j, nc in enumerate(new_c):
                D[i, j] = math.dist(oc, nc)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_r, used_c = set(), set()

        for r, c in zip(rows, cols):
            if r in used_r or c in used_c or D[r, c] > 150:
                continue
            oid = oids[r]
            cent, bbox, cls = detections[c]
            self.objects[oid] = cent
            self.bboxes[oid] = bbox
            self.classes[oid] = cls
            self.disappeared[oid] = 0
            self.histories[oid].append(cent)
            self.speeds[oid] = self._compute_speed(oid)
            used_r.add(r)
            used_c.add(c)

        for r in range(len(old_c)):
            if r not in used_r:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > MAX_DISAPPEARED:
                    self.deregister(oid)

        for c in range(len(new_c)):
            if c not in used_c:
                self.register(*detections[c])

        return self._result()

    def _result(self):
        return {oid: (self.objects[oid], self.bboxes[oid], self.classes[oid])
                for oid in self.objects}


# ═══════════════════════════════════════════════════════════════════
# TRAJECTORY PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_position(history, fps):
    """Predict where object will be in 3 seconds using linear motion."""
    if len(history) < 2:
        return None, None

    pts = list(history)
    n = len(pts)

    # Weighted velocity — recent motion matters more
    tvx, tvy, tw = 0.0, 0.0, 0.0
    for i in range(1, n):
        w = i / n
        tvx += (pts[i][0] - pts[i-1][0]) * w
        tvy += (pts[i][1] - pts[i-1][1]) * w
        tw += w

    if tw < 0.01:
        return None, None

    vx, vy = tvx / tw, tvy / tw
    frames_ahead = LOOKAHEAD_SECONDS * max(fps, 1)

    px = pts[-1][0] + vx * frames_ahead
    py = pts[-1][1] + vy * frames_ahead
    return (px, py), (vx, vy)


# ═══════════════════════════════════════════════════════════════════
# CONFLICT ENGINE — Collision probability for ped-vehicle pairs
# ═══════════════════════════════════════════════════════════════════

def compute_conflict_probability(ped_pred, ped_vel, veh_pred, veh_vel):
    """
    3-factor conflict probability:
      60% distance at T+3s
      25% closing speed
      15% trajectory convergence
    """
    if ped_pred is None or veh_pred is None:
        return 0.0

    # Distance factor
    dist = math.dist(ped_pred, veh_pred)
    if dist < CONFLICT_CLOSE_PX:
        df = 1.0
    elif dist > CONFLICT_FAR_PX:
        df = 0.0
    else:
        df = 1.0 - (dist - CONFLICT_CLOSE_PX) / (CONFLICT_FAR_PX - CONFLICT_CLOSE_PX)

    # Speed factor
    if ped_vel and veh_vel:
        rv = math.sqrt((veh_vel[0]-ped_vel[0])**2 + (veh_vel[1]-ped_vel[1])**2)
        sf = min(rv / 8.0, 1.0)
    else:
        sf = 0.5

    # Convergence factor
    if ped_vel and veh_vel:
        ps = math.sqrt(ped_vel[0]**2 + ped_vel[1]**2)
        vs = math.sqrt(veh_vel[0]**2 + veh_vel[1]**2)
        if ps > 0.1 and vs > 0.1:
            dot = (ped_vel[0]/ps * veh_vel[0]/vs + ped_vel[1]/ps * veh_vel[1]/vs)
            cf = (1.0 - dot) / 2.0
        else:
            cf = 0.5
    else:
        cf = 0.5

    prob = 0.60*df + 0.25*sf + 0.15*cf
    return round(min(max(prob, 0.0), 1.0), 3)


def get_decision(prob):
    """Convert probability to HOLD/CAUTION/ALLOW."""
    if prob > HOLD_THRESHOLD:
        return "HOLD", RED
    elif prob > CAUTION_THRESHOLD:
        return "CAUTION", YELLOW
    else:
        return "ALLOW", GREEN


# ═══════════════════════════════════════════════════════════════════
# EXTENDED: ACCIDENT DETECTION
# ═══════════════════════════════════════════════════════════════════
# Logic: If any vehicle-pedestrian OR vehicle-vehicle pair has:
#   1. Bounding boxes overlapping (IoU > 0 or distance < threshold)
#   2. AND at least one object's speed dropped significantly
# → Accident detected
# ═══════════════════════════════════════════════════════════════════

class AccidentDetector:
    """
    Detects accidents by looking for:
    1. Sudden object overlap (bboxes collide)
    2. Speed drops after overlap (objects stop/slow after impact)
    """

    def __init__(self):
        self.accident_active = False
        self.accident_frame = 0       # Frame when accident was detected
        self.accident_severity = "NONE"
        self.accident_location = (0, 0)
        self.accident_pair = (None, None)
        self.prev_speeds = {}         # id → previous speed for drop detection
        self.cooldown_counter = 0

    def _bbox_overlap(self, b1, b2):
        """Check if two bounding boxes overlap. Returns overlap area."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x1 < x2 and y1 < y2:
            return (x2 - x1) * (y2 - y1)
        return 0

    def _bbox_distance(self, b1, b2):
        """Distance between bbox centers."""
        c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
        c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
        return math.dist(c1, c2)

    def _estimate_severity(self, speed1, speed2, overlap_area):
        """
        Simple severity estimation:
        - HIGH: fast objects + large overlap
        - MEDIUM: moderate
        - LOW: slow objects or small overlap
        """
        combined_speed = speed1 + speed2
        if combined_speed > 10 and overlap_area > 500:
            return "HIGH"
        elif combined_speed > 5 or overlap_area > 200:
            return "MEDIUM"
        else:
            return "LOW"

    def update(self, tracked, tracker, frame_count):
        """
        Check for accidents among all tracked objects.

        Args:
            tracked: {id: (centroid, bbox, class_id)}
            tracker: CentroidTracker instance (for speed data)
            frame_count: current frame number

        Returns:
            dict with accident status info
        """
        # If accident is active, count down cooldown
        if self.accident_active:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.accident_active = False
                self.accident_severity = "NONE"

        # Get all object pairs (vehicle-pedestrian AND vehicle-vehicle)
        oids = list(tracked.keys())

        for i in range(len(oids)):
            for j in range(i + 1, len(oids)):
                id1, id2 = oids[i], oids[j]
                _, bbox1, cls1 = tracked[id1]
                _, bbox2, cls2 = tracked[id2]

                # At least one must be a vehicle
                if cls1 not in VEHICLE_CLASSES and cls2 not in VEHICLE_CLASSES:
                    continue

                # Check overlap
                overlap = self._bbox_overlap(bbox1, bbox2)
                dist = self._bbox_distance(bbox1, bbox2)

                if overlap > 0 or dist < ACCIDENT_OVERLAP_PX:
                    # Check speed drop (objects stopped/slowed after collision)
                    speed1 = tracker.speeds.get(id1, 0)
                    speed2 = tracker.speeds.get(id2, 0)
                    prev1 = self.prev_speeds.get(id1, speed1)
                    prev2 = self.prev_speeds.get(id2, speed2)

                    # Speed drop detection: was moving fast, now slow
                    speed_dropped = False
                    if prev1 > 3.0 and speed1 < prev1 * ACCIDENT_SPEED_DROP:
                        speed_dropped = True
                    if prev2 > 3.0 and speed2 < prev2 * ACCIDENT_SPEED_DROP:
                        speed_dropped = True

                    # Or very close + at least one is moving
                    if overlap > 100 or (dist < 30 and (speed1 > 2 or speed2 > 2)):
                        speed_dropped = True

                    if speed_dropped and not self.accident_active:
                        # ═══ ACCIDENT DETECTED ═══
                        self.accident_active = True
                        self.accident_frame = frame_count
                        self.cooldown_counter = ACCIDENT_COOLDOWN
                        self.accident_severity = self._estimate_severity(
                            prev1, prev2, max(overlap, 1))
                        mid_x = int((bbox1[0]+bbox1[2]+bbox2[0]+bbox2[2]) / 4)
                        mid_y = int((bbox1[1]+bbox1[3]+bbox2[1]+bbox2[3]) / 4)
                        self.accident_location = (mid_x, mid_y)
                        self.accident_pair = (id1, id2)

        # Store current speeds for next frame comparison
        for oid in tracked:
            self.prev_speeds[oid] = tracker.speeds.get(oid, 0)

        return {
            "active": self.accident_active,
            "severity": self.accident_severity,
            "location": self.accident_location,
            "pair": self.accident_pair,
            "frame": self.accident_frame,
        }

    def force_trigger(self, tracked, frame_count):
        """Force-trigger an accident for demo purposes (press A key)."""
        if len(tracked) >= 2:
            oids = list(tracked.keys())
            id1, id2 = oids[0], oids[1]
            c1 = tracked[id1][0]
            c2 = tracked[id2][0]
            self.accident_active = True
            self.accident_frame = frame_count
            self.cooldown_counter = ACCIDENT_COOLDOWN
            self.accident_severity = "HIGH"
            self.accident_location = (
                int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)
            )
            self.accident_pair = (id1, id2)


# ═══════════════════════════════════════════════════════════════════
# EXTENDED: CROWD DENSITY DETECTOR
# ═══════════════════════════════════════════════════════════════════

def get_crowd_density(tracked):
    """
    Count pedestrians (class 0) in the scene.
    Returns: (count, level_string, color)
    """
    ped_count = sum(1 for _, (_, _, cls) in tracked.items() if cls == 0)

    if ped_count >= CROWD_HIGH_THRESHOLD:
        return ped_count, "HIGH", RED
    elif ped_count >= CROWD_MED_THRESHOLD:
        return ped_count, "MEDIUM", YELLOW
    else:
        return ped_count, "LOW", GREEN


# ═══════════════════════════════════════════════════════════════════
# MOCK DETECTION GENERATOR
# ═══════════════════════════════════════════════════════════════════

class MockDetectionGenerator:
    """
    Simulates a junction with pedestrians, cars, and a motorcycle.
    Periodically creates a near-collision scenario for demo.
    """

    def __init__(self, w=960, h=540):
        self.w, self.h = w, h
        self.frame = 0

        # Pedestrian 1: crosses left → right
        self.ped1 = {"x": 80.0, "y": h*0.45, "vx": 2.2, "vy": 0.3}
        # Pedestrian 2: crosses right → left
        self.ped2 = {"x": w*0.7, "y": h*0.48, "vx": -1.8, "vy": -0.2}
        # Pedestrian 3: standing near crossing
        self.ped3 = {"x": w*0.3, "y": h*0.40, "vx": 0.3, "vy": 0.1}
        # Car: free-left turn
        self.car1 = {"x": w*0.78, "y": h*0.88, "vx": -3.2, "vy": -2.8}
        # Car 2: slower
        self.car2 = {"x": w*0.85, "y": h*0.78, "vx": -1.0, "vy": -0.5}
        # Motorcycle
        self.moto = {"x": w*0.6, "y": h*0.9, "vx": -2.5, "vy": -3.5}

    def _move(self, obj, nx=0.3, ny=0.2):
        obj["x"] += obj["vx"] + np.random.normal(0, nx)
        obj["y"] += obj["vy"] + np.random.normal(0, ny)

    def _reset(self, obj, sx, sy, r=30):
        if obj["x"]<-80 or obj["x"]>self.w+80 or obj["y"]<-80 or obj["y"]>self.h+80:
            obj["x"] = sx + np.random.uniform(-r, r)
            obj["y"] = sy + np.random.uniform(-r, r)

    def generate(self):
        self.frame += 1
        dets = []

        # Pedestrian 1 — always
        self._move(self.ped1)
        self._reset(self.ped1, 80, self.h*0.45)
        px, py = int(self.ped1["x"]), int(self.ped1["y"])
        dets.append(((px, py), (px-15, py-35, px+15, py+35), 0))

        # Pedestrian 2 — after frame 30
        if self.frame > 30:
            self._move(self.ped2)
            self._reset(self.ped2, self.w*0.7, self.h*0.48)
            px2, py2 = int(self.ped2["x"]), int(self.ped2["y"])
            dets.append(((px2, py2), (px2-15, py2-35, px2+15, py2+35), 0))

        # Pedestrian 3 — slow walker near crossing
        if self.frame > 20:
            self._move(self.ped3, 0.1, 0.1)
            self._reset(self.ped3, self.w*0.3, self.h*0.40)
            px3, py3 = int(self.ped3["x"]), int(self.ped3["y"])
            dets.append(((px3, py3), (px3-15, py3-35, px3+15, py3+35), 0))

        # Car 1 — free-left turn
        self._move(self.car1, 0.5, 0.3)
        self._reset(self.car1, self.w*0.78, self.h*0.88)
        cx, cy = int(self.car1["x"]), int(self.car1["y"])
        dets.append(((cx, cy), (cx-50, cy-30, cx+50, cy+30), 2))

        # Car 2 — slower
        if self.frame > 50:
            self._move(self.car2, 0.3, 0.2)
            self._reset(self.car2, self.w*0.85, self.h*0.78)
            cx2, cy2 = int(self.car2["x"]), int(self.car2["y"])
            dets.append(((cx2, cy2), (cx2-50, cy2-30, cx2+50, cy2+30), 2))

        # Motorcycle — periodic
        if self.frame % 200 < 130:
            self._move(self.moto, 0.4, 0.4)
            self._reset(self.moto, self.w*0.6, self.h*0.9)
            mx, my = int(self.moto["x"]), int(self.moto["y"])
            dets.append(((mx, my), (mx-20, my-18, mx+20, my+18), 3))

        return dets

    def generate_frame(self):
        """Draw junction background."""
        f = np.full((self.h, self.w, 3), (55,55,55), dtype=np.uint8)
        # Road
        cv2.rectangle(f, (0,int(self.h*0.30)), (self.w,int(self.h*0.70)), (45,45,45), -1)
        # Side road
        cv2.rectangle(f, (int(self.w*0.55),int(self.h*0.55)), (int(self.w*0.80),self.h), (45,45,45), -1)
        # Crosswalk stripes
        for x in range(50, self.w-100, 55):
            cv2.rectangle(f, (x,int(self.h*0.36)), (x+28,int(self.h*0.50)), (190,190,190), -1)
        # Free left arrow
        ax, ay = int(self.w*0.65), int(self.h*0.72)
        cv2.arrowedLine(f, (ax,ay+30), (ax-40,ay-10), (80,100,80), 2, tipLength=0.3)
        cv2.putText(f, "FREE LEFT", (ax-45,ay+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,100,80), 1)
        # Camera label
        cv2.putText(f, "JUNCTION CAM 01", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        cv2.putText(f, time.strftime("%H:%M:%S"), (self.w-100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        return f


# ═══════════════════════════════════════════════════════════════════
# YOLO DETECTION WRAPPER
# ═══════════════════════════════════════════════════════════════════

def run_yolo(model, frame):
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    dets = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in TARGET_CLASSES: continue
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            dets.append(((( x1+x2)/2,(y1+y2)/2), (int(x1),int(y1),int(x2),int(y2)), cls))
    return dets


# ═══════════════════════════════════════════════════════════════════
# OVERLAY RENDERER — All visual elements
# ═══════════════════════════════════════════════════════════════════

def draw_overlay(frame, tracked, predictions, conflicts, decision,
                 dec_color, max_prob, fps, stats, tracker,
                 accident_info, crowd_info):
    """
    Draw EVERYTHING on frame:
    - Bounding boxes + IDs
    - Trajectory trails + predicted positions
    - Conflict zones
    - Decision banner (HOLD/CAUTION/ALLOW)
    - Probability bar
    - Stats panel
    - Accident alerts (if active)
    - Crowd density
    - Traffic signal status
    """
    h, w = frame.shape[:2]

    # ── Trajectory trails ──
    for oid, hist in tracker.histories.items():
        if len(hist) < 2: continue
        pts = list(hist)
        cls_id = tracked.get(oid, (None,None,2))[2]
        color = CLASS_COLORS.get(cls_id, WHITE)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            t = max(1, int(alpha * 3))
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, (int(pts[i-1][0]),int(pts[i-1][1])),
                     (int(pts[i][0]),int(pts[i][1])), c, t)

    # ── Conflict zones ──
    for (pid, vid), prob in conflicts.items():
        if prob < CAUTION_THRESHOLD: continue
        p1 = predictions.get(pid, (None,None))[0]
        p2 = predictions.get(vid, (None,None))[0]
        if p1 and p2:
            mx = int((p1[0]+p2[0])/2)
            my = int((p1[1]+p2[1])/2)
            pulse = math.sin(time.time()*6) * 5
            r = max(10, int(25 + prob*50 + pulse))
            zc = RED if prob > HOLD_THRESHOLD else YELLOW
            cv2.circle(frame, (mx,my), r, zc, 2)
            if prob > HOLD_THRESHOLD:
                ov = frame.copy()
                cv2.circle(ov, (mx,my), r, zc, -1)
                cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
            cv2.putText(frame, f"{prob:.0%}", (mx-15,my-r-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zc, 2)

    # ── Bounding boxes + labels ──
    for oid, (centroid, bbox, cls_id) in tracked.items():
        x1,y1,x2,y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(cls_id, WHITE)
        label = TARGET_CLASSES.get(cls_id, "?")

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # Label
        txt = f"#{oid} {label}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+4,y1), color, -1)
        cv2.putText(frame, txt, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
        cv2.circle(frame, (int(centroid[0]),int(centroid[1])), 4, color, -1)

    # ── Predicted positions ──
    for oid, (pred, vel) in predictions.items():
        if pred is None: continue
        px, py = int(pred[0]), int(pred[1])
        cls_id = tracked[oid][2] if oid in tracked else 0
        color = CLASS_COLORS.get(cls_id, WHITE)
        cv2.drawMarker(frame, (px,py), color, cv2.MARKER_CROSS, 14, 1)
        cv2.circle(frame, (px,py), 10, color, 1)
        if oid in tracked:
            cx, cy = tracked[oid][0]
            cv2.arrowedLine(frame, (int(cx),int(cy)), (px,py), color, 1, tipLength=0.12)

    # ═══════════════════════════════════════════════════════════════
    # ACCIDENT OVERLAY (if active)
    # ═══════════════════════════════════════════════════════════════
    if accident_info["active"]:
        # ── FLASHING RED BORDER ──
        flash = int(time.time() * 4) % 2 == 0
        border_color = RED if flash else DARK_RED
        cv2.rectangle(frame, (0,0), (w-1,h-1), border_color, 6)
        cv2.rectangle(frame, (4,4), (w-5,h-5), border_color, 2)

        # ── ACCIDENT DETECTED banner ──
        banner_y = h // 2 - 80
        ov = frame.copy()
        cv2.rectangle(ov, (0, banner_y-10), (w, banner_y+150), (0,0,100), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

        # Accident text
        cv2.putText(frame, "ACCIDENT DETECTED", (w//2-200, banner_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

        # All signals RED
        cv2.putText(frame, "ALL SIGNALS RED", (w//2-140, banner_y+65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)

        # Traffic light icons
        for i, (lx, clr) in enumerate([(w//2-60, RED), (w//2, RED), (w//2+60, RED)]):
            cv2.circle(frame, (lx, banner_y+90), 15, clr, -1)
            cv2.circle(frame, (lx, banner_y+90), 17, WHITE, 2)

        # Severity
        sev = accident_info["severity"]
        sev_color = RED if sev == "HIGH" else YELLOW if sev == "MEDIUM" else GREEN
        cv2.putText(frame, f"Severity: {sev}", (w//2-80, banner_y+125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sev_color, 2)

        # Ambulance alert
        cv2.putText(frame, "AMBULANCE ALERT TRIGGERED", (w//2-190, banner_y+155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2)

        # Accident location marker
        loc = accident_info["location"]
        cv2.circle(frame, loc, 25, RED, 3)
        cv2.drawMarker(frame, loc, RED, cv2.MARKER_TILTED_CROSS, 30, 2)

    # ═══════════════════════════════════════════════════════════════
    # DECISION BANNER (top) — Only show if no accident
    # ═══════════════════════════════════════════════════════════════
    if not accident_info["active"]:
        banner_h = 60
        ov = frame.copy()
        for y in range(banner_h):
            a = 1.0 - (y/banner_h)*0.3
            c = tuple(int(v*a) for v in dec_color)
            cv2.line(ov, (0,y), (w,y), c, 1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, f"SafeTurn: {decision}", (w//2-130, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

        # Probability bar
        bar_y, bar_h, bar_w = banner_h+6, 18, int(w*0.40)
        bar_x = (w-bar_w)//2
        fill = int(bar_w * max_prob)
        cv2.rectangle(frame, (bar_x,bar_y), (bar_x+bar_w,bar_y+bar_h), (60,60,60), -1)
        if fill > 0:
            bc = RED if max_prob>0.7 else YELLOW if max_prob>0.4 else GREEN
            cv2.rectangle(frame, (bar_x,bar_y), (bar_x+fill,bar_y+bar_h), bc, -1)
        cv2.rectangle(frame, (bar_x,bar_y), (bar_x+bar_w,bar_y+bar_h), WHITE, 1)
        # Threshold markers
        cv2.line(frame, (bar_x+int(bar_w*0.4),bar_y), (bar_x+int(bar_w*0.4),bar_y+bar_h), YELLOW, 1)
        cv2.line(frame, (bar_x+int(bar_w*0.7),bar_y), (bar_x+int(bar_w*0.7),bar_y+bar_h), RED, 1)
        cv2.putText(frame, f"Conflict: {max_prob:.1%}", (bar_x+bar_w+10,bar_y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # ═══════════════════════════════════════════════════════════════
    # STATS PANEL (bottom-left)
    # ═══════════════════════════════════════════════════════════════
    panel_w, panel_h = 280, 140
    py_start = h - panel_h - 5
    ov = frame.copy()
    cv2.rectangle(ov, (5,py_start), (5+panel_w,py_start+panel_h), (20,20,20), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (5,py_start), (5+panel_w,py_start+panel_h), (80,80,80), 1)

    cv2.putText(frame, "SESSION STATS", (15,py_start+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
    y_off = py_start + 40
    cv2.putText(frame, f"Conflicts Prevented: {stats['holds']}", (15,y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
    cv2.putText(frame, f"Cautions Issued:     {stats['cautions']}", (15,y_off+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
    cv2.putText(frame, f"Turns Allowed:       {stats['allows']}", (15,y_off+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
    cv2.putText(frame, f"Accidents Detected:  {stats['accidents']}", (15,y_off+60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1)
    cv2.putText(frame, f"Objects Tracked:     {len(tracked)}", (15,y_off+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    # ═══════════════════════════════════════════════════════════════
    # CROWD DENSITY (bottom-right area)
    # ═══════════════════════════════════════════════════════════════
    ped_count, crowd_level, crowd_color = crowd_info
    crowd_y = h - 80
    cv2.putText(frame, f"Crowd: {crowd_level} ({ped_count})",
                (w-200, crowd_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, crowd_color, 2)

    if crowd_level == "HIGH":
        cv2.putText(frame, "HIGH CROWD DENSITY", (w-230, crowd_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # ── FPS (bottom-right) ──
    cv2.putText(frame, f"FPS: {fps:.0f}", (w-110, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # ── Branding ──
    brand_y = 60 if not accident_info["active"] else 25
    cv2.putText(frame, "SafeTurn AI", (w-150, brand_y+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
    cv2.putText(frame, "3s Predictive | HackSETU 2025", (w-280, brand_y+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP — Everything runs here
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SafeTurn AI — Complete System")
    parser.add_argument("--video", type=str, default=None, help="Video file path")
    parser.add_argument("--mock", action="store_true", help="Use mock detections")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--headless", action="store_true", help="No display")
    args = parser.parse_args()

    use_mock = args.mock or (args.video is None and not YOLO_AVAILABLE)

    if use_mock:
        print("=" * 58)
        print("  SafeTurn AI — COMPLETE SYSTEM (MOCK MODE)")
        print("  Core + Accident Detection + Crowd Density")
        print("=" * 58)
        mock = MockDetectionGenerator()
        cap = None
    else:
        if args.video is None:
            print("[ERROR] No video. Use --video <path> or --mock")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {args.video} — using MOCK")
            use_mock, mock, cap = True, MockDetectionGenerator(), None
        else:
            print(f"  SafeTurn AI — VIDEO: {args.video}")

    # Load YOLO
    model = None
    if not use_mock and YOLO_AVAILABLE:
        try:
            model = YOLO(args.model)
            print(f"[INFO] YOLOv8 loaded: {args.model}")
        except Exception as e:
            print(f"[WARN] YOLO failed: {e} — using MOCK")
            use_mock, mock = True, MockDetectionGenerator()

    # Init systems
    tracker = CentroidTracker()
    accident_detector = AccidentDetector()
    stats = {"holds": 0, "cautions": 0, "allows": 0, "accidents": 0}
    fps = 10.0
    frame_count = 0
    prev_decision = None

    print("\n[READY] Q=quit  S=screenshot  A=simulate accident\n")

    # ════════════════════════
    # MAIN LOOP
    # ════════════════════════
    while True:
        t0 = time.time()

        # Get frame
        if use_mock:
            frame = mock.generate_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret: break

        # Detection
        if use_mock:
            detections = mock.generate()
        elif model:
            detections = run_yolo(model, frame)
        else:
            detections = []

        # Tracking
        tracked = tracker.update(detections)

        # Prediction
        predictions = {}
        for oid in tracked:
            h = tracker.histories.get(oid)
            if h and len(h) >= 2:
                predictions[oid] = predict_position(h, fps)

        # Conflict Engine
        peds = [oid for oid, (_,_,c) in tracked.items() if c == 0]
        vehs = [oid for oid, (_,_,c) in tracked.items() if c in VEHICLE_CLASSES]

        conflicts = {}
        max_prob = 0.0
        for pid in peds:
            for vid in vehs:
                if pid in predictions and vid in predictions:
                    prob = compute_conflict_probability(
                        predictions[pid][0], predictions[pid][1],
                        predictions[vid][0], predictions[vid][1])
                    conflicts[(pid, vid)] = prob
                    max_prob = max(max_prob, prob)

        # Decision
        decision, dec_color = get_decision(max_prob)
        if decision != prev_decision:
            if decision == "HOLD": stats["holds"] += 1
            elif decision == "CAUTION": stats["cautions"] += 1
            else: stats["allows"] += 1
            prev_decision = decision

        # ═══ EXTENDED: Accident Detection ═══
        accident_info = accident_detector.update(tracked, tracker, frame_count)
        if accident_info["active"] and accident_info["frame"] == frame_count:
            stats["accidents"] += 1
            print(f"[ACCIDENT] Frame {frame_count} | Severity: {accident_info['severity']} | 🚑 Ambulance Alert!")

        # ═══ EXTENDED: Crowd Density ═══
        crowd_info = get_crowd_density(tracked)

        # Draw everything
        frame = draw_overlay(
            frame, tracked, predictions, conflicts, decision,
            dec_color, max_prob, fps, stats, tracker,
            accident_info, crowd_info)

        # Display
        if not args.headless:
            cv2.imshow("SafeTurn AI — Complete System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                fn = f"safeturn_final_{frame_count:05d}.png"
                cv2.imwrite(fn, frame)
                print(f"[SCREENSHOT] {fn}")
            elif key == ord('a') or key == ord('A'):
                # Simulate accident for demo
                accident_detector.force_trigger(tracked, frame_count)
                stats["accidents"] += 1
                print("[DEMO] Accident simulated! Press A again after cooldown.")

        # FPS
        fps = 1.0 / max(time.time() - t0, 0.001)
        frame_count += 1

        if args.frames > 0 and frame_count >= args.frames:
            break

    # Cleanup
    if cap: cap.release()
    cv2.destroyAllWindows()

    # Final report
    print("\n" + "=" * 55)
    print("  SafeTurn AI — FINAL Session Summary")
    print("=" * 55)
    print(f"  Total Frames:        {frame_count}")
    print(f"  Conflicts Prevented: {stats['holds']}")
    print(f"  Cautions Issued:     {stats['cautions']}")
    print(f"  Turns Allowed:       {stats['allows']}")
    print(f"  Accidents Detected:  {stats['accidents']}")
    print(f"  Final FPS:           {fps:.1f}")
    print("=" * 55)
    print("  'Predictive, not reactive — we act before danger arrives.'")
    print("=" * 55)


if __name__ == "__main__":
    main()
