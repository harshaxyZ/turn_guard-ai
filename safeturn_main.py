"""
SafeTurn AI — Unified Single-Camera Traffic Control
=====================================================
HackSETU 2025 | Theme 3: Intelligent Free-Left Turn Management

ONE camera, ONE file, DEMO-READY.

PIPELINE:
  Video/Camera → YOLOv8 → Filter to Zone → Count → Signal Decision

SIGNAL LOGIC:
  0 pedestrians in zone  → 🟢 GREEN  (vehicles move)
  1–2 pedestrians        → 🔴 RED    (short stop — 3s)
  3+ pedestrians         → 🔴 RED    (long stop  — 6s)

ACCIDENT:
  Press 'A' → RED + "ACCIDENT DETECTED" + Ambulance Alert (5s)

Controls:
  Q — Quit
  A — Simulate accident
  R — Rewind video
  S — Screenshot

Usage:
  python safeturn_main.py                          # uses test_video.mp4
  python safeturn_main.py --video test_video1.mp4  # custom video
  python safeturn_main.py --camera 0               # webcam
  python safeturn_main.py --mock                   # no video / no YOLO
"""

import sys
import time
import argparse
import random
import math
import numpy as np

# ─── OpenCV ────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found.  Fix: pip install opencv-python")
    sys.exit(1)

# ─── YOLOv8 (optional — graceful fallback to mock) ────────────────
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not found — will use MOCK detections.")
    print("       Install: pip install ultralytics")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — All tunables in one place
# ═══════════════════════════════════════════════════════════════════

# YOLO model
YOLO_MODEL_PATH      = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.40
PERSON_CLASS_ID      = 0          # COCO class 0 = person
VEHICLE_CLASS_IDS    = {1, 2, 3, 5, 7}  # bicycle, car, motorbike, bus, truck
VEHICLE_NAMES        = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

# ─── Pedestrian Waiting Zone ──────────────────────────────────────
# Fractions of frame size (0.0 to 1.0) — auto-scales to any resolution
# Adjust these to match where the footpath/waiting area is in YOUR video

ZONE_LEFT   = 0.02    # 2% from left
ZONE_RIGHT  = 0.35    # 35% from left
ZONE_TOP    = 0.30    # 30% from top
ZONE_BOTTOM = 0.85    # 85% from top

# ─── Signal Timing (seconds) ─────────────────────────────────────
SHORT_RED_DURATION  = 3.0     # 1–2 pedestrians
LONG_RED_DURATION   = 6.0     # 3+ pedestrians
ACCIDENT_DURATION   = 5.0     # Accident override
MIN_GREEN_HOLD      = 1.5     # Don't flicker — hold green at least this long

# ─── Colors (BGR for OpenCV) ─────────────────────────────────────
COL_RED     = (0, 0, 255)
COL_GREEN   = (0, 200, 0)
COL_YELLOW  = (0, 220, 255)
COL_WHITE   = (255, 255, 255)
COL_BLACK   = (0, 0, 0)
COL_CYAN    = (255, 255, 0)
COL_ZONE    = (0, 255, 255)      # Zone border


# ═══════════════════════════════════════════════════════════════════
# PEDESTRIAN ZONE — compute & draw
# ═══════════════════════════════════════════════════════════════════

def compute_zone(h, w):
    """Convert fractional zone config → pixel coordinates."""
    x1 = int(ZONE_LEFT   * w)
    x2 = int(ZONE_RIGHT  * w)
    y1 = int(ZONE_TOP    * h)
    y2 = int(ZONE_BOTTOM * h)
    return (x1, y1), (x2, y2)


def is_inside_zone(cx, cy, zone_pt1, zone_pt2):
    """Check if a centroid (cx, cy) is inside the zone rectangle."""
    return (zone_pt1[0] <= cx <= zone_pt2[0] and
            zone_pt1[1] <= cy <= zone_pt2[1])


def draw_zone(frame, pt1, pt2, ped_count):
    """
    Draw the pedestrian waiting zone:
    - Semi-transparent fill (green if empty, red if occupied)
    - 3px border
    - Label + count
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Fill color depends on occupancy
    if ped_count == 0:
        fill_color = (0, 180, 0)       # green-ish
        border_color = COL_GREEN
    elif ped_count <= 2:
        fill_color = (0, 180, 255)     # orange-ish
        border_color = COL_YELLOW
    else:
        fill_color = (0, 0, 200)       # red-ish
        border_color = COL_RED

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, fill_color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Solid border
    cv2.rectangle(frame, pt1, pt2, border_color, 3)

    # Label above zone
    label = f"PEDESTRIAN ZONE  [{ped_count} detected]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
    label_y = max(y1 - 10, th + 4)
    cv2.rectangle(frame, (x1 - 2, label_y - th - 4),
                  (x1 + tw + 6, label_y + 4), COL_BLACK, -1)
    cv2.putText(frame, label, (x1 + 2, label_y),
                font, 0.55, border_color, 2)

    return frame


# ═══════════════════════════════════════════════════════════════════
# SIGNAL CONTROLLER — State machine with stability
# ═══════════════════════════════════════════════════════════════════

class SignalController:
    """
    Simple state machine: GREEN ↔ RED
    - Enforces minimum hold times to prevent flickering
    - Accident override forces RED for a set duration
    """

    def __init__(self):
        self.state          = "GREEN"
        self.state_start    = time.time()
        self.red_duration   = 0.0
        self.accident       = False
        self.accident_start = 0.0

    def trigger_accident(self):
        self.accident       = True
        self.accident_start = time.time()
        self.state          = "RED"
        self.state_start    = time.time()
        self.red_duration   = ACCIDENT_DURATION

    def update(self, zone_count):
        """
        Update signal based on pedestrian count in zone.
        Returns: (state: str, color: tuple, remaining: float)
        """
        now     = time.time()
        elapsed = now - self.state_start

        # ── Accident override ─────────────────────────────────────
        if self.accident:
            t = now - self.accident_start
            if t >= ACCIDENT_DURATION:
                self.accident = False
            else:
                return "RED", COL_RED, ACCIDENT_DURATION - t

        # ── Currently RED — hold until timer expires ──────────────
        if self.state == "RED":
            if elapsed < self.red_duration:
                return "RED", COL_RED, self.red_duration - elapsed
            # Timer expired → go GREEN only if zone is clear
            if zone_count == 0:
                self.state = "GREEN"
                self.state_start = now
                return "GREEN", COL_GREEN, 0.0
            else:
                # Still occupied → restart RED timer
                self._set_red(zone_count)
                self.state_start = now
                return "RED", COL_RED, self.red_duration

        # ── Currently GREEN ───────────────────────────────────────
        if zone_count == 0:
            return "GREEN", COL_GREEN, 0.0

        # Pedestrians appeared → switch to RED (but respect min green hold)
        if elapsed < MIN_GREEN_HOLD:
            return "GREEN", COL_GREEN, 0.0

        self._set_red(zone_count)
        self.state       = "RED"
        self.state_start = now
        return "RED", COL_RED, self.red_duration

    def _set_red(self, count):
        self.red_duration = LONG_RED_DURATION if count >= 3 else SHORT_RED_DURATION


# ═══════════════════════════════════════════════════════════════════
# YOLO DETECTION — returns persons + vehicles separately
# ═══════════════════════════════════════════════════════════════════

def detect_objects(model, frame):
    """
    Run YOLOv8 and return detected persons and vehicles.

    Returns:
      persons:  [(x1, y1, x2, y2, cx, cy, conf), ...]
      vehicles: [(x1, y1, x2, y2, cx, cy, conf, class_id), ...]
    """
    results  = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    persons  = []
    vehicles = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cls == PERSON_CLASS_ID:
                persons.append((x1, y1, x2, y2, cx, cy, conf))
            elif cls in VEHICLE_CLASS_IDS:
                vehicles.append((x1, y1, x2, y2, cx, cy, conf, cls))

    return persons, vehicles


# ═══════════════════════════════════════════════════════════════════
# MOCK DETECTION GENERATOR
# ═══════════════════════════════════════════════════════════════════

class MockGenerator:
    """Simulates pedestrians + vehicles for demo without video/YOLO."""

    def __init__(self, w=960, h=540):
        self.w, self.h = w, h
        self.frame_num = 0
        self.ped_count = 0
        self.next_change = time.time() + 2.0

    def generate_frame(self):
        """Dark background with road markings."""
        f = np.full((self.h, self.w, 3), (50, 50, 50), dtype=np.uint8)
        # Road
        cv2.rectangle(f, (0, int(self.h*0.3)), (self.w, int(self.h*0.7)),
                      (42, 42, 42), -1)
        # Crosswalk stripes
        for x in range(50, self.w - 50, 60):
            cv2.rectangle(f, (x, int(self.h*0.36)),
                          (x + 30, int(self.h*0.50)), (180, 180, 180), -1)
        # Free-left label
        cv2.putText(f, "FREE LEFT", (self.w//2 - 60, int(self.h*0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        cv2.putText(f, "MOCK MODE", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(f, time.strftime("%H:%M:%S"), (self.w - 110, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return f

    def generate_detections(self, zone_pt1, zone_pt2):
        """
        Simulate random pedestrians inside-and-outside the zone,
        plus a couple of vehicles.
        """
        self.frame_num += 1

        # Change count periodically
        if time.time() > self.next_change:
            self.ped_count = random.choice([0, 0, 1, 2, 2, 3, 4])
            self.next_change = time.time() + random.uniform(2.0, 4.5)

        persons = []
        zx1, zy1 = zone_pt1
        zx2, zy2 = zone_pt2

        for i in range(self.ped_count):
            # Place inside zone with some noise
            cx = random.randint(zx1 + 20, max(zx1 + 21, zx2 - 20))
            cy = random.randint(zy1 + 40, max(zy1 + 41, zy2 - 40))
            pw, ph = 28, 65
            persons.append((cx - pw//2, cy - ph//2,
                            cx + pw//2, cy + ph//2,
                            cx, cy, 0.85 + random.uniform(-0.1, 0.1)))

        # Sometimes add a person OUTSIDE the zone (should not count)
        if self.frame_num % 5 == 0:
            ox = random.randint(int(self.w * 0.6), int(self.w * 0.85))
            oy = random.randint(int(self.h * 0.3), int(self.h * 0.6))
            persons.append((ox - 14, oy - 32, ox + 14, oy + 32, ox, oy, 0.78))

        # Add vehicles
        vehicles = []
        vx = int(self.w * 0.65 + math.sin(self.frame_num * 0.05) * 80)
        vy = int(self.h * 0.60)
        vehicles.append((vx - 50, vy - 25, vx + 50, vy + 25, vx, vy, 0.90, 2))

        if self.frame_num % 3 != 0:
            vx2 = int(self.w * 0.75 + math.cos(self.frame_num * 0.03) * 40)
            vy2 = int(self.h * 0.80)
            vehicles.append((vx2 - 22, vy2 - 16, vx2 + 22, vy2 + 16, vx2, vy2, 0.82, 3))

        return persons, vehicles


# ═══════════════════════════════════════════════════════════════════
# OVERLAY RENDERER
# ═══════════════════════════════════════════════════════════════════

def draw_detections(frame, persons, vehicles, zone_pt1, zone_pt2):
    """Draw bounding boxes — persons are cyan (green if in zone), vehicles are blue."""
    for (x1, y1, x2, y2, cx, cy, conf) in persons:
        in_zone = is_inside_zone(cx, cy, zone_pt1, zone_pt2)
        color = COL_GREEN if in_zone else (180, 180, 180)
        label = f"Person {conf:.0%}" + (" [ZONE]" if in_zone else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.circle(frame, (cx, cy), 3, color, -1)

    for det in vehicles:
        x1, y1, x2, y2, cx, cy, conf, cls = det
        name = VEHICLE_NAMES.get(cls, "vehicle")
        color = (255, 120, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def draw_signal_banner(frame, signal_state, signal_color, remaining,
                       zone_count, accident_active):
    """Big banner + signal light in top-right corner."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top banner (semi-transparent) ─────────────────────────────
    banner_h = 55
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, banner_h), signal_color, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # Signal text
    cv2.putText(frame, f"Signal: {signal_state}", (12, 38),
                font, 1.1, COL_WHITE, 2)

    # Density text
    if zone_count == 0:
        density = "No pedestrians - FREE TURN"
        dc = COL_GREEN
    elif zone_count <= 2:
        density = f"{zone_count} pedestrian(s) - SHORT STOP"
        dc = COL_YELLOW
    else:
        density = f"{zone_count} pedestrians - LONG STOP"
        dc = COL_RED
    cv2.putText(frame, density, (12, banner_h + 22), font, 0.55, dc, 2)

    # ── Signal light (top-right circle) ───────────────────────────
    lx = w - 55
    ly = 40
    lr = 30
    cv2.rectangle(frame, (lx - 38, 5), (lx + 38, 80), (25, 25, 25), -1)
    cv2.rectangle(frame, (lx - 38, 5), (lx + 38, 80), (90, 90, 90), 2)
    cv2.circle(frame, (lx, ly), lr, signal_color, -1)
    cv2.circle(frame, (lx, ly), lr, COL_WHITE, 2)

    # RED timer
    if signal_state == "RED" and remaining > 0:
        cv2.putText(frame, f"{remaining:.1f}s", (lx - 22, 100),
                    font, 0.6, COL_RED, 2)

    # ── Accident overlay ──────────────────────────────────────────
    if accident_active:
        flash = int(time.time() * 3) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COL_RED, 6)

        # Accident banner (center of frame)
        bx, by = w // 2, h // 2
        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx - 220, by - 60), (bx + 220, by + 70),
                      (0, 0, 120), -1)
        cv2.addWeighted(ov2, 0.8, frame, 0.2, 0, frame)

        cv2.putText(frame, "ACCIDENT DETECTED", (bx - 185, by - 20),
                    font, 1.0, COL_WHITE, 2)
        cv2.putText(frame, "ALL SIGNALS RED", (bx - 120, by + 15),
                    font, 0.7, COL_RED, 2)
        cv2.putText(frame, "Ambulance Alert Triggered",
                    (bx - 160, by + 50), font, 0.6, COL_CYAN, 2)

    return frame


def draw_info_panel(frame, fps, frame_num, total_frames):
    """Bottom bar with controls, FPS, and branding."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dark bar at bottom
    cv2.rectangle(frame, (0, h - 32), (w, h), (20, 20, 20), -1)

    # Controls
    cv2.putText(frame, "Q=Quit  A=Accident  R=Rewind  S=Screenshot",
                (8, h - 10), font, 0.4, (180, 180, 180), 1)

    # Frame + FPS
    info = f"Frame {frame_num}"
    if total_frames > 0:
        info += f"/{total_frames}"
    info += f"  |  FPS: {fps:.0f}"
    cv2.putText(frame, info, (w - 280, h - 10), font, 0.4, (180, 180, 180), 1)

    # Branding (above bottom bar)
    cv2.putText(frame, "SafeTurn AI | HackSETU 2025",
                (w - 240, h - 40), font, 0.42, COL_CYAN, 1)

    return frame


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SafeTurn AI — Single-Camera Traffic Control"
    )
    parser.add_argument("--video",  type=str, default="test_video.mp4",
                        help="Path to video file (default: test_video.mp4)")
    parser.add_argument("--camera", type=int, default=None,
                        help="Webcam index (overrides --video)")
    parser.add_argument("--mock",   action="store_true",
                        help="Run with mock detections (no video / no YOLO)")
    parser.add_argument("--model",  type=str, default=YOLO_MODEL_PATH,
                        help="YOLOv8 model path (default: yolov8n.pt)")
    args = parser.parse_args()

    # ── Flags ─────────────────────────────────────────────────────
    use_mock = args.mock
    use_yolo = YOLO_AVAILABLE and not use_mock

    # ── Load YOLO ─────────────────────────────────────────────────
    model = None
    if use_yolo:
        print(f"[INFO] Loading YOLOv8: {args.model}")
        try:
            model = YOLO(args.model)
            print("[INFO] YOLOv8 loaded OK.")
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e} — falling back to MOCK.")
            use_yolo = False
            use_mock = True

    # ── Open video / camera ───────────────────────────────────────
    cap = None
    total_frames = 0
    mock_gen = None

    if use_mock:
        mock_gen = MockGenerator()
        print("[INFO] MOCK mode — synthetic detections.")
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[WARN] Cannot open webcam {args.camera} — using MOCK.")
            use_mock = True
            mock_gen = MockGenerator()
        else:
            print(f"[INFO] Webcam {args.camera} opened.")
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[WARN] Cannot open {args.video} — using MOCK.")
            use_mock = True
            mock_gen = MockGenerator()
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Video: {args.video}  ({total_frames} frames, {vid_fps:.0f} FPS)")

    # ── Init ──────────────────────────────────────────────────────
    signal      = SignalController()
    frame_num   = 0
    prev_time   = time.time()
    display_fps = 0.0

    print()
    print("=" * 55)
    print("  SafeTurn AI — Single-Camera Traffic Control")
    print("  Q=Quit  A=Accident  R=Rewind  S=Screenshot")
    print("=" * 55)
    print()

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════
    while True:
        # ── 1. Get frame ──────────────────────────────────────────
        if use_mock:
            frame = mock_gen.generate_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                if args.camera is not None:
                    print("[WARN] Camera read failed.")
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                continue

        frame_num += 1
        h, w = frame.shape[:2]

        # ── 2. Compute zone ───────────────────────────────────────
        zone_pt1, zone_pt2 = compute_zone(h, w)

        # ── 3. Detect objects ─────────────────────────────────────
        if use_mock:
            persons, vehicles = mock_gen.generate_detections(zone_pt1, zone_pt2)
        elif model is not None:
            persons, vehicles = detect_objects(model, frame)
        else:
            persons, vehicles = [], []

        # ── 4. Count persons INSIDE zone only ─────────────────────
        zone_count = 0
        for p in persons:
            cx, cy = p[4], p[5]
            if is_inside_zone(cx, cy, zone_pt1, zone_pt2):
                zone_count += 1

        # ── 5. Update signal ─────────────────────────────────────
        sig_state, sig_color, remaining = signal.update(zone_count)

        # ── 6. Draw everything ────────────────────────────────────
        frame = draw_detections(frame, persons, vehicles, zone_pt1, zone_pt2)
        frame = draw_zone(frame, zone_pt1, zone_pt2, zone_count)
        frame = draw_signal_banner(frame, sig_state, sig_color, remaining,
                                   zone_count, signal.accident)

        now = time.time()
        display_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        frame = draw_info_panel(frame, display_fps, frame_num, total_frames)

        # ── 7. Show ───────────────────────────────────────────────
        cv2.imshow("SafeTurn AI", frame)

        # ── 8. Key handling ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("[INFO] Quit.")
            break

        elif key == ord('a'):
            signal.trigger_accident()
            print(f"[ALERT] Accident simulated! RED for {ACCIDENT_DURATION:.0f}s")

        elif key == ord('r'):
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                print("[INFO] Rewound.")

        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}")

    # ── Cleanup ───────────────────────────────────────────────────
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[DONE] SafeTurn AI finished.")


if __name__ == "__main__":
    main()
