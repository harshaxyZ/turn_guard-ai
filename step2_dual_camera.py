"""
SafeTurn AI — Step 2: Dual Camera System
==========================================
HackSETU 2025

Two windows:
  1. "Pedestrian Camera" — webcam with YOLOv8 person detection + count
  2. "Traffic View"      — video file with signal overlay (RED / GREEN)

Decision Logic:
  0 people    → 🟢 GREEN   (vehicles move)
  1–2 people  → 🔴 RED     (short stop — 3s)
  3+ people   → 🔴 RED     (long stop  — 6s)

Accident Simulation:
  Press 'A' → trigger accident → RED + ambulance alert for 5s

Controls:
  Q — Quit
  A — Simulate accident
  R — Rewind traffic video

Usage:
  python step2_dual_camera.py                          # webcam + test_video.mp4
  python step2_dual_camera.py --video test_video1.mp4  # custom video
  python step2_dual_camera.py --mock                   # no webcam / no YOLO
"""

import sys
import time
import argparse
import random
import numpy as np

# ─── OpenCV ────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found.  Fix: pip install opencv-python")
    sys.exit(1)

# ─── YOLOv8 (optional) ────────────────────────────────────────────
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not found — will use MOCK pedestrian counts.")
    print("       Install: pip install ultralytics")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# YOLO
YOLO_MODEL_PATH     = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.40
PERSON_CLASS_ID      = 0           # COCO class 0 = person

# Signal timing (seconds)
SHORT_RED_DURATION   = 3.0         # 1–2 pedestrians
LONG_RED_DURATION    = 6.0         # 3+ pedestrians
ACCIDENT_DURATION    = 5.0         # Accident override
MIN_GREEN_HOLD       = 1.0         # Hold GREEN at least this long

# Colors (BGR)
COL_RED    = (0, 0, 255)
COL_GREEN  = (0, 200, 0)
COL_YELLOW = (0, 220, 255)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_CYAN   = (255, 255, 0)
COL_ORANGE = (0, 140, 255)


# ═══════════════════════════════════════════════════════════════════
# SIGNAL STATE MACHINE
# ═══════════════════════════════════════════════════════════════════

class SignalController:
    """
    Manages the traffic signal state with stability rules.
    Prevents rapid flickering by enforcing minimum hold times.
    """

    def __init__(self):
        self.state       = "GREEN"       # Current signal: GREEN or RED
        self.state_start = time.time()   # When current state started
        self.red_duration = 0.0          # How long RED should last
        self.accident     = False        # Accident override active?
        self.accident_start = 0.0

    def trigger_accident(self):
        """Activate accident override → RED for ACCIDENT_DURATION."""
        self.accident = True
        self.accident_start = time.time()
        self.state = "RED"
        self.state_start = time.time()
        self.red_duration = ACCIDENT_DURATION

    def update(self, ped_count):
        """
        Update signal based on pedestrian count.
        Returns: (state_str, color_bgr, remaining_seconds)
        """
        now = time.time()
        elapsed = now - self.state_start

        # ── Accident override ─────────────────────────────────────
        if self.accident:
            accident_elapsed = now - self.accident_start
            if accident_elapsed >= ACCIDENT_DURATION:
                # Accident period over — clear it
                self.accident = False
            else:
                remaining = ACCIDENT_DURATION - accident_elapsed
                return "RED", COL_RED, remaining

        # ── Currently RED — wait for duration to expire ───────────
        if self.state == "RED":
            if elapsed < self.red_duration:
                remaining = self.red_duration - elapsed
                return "RED", COL_RED, remaining
            else:
                # RED expired → go GREEN if no pedestrians
                if ped_count == 0:
                    self.state = "GREEN"
                    self.state_start = now
                    return "GREEN", COL_GREEN, 0.0
                else:
                    # Still pedestrians → extend RED
                    self._set_red_for(ped_count)
                    return "RED", COL_RED, self.red_duration

        # ── Currently GREEN ───────────────────────────────────────
        if ped_count == 0:
            return "GREEN", COL_GREEN, 0.0

        # Pedestrians detected → switch to RED
        # But only if we've been GREEN for at least MIN_GREEN_HOLD
        if elapsed < MIN_GREEN_HOLD:
            return "GREEN", COL_GREEN, 0.0

        self._set_red_for(ped_count)
        self.state = "RED"
        self.state_start = now
        return "RED", COL_RED, self.red_duration

    def _set_red_for(self, ped_count):
        """Set RED duration based on pedestrian density."""
        if ped_count >= 3:
            self.red_duration = LONG_RED_DURATION
        else:
            self.red_duration = SHORT_RED_DURATION


# ═══════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════

def draw_pedestrian_window(frame, ped_count, detections, accident_active):
    """
    Draw overlay on the pedestrian camera feed.
    Shows: bounding boxes, pedestrian count, accident alert.
    """
    h, w = frame.shape[:2]

    # ── Draw bounding boxes for each detected person ──────────────
    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_CYAN, 2)
        label = f"Person {conf:.0%}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CYAN, 1)

    # ── Top banner: pedestrian count ──────────────────────────────
    banner_h = 50
    overlay = frame.copy()
    banner_color = COL_GREEN if ped_count == 0 else COL_RED
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    count_text = f"Pedestrian Count: {ped_count}"
    cv2.putText(frame, count_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_WHITE, 2)

    # ── Accident alert ────────────────────────────────────────────
    if accident_active:
        # Flashing red bar
        if int(time.time() * 3) % 2 == 0:
            cv2.rectangle(frame, (0, banner_h), (w, banner_h + 40), COL_RED, -1)
            cv2.putText(frame, "!! ACCIDENT DETECTED !!", (10, banner_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)

        # Ambulance alert at bottom
        cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "Ambulance Alert Triggered", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2)

    # ── Title ─────────────────────────────────────────────────────
    cv2.putText(frame, "SafeTurn AI", (w - 140, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1)

    return frame


def draw_traffic_window(frame, signal_state, signal_color, remaining,
                        ped_count, accident_active):
    """
    Draw overlay on the traffic camera feed.
    Shows: BIG signal indicator, timer, density info.
    """
    h, w = frame.shape[:2]

    # ── Signal light (big circle + background) ────────────────────
    light_x = w - 70
    light_y = 80
    light_r = 45

    # Dark housing behind the signal
    cv2.rectangle(frame, (light_x - 55, 15), (light_x + 55, 155),
                  (30, 30, 30), -1)
    cv2.rectangle(frame, (light_x - 55, 15), (light_x + 55, 155),
                  (100, 100, 100), 2)

    # Signal circle
    cv2.circle(frame, (light_x, light_y), light_r, signal_color, -1)
    cv2.circle(frame, (light_x, light_y), light_r, COL_WHITE, 2)

    # Signal text inside circle
    cv2.putText(frame, signal_state,
                (light_x - 25 if signal_state == "GREEN" else light_x - 15,
                 light_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_BLACK, 2)

    # ── Timer below signal ────────────────────────────────────────
    if signal_state == "RED" and remaining > 0:
        timer_text = f"{remaining:.1f}s"
        cv2.putText(frame, timer_text, (light_x - 25, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_RED, 2)

    # ── Top-left info banner ──────────────────────────────────────
    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w - 130, banner_h), signal_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status_text = f"Signal: {signal_state}"
    cv2.putText(frame, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_WHITE, 2)

    # ── Density info ──────────────────────────────────────────────
    if ped_count == 0:
        density_label = "No pedestrians — FREE TURN"
        density_color = COL_GREEN
    elif ped_count <= 2:
        density_label = f"{ped_count} pedestrian(s) — SHORT STOP"
        density_color = COL_YELLOW
    else:
        density_label = f"{ped_count} pedestrians — LONG STOP"
        density_color = COL_RED

    cv2.putText(frame, density_label, (10, banner_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_color, 2)

    # ── Accident overlay ──────────────────────────────────────────
    if accident_active:
        if int(time.time() * 3) % 2 == 0:
            cv2.rectangle(frame, (0, h - 60), (w, h), COL_RED, -1)
            cv2.putText(frame, "ACCIDENT — ALL STOP",
                        (w // 2 - 150, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_WHITE, 2)

    # ── Controls hint ─────────────────────────────────────────────
    hints = "Q=Quit  A=Accident  R=Rewind"
    cv2.putText(frame, hints, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # ── Branding ──────────────────────────────────────────────────
    cv2.putText(frame, "SafeTurn AI | HackSETU 2025", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_CYAN, 1)

    return frame


# ═══════════════════════════════════════════════════════════════════
# YOLO DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_persons(model, frame):
    """
    Run YOLOv8 on frame, return only 'person' detections.
    Returns: list of (x1, y1, x2, y2, confidence)
    """
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    persons = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            persons.append((x1, y1, x2, y2, conf))

    return persons


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SafeTurn AI — Step 2: Dual Camera System"
    )
    parser.add_argument("--video", type=str, default="test_video.mp4",
                        help="Traffic video file (default: test_video.mp4)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default: 0)")
    parser.add_argument("--mock", action="store_true",
                        help="Skip webcam — use random pedestrian counts")
    parser.add_argument("--model", type=str, default=YOLO_MODEL_PATH,
                        help="YOLOv8 model path")
    args = parser.parse_args()

    # ── Setup flags ───────────────────────────────────────────────
    use_mock_cam  = args.mock
    use_yolo      = YOLO_AVAILABLE and not args.mock

    # ── Load YOLO model ───────────────────────────────────────────
    model = None
    if use_yolo:
        print(f"[INFO] Loading YOLOv8: {args.model}")
        try:
            model = YOLO(args.model)
            print("[INFO] YOLOv8 loaded OK.")
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e}")
            print("       Falling back to MOCK mode.")
            use_yolo = False
            use_mock_cam = True

    # ── Open pedestrian camera (webcam) ───────────────────────────
    ped_cap = None
    if not use_mock_cam:
        ped_cap = cv2.VideoCapture(args.camera)
        if not ped_cap.isOpened():
            print(f"[WARN] Cannot open webcam {args.camera} — using MOCK mode.")
            use_mock_cam = True
            ped_cap = None
        else:
            print(f"[INFO] Webcam {args.camera} opened OK.")

    if use_mock_cam:
        print("[INFO] MOCK mode — simulating pedestrian counts (0–5).")

    # ── Open traffic video ────────────────────────────────────────
    traf_cap = cv2.VideoCapture(args.video)
    if not traf_cap.isOpened():
        print(f"[ERROR] Cannot open traffic video: {args.video}")
        sys.exit(1)
    total_frames = int(traf_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps      = traf_cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Traffic video: {args.video}  ({total_frames} frames, {vid_fps:.0f} FPS)")

    # ── Initialize ────────────────────────────────────────────────
    signal = SignalController()
    mock_count = 0
    mock_change_time = time.time()

    print()
    print("=" * 55)
    print("  SafeTurn AI — Dual Camera System")
    print("  Q = Quit | A = Simulate Accident | R = Rewind")
    print("=" * 55)
    print()

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════
    while True:
        # ── 1. Read pedestrian frame ──────────────────────────────
        detections = []

        if use_mock_cam:
            # Generate a blank frame with "MOCK" label
            ped_frame = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
            cv2.putText(ped_frame, "MOCK — No Webcam", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            # Change mock count every 2–4 seconds
            if time.time() - mock_change_time > random.uniform(2.0, 4.0):
                mock_count = random.randint(0, 5)
                mock_change_time = time.time()
            ped_count = mock_count
        else:
            ret, ped_frame = ped_cap.read()
            if not ret:
                ped_frame = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
                cv2.putText(ped_frame, "Webcam Error", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_RED, 2)
                ped_count = 0
            elif use_yolo and model is not None:
                detections = detect_persons(model, ped_frame)
                ped_count = len(detections)
            else:
                ped_count = 0

        # ── 2. Read traffic frame ─────────────────────────────────
        ret2, traf_frame = traf_cap.read()
        if not ret2:
            traf_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # Loop video
            ret2, traf_frame = traf_cap.read()
            if not ret2:
                break

        # ── 3. Update signal ─────────────────────────────────────
        signal_state, signal_color, remaining = signal.update(ped_count)

        # ── 4. Draw pedestrian window ────────────────────────────
        ped_display = draw_pedestrian_window(
            ped_frame, ped_count, detections, signal.accident
        )

        # ── 5. Draw traffic window ───────────────────────────────
        traf_display = draw_traffic_window(
            traf_frame, signal_state, signal_color, remaining,
            ped_count, signal.accident
        )

        # ── 6. Show windows ──────────────────────────────────────
        cv2.imshow("Pedestrian Camera", ped_display)
        cv2.imshow("Traffic View", traf_display)

        # ── 7. Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:        # Q / ESC → quit
            print("[INFO] Quit.")
            break

        elif key == ord('a'):                    # A → simulate accident
            signal.trigger_accident()
            print("[ALERT] Accident simulated! Signal → RED for"
                  f" {ACCIDENT_DURATION:.0f}s")

        elif key == ord('r'):                    # R → rewind traffic video
            traf_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("[INFO] Traffic video rewound.")

    # ── Cleanup ─────────────────────────────────────────────────
    if ped_cap is not None:
        ped_cap.release()
    traf_cap.release()
    cv2.destroyAllWindows()
    print("[DONE] SafeTurn Step 2 finished.")


if __name__ == "__main__":
    main()
