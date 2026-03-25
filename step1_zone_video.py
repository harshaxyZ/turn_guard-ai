"""
SafeTurn AI — Step 1: Video Input + Pedestrian Zone
=====================================================
HackSETU 2025

What this script does:
  1. Opens a video file (or webcam if no file given)
  2. Draws a fixed "Pedestrian Waiting Zone" rectangle on every frame
  3. Shows the live feed so you can confirm the zone position

Controls:
  Q — Quit
  R — Reset / Rewind video to beginning

Usage:
  python step1_zone_video.py                      # uses test_video.mp4
  python step1_zone_video.py --video myfile.mp4   # custom video
  python step1_zone_video.py --camera 0           # use webcam
"""

import sys
import argparse

# ─── OpenCV ────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[FATAL] OpenCV not found.")
    print("        Fix: pip install opencv-python")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# ZONE CONFIGURATION — adjust these values to match your camera view
# ═══════════════════════════════════════════════════════════════════
# These are fractions of the frame size (0.0 to 1.0), so the zone
# scales automatically with any video resolution.

ZONE_LEFT   = 0.02   # left edge   (2% from left)
ZONE_RIGHT  = 0.35   # right edge  (35% from left)
ZONE_TOP    = 0.30   # top edge    (30% from top)
ZONE_BOTTOM = 0.85   # bottom edge (85% from top)

# Zone overlay color (BGR) and opacity
ZONE_COLOR   = (0, 255, 255)   # Cyan border
ZONE_FILL    = (0, 200, 200)   # Cyan fill (semi-transparent)
ZONE_OPACITY = 0.15            # Fill transparency (0=invisible, 1=solid)

# ═══════════════════════════════════════════════════════════════════

def compute_zone(frame_h, frame_w):
    """Convert fractional zone coords → pixel coords."""
    x1 = int(ZONE_LEFT   * frame_w)
    x2 = int(ZONE_RIGHT  * frame_w)
    y1 = int(ZONE_TOP    * frame_h)
    y2 = int(ZONE_BOTTOM * frame_h)
    return (x1, y1), (x2, y2)   # top-left, bottom-right


def draw_pedestrian_zone(frame, pt1, pt2):
    """
    Draw the pedestrian waiting zone on the frame.
    - Semi-transparent fill
    - Thick solid border (3 px)
    - "PEDESTRIAN ZONE" label above the rectangle
    - Zone pixel coordinates as small text inside the box
    """
    x1, y1 = pt1
    x2, y2 = pt2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Semi-transparent fill ──────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, ZONE_FILL, thickness=-1)
    cv2.addWeighted(overlay, ZONE_OPACITY, frame, 1 - ZONE_OPACITY, 0, frame)

    # ── Thick solid border (3 px) ──────────────────────────────────
    cv2.rectangle(frame, pt1, pt2, ZONE_COLOR, thickness=3)

    # ── Label ABOVE the rectangle ──────────────────────────────────
    label = "PEDESTRIAN ZONE"
    font_scale = 0.6
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = x1
    label_y = max(y1 - 10, th + 4)   # don't clip above frame top
    # Dark background behind label
    cv2.rectangle(frame,
                  (label_x - 2, label_y - th - 4),
                  (label_x + tw + 4, label_y + 4),
                  (0, 0, 0), -1)
    cv2.putText(frame, label, (label_x, label_y),
                font, font_scale, ZONE_COLOR, thickness)

    # ── Zone coordinates (small, inside top-left of box) ──────────
    coord_text = f"({x1},{y1}) - ({x2},{y2})"
    cv2.putText(frame, coord_text, (x1 + 5, y1 + 16),
                font, 0.38, (180, 255, 255), 1)

    return frame


def draw_info_overlay(frame, frame_num, total_frames, fps):
    """Show frame counter, FPS, and key instructions."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top-right info panel ───────────────────────────────────────
    info_lines = [
        "SafeTurn AI  Step 1",
        f"Frame: {frame_num}/{total_frames if total_frames > 0 else '?'}",
        f"FPS: {fps:.1f}",
    ]
    x_start = w - 210
    y_start = 10
    line_h  = 20
    box_h   = len(info_lines) * line_h + 10
    cv2.rectangle(frame, (x_start - 6, y_start),
                  (w - 6, y_start + box_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x_start - 6, y_start),
                  (w - 6, y_start + box_h), (100, 100, 100), 1)
    for i, line in enumerate(info_lines):
        color = (0, 220, 220) if i == 0 else (200, 200, 200)
        cv2.putText(frame, line, (x_start, y_start + (i + 1) * line_h),
                    font, 0.45, color, 1)

    # ── Bottom-left instructions panel ────────────────────────────
    instr_lines = [
        "Controls:",
        "  Q  ->  Quit",
        "  R  ->  Restart",
    ]
    ix = 8
    iy = h - len(instr_lines) * 20 - 10
    box_w = 145
    box_h2 = len(instr_lines) * 20 + 8
    cv2.rectangle(frame, (ix - 4, iy - 4),
                  (ix + box_w, iy + box_h2), (20, 20, 20), -1)
    cv2.rectangle(frame, (ix - 4, iy - 4),
                  (ix + box_w, iy + box_h2), (100, 100, 100), 1)
    for i, line in enumerate(instr_lines):
        color = (0, 220, 220) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (ix, iy + (i + 1) * 20),
                    font, 0.45, color, 1)

    return frame


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SafeTurn AI Step 1 — Video Input + Pedestrian Zone"
    )
    parser.add_argument("--video",  type=str, default="test_video.mp4",
                        help="Path to video file (default: test_video.mp4)")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use instead of a video file")
    args = parser.parse_args()

    # ── Open capture ────────────────────────────────────────────────
    if args.camera is not None:
        source = args.camera
        cap = cv2.VideoCapture(source)
        total_frames = 0   # Unknown for live camera
        print(f"[INFO] Using webcam index {source}")
    else:
        source = args.video
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Video : {source}")
        print(f"[INFO] Frames: {total_frames}  |  FPS: {vid_fps:.1f}")

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        print("        Check the file path, or use --camera 0 for webcam.")
        sys.exit(1)

    print("[READY] Window will open. Press Q to quit, R to rewind.\n")

    import time
    frame_num   = 0
    prev_time   = time.time()
    display_fps = 0.0

    while True:
        ret, frame = cap.read()

        # ── Handle end of video ──────────────────────────────────────
        if not ret:
            if args.camera is not None:
                print("[WARN] Camera read failed.")
                break
            else:
                # Loop the video automatically
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                continue

        frame_num += 1
        h, w = frame.shape[:2]

        # ── Compute zone pixel coords ────────────────────────────────
        pt1, pt2 = compute_zone(h, w)

        # ── Draw pedestrian zone ─────────────────────────────────────
        frame = draw_pedestrian_zone(frame, pt1, pt2)

        # ── Draw info overlay ────────────────────────────────────────
        now          = time.time()
        display_fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time    = now
        frame = draw_info_overlay(frame, frame_num, total_frames, display_fps)

        # ── Show result ──────────────────────────────────────────────
        cv2.imshow("SafeTurn AI — Step 1: Pedestrian Zone", frame)

        # ── Key handling ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q or ESC → quit
            print("[INFO] Quit.")
            break
        elif key == ord('r'):              # R → rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            print("[INFO] Rewound to start.")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Step 1 complete.")


if __name__ == "__main__":
    main()
