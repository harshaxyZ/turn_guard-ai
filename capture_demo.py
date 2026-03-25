"""Quick demo capture — runs SafeTurn AI and saves screenshot frames."""
import sys
sys.argv = ['', '--mock', '--headless', '--frames', '80']

# Import everything from safeturn_final
from safeturn_final import *

# Run a custom loop that saves specific frames
mock = MockDetectionGenerator()
tracker = CentroidTracker()
accident_detector = AccidentDetector()
stats = {"holds": 0, "cautions": 0, "allows": 0, "accidents": 0}
fps = 30.0
prev_decision = None

for frame_count in range(80):
    frame = mock.generate_frame()
    detections = mock.generate()
    tracked = tracker.update(detections)

    predictions = {}
    for oid in tracked:
        h = tracker.histories.get(oid)
        if h and len(h) >= 2:
            predictions[oid] = predict_position(h, fps)

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

    decision, dec_color = get_decision(max_prob)
    if decision != prev_decision:
        if decision == "HOLD": stats["holds"] += 1
        elif decision == "CAUTION": stats["cautions"] += 1
        else: stats["allows"] += 1
        prev_decision = decision

    accident_info = accident_detector.update(tracked, tracker, frame_count)
    crowd_info = get_crowd_density(tracked)

    frame = draw_overlay(frame, tracked, predictions, conflicts, decision,
                         dec_color, max_prob, fps, stats, tracker,
                         accident_info, crowd_info)

    # Save normal mode screenshot
    if frame_count == 55:
        cv2.imwrite("demo_normal.png", frame)
        print(f"[SAVED] demo_normal.png — Decision: {decision}, Prob: {max_prob:.1%}")

    # Trigger accident at frame 60 and save
    if frame_count == 60:
        accident_detector.force_trigger(tracked, frame_count)
        stats["accidents"] += 1

    if frame_count == 62:
        accident_info = accident_detector.update(tracked, tracker, frame_count)
        frame2 = mock.generate_frame()
        detections2 = mock.generate()
        tracked2 = tracker.update(detections2)
        for oid in tracked2:
            hh = tracker.histories.get(oid)
            if hh and len(hh) >= 2:
                predictions[oid] = predict_position(hh, fps)
        frame2 = draw_overlay(frame2, tracked2, predictions, conflicts, decision,
                              dec_color, max_prob, fps, stats, tracker,
                              accident_info, crowd_info)
        cv2.imwrite("demo_accident.png", frame2)
        print(f"[SAVED] demo_accident.png — ACCIDENT MODE")

print("[DONE] Demo screenshots saved!")
