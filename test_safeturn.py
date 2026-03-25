"""SafeTurn AI — Verification Tests"""
import collections
import sys

# Suppress YOLO warning during import
from safeturn_phase1 import (
    CentroidTracker, predict_position,
    compute_conflict_probability, get_decision
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1

print("=" * 50)
print("  SafeTurn AI — Verification Suite")
print("=" * 50)

# --- Tracker Tests ---
print("\n📦 CentroidTracker:")
t = CentroidTracker()
dets = [
    ((100, 200), (80, 170, 120, 230), 0),   # pedestrian
    ((400, 300), (350, 270, 450, 330), 2),   # car
]
result = t.update(dets)
test("Registers 2 objects", len(result) == 2)

# Update with slightly moved detections
dets2 = [
    ((105, 205), (85, 175, 125, 235), 0),
    ((395, 295), (345, 265, 445, 325), 2),
]
result2 = t.update(dets2)
test("Maintains same IDs after update", set(result.keys()) == set(result2.keys()))

# Empty update
result3 = t.update([])
test("Objects persist with empty detection", len(result3) == 2)

# --- Prediction Tests ---
print("\n🔮 Trajectory Prediction:")
h = collections.deque([(10,10),(20,20),(30,30),(40,40),(50,50)], maxlen=5)
pos, vel = predict_position(h, 10)
test("Returns valid prediction", pos is not None)
test("Prediction moves in correct direction", pos[0] > 50 and pos[1] > 50)
test("Velocity is positive", vel[0] > 0 and vel[1] > 0)

h_short = collections.deque([(100, 100)], maxlen=5)
pos_s, vel_s = predict_position(h_short, 10)
test("Returns None for insufficient history", pos_s is None)

# --- Conflict Engine Tests ---
print("\n⚡ Conflict Engine:")
p_close = compute_conflict_probability(
    (100, 100), (2, 2), (120, 120), (-2, -2)
)
test(f"Head-on close → high prob ({p_close:.2f} > 0.7)", p_close > 0.7)

p_far = compute_conflict_probability(
    (100, 100), (1, 0), (900, 900), (0, 1)
)
test(f"Far apart → low prob ({p_far:.2f} < 0.2)", p_far < 0.2)

p_none = compute_conflict_probability(None, None, (100, 100), (1, 1))
test("None input → zero prob", p_none == 0.0)

# --- Decision Tests ---
print("\n🚦 Decision Engine:")
d1, c1 = get_decision(0.85)
test(f"0.85 → HOLD (got {d1})", d1 == "HOLD")

d2, c2 = get_decision(0.55)
test(f"0.55 → CAUTION (got {d2})", d2 == "CAUTION")

d3, c3 = get_decision(0.20)
test(f"0.20 → ALLOW (got {d3})", d3 == "ALLOW")

d4, c4 = get_decision(0.70)
test(f"0.70 (boundary) → CAUTION (got {d4})", d4 == "CAUTION")

d5, c5 = get_decision(0.40)
test(f"0.40 (boundary) → ALLOW (got {d5})", d5 == "ALLOW")

# --- Summary ---
print(f"\n{'=' * 50}")
total = passed + failed
print(f"  Results: {passed}/{total} passed")
if failed == 0:
    print("  🎉 ALL TESTS PASSED")
else:
    print(f"  ⚠️  {failed} test(s) FAILED")
print(f"{'=' * 50}")

sys.exit(0 if failed == 0 else 1)
