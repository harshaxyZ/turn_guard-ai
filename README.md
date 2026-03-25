# 🛡️ SafeTurn AI

**Predictive Free-Left Turn Safety System**

> *"Predictive, not reactive — we act before danger arrives."*

Built for **HackSETU 2025** | Theme 3: Intelligent Free-Left Turn Management for Safer Roads

---

## 💡 What It Does

SafeTurn AI watches a junction camera feed and **predicts pedestrian-vehicle collisions 3 seconds before they happen**. When a free-left turning vehicle is on a path to intersect with a crossing pedestrian, the system fires a **HOLD** signal to restrict the turn — *before anyone is in danger*.

```
    ┌─────────────────────────────────────────────┐
    │            SafeTurn AI Pipeline              │
    │                                              │
    │  CCTV Feed                                   │
    │     │                                        │
    │     ▼                                        │
    │  YOLOv8 Detection (pedestrians, vehicles)    │
    │     │                                        │
    │     ▼                                        │
    │  Centroid Tracking (persistent object IDs)   │
    │     │                                        │
    │     ▼                                        │
    │  Trajectory Prediction (3-second lookahead)  │
    │     │                                        │
    │     ▼                                        │
    │  Conflict Engine (collision probability)     │
    │     │                                        │
    │     ▼                                        │
    │  Decision: 🟢 ALLOW │ 🟡 CAUTION │ 🔴 HOLD  │
    └─────────────────────────────────────────────┘
```

---

## 🧠 Key Innovation

| Feature | SafeTurn AI | Traditional Systems |
|---------|-------------|-------------------|
| Approach | **Predictive** — acts 3s before collision | Reactive — responds after detection |
| Tracking | Persistent object IDs + velocity history | Frame-by-frame, no memory |
| Decision | Probability-based (0-100%) | Binary signal timers |
| Hardware | Existing CCTV — **₹6,000/junction** | New sensors needed |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with a Video

```bash
python safeturn_phase1.py --video traffic.mp4
```

### 3. Run in Demo Mode (No Video Needed)

```bash
python safeturn_phase1.py --mock
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |

---

## 🚨 Extended Features (Post-Accident Response)

When an accident IS detected (overlap + speed drop):

| Feature | What Happens |
|---------|-------------|
| 🚧 Accident Detection | Bounding box overlap + speed drop = collision |
| 🔴 Traffic Control | ALL SIGNALS RED displayed immediately |
| ⚡ Severity Estimation | HIGH / MEDIUM / LOW based on speed + overlap |
| 🚑 Ambulance Alert | "Ambulance Alert Triggered" displayed |
| 👥 Crowd Density | Pedestrian count → LOW / MEDIUM / HIGH |

Press **A** during the demo to simulate an accident!

---

## ⚙️ How the Prediction Works

1. **Detect** objects using YOLOv8 nano (optimized for CPU)
2. **Track** each object across frames with unique IDs
3. **Record** last 5 positions to compute velocity vectors
4. **Extrapolate** each trajectory 3 seconds into the future
5. **Compute conflict probability** for every pedestrian-vehicle pair:
   - Distance between predicted positions (60% weight)
   - Closing speed between objects (25% weight)
   - Trajectory convergence angle (15% weight)
6. **Fire decision**:
   - `> 70%` → 🔴 **HOLD** — restrict the turn
   - `> 40%` → 🟡 **CAUTION** — warn
   - `≤ 40%` → 🟢 **ALLOW** — safe to proceed

---

## 🏗️ Architecture

```
safeturn_final.py           ← COMPLETE SYSTEM (recommended for demo)
├── CentroidTracker         ← Object tracking (centroid matching)
├── predict_position()      ← Linear extrapolation from velocity
├── compute_conflict_probability()  ← Multi-factor risk scoring
├── get_decision()          ← Threshold-based HOLD/CAUTION/ALLOW
├── AccidentDetector        ← Post-accident detection + severity
├── get_crowd_density()     ← Pedestrian count + density level
├── draw_overlay()          ← Full OpenCV visualization + accident alerts
├── MockDetectionGenerator  ← Synthetic demo mode
└── run_yolo()              ← YOLOv8 wrapper with class filtering

safeturn_phase1.py          ← Core-only standalone
safeturn_phase2.py          ← Enhanced tracking (DeepSORT)
dashboard.py                ← Streamlit dashboard
```

---

## 📊 Decision Thresholds

```
0%                     40%                    70%                  100%
├─────────────────────┼──────────────────────┼─────────────────────┤
│    🟢 ALLOW         │    🟡 CAUTION        │    🔴 HOLD          │
│    Safe to turn     │    Warn driver       │    Restrict turn    │
└─────────────────────┴──────────────────────┴─────────────────────┘
```

---

## 👥 Team

Built at **HackSETU 2025** 


