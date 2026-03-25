"""
SafeTurn AI — Phase 3: Streamlit Dashboard
===========================================
HackSETU 2025 | Theme 3: Intelligent Free-Left Turn Management

Real-time dashboard for monitoring the SafeTurn AI system.
Runs alongside the main detection pipeline.

Usage:
    streamlit run dashboard.py

Features:
    ✓ Live conflict probability gauge
    ✓ Decision history timeline
    ✓ Session statistics with charts
    ✓ System architecture explainer
    ✓ Judges-ready presentation mode
"""

import streamlit as st
import time
import math
import random
import json
import collections

# ─── Optional imports ─────────────────────────────────────
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SafeTurn AI — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium dark theme
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Dark premium theme */
    .stApp {
        background-color: #0a0e1a;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f35, #0d1117);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 2.8em;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85em;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .red { color: #ff4444; }
    .yellow { color: #ffcc00; }
    .green { color: #00cc66; }
    .cyan { color: #00d4ff; }
    
    /* Decision banner */
    .decision-hold {
        background: linear-gradient(90deg, #ff0000, #cc0000);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 2em;
        font-weight: 700;
        animation: pulse 1s ease-in-out infinite alternate;
    }
    .decision-caution {
        background: linear-gradient(90deg, #ff9900, #cc7700);
        color: black;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 2em;
        font-weight: 700;
    }
    .decision-allow {
        background: linear-gradient(90deg, #00cc44, #009933);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 2em;
        font-weight: 700;
    }
    
    @keyframes pulse {
        from { opacity: 1; }
        to { opacity: 0.7; }
    }
    
    /* Pipeline diagram */
    .pipeline-step {
        background: #1a1f35;
        border-left: 3px solid #00d4ff;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .pipeline-step-title {
        color: #00d4ff;
        font-weight: 600;
        font-size: 1em;
    }
    .pipeline-step-desc {
        color: #8b95a5;
        font-size: 0.85em;
        margin-top: 4px;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f1729, #1a2744);
        border: 1px solid #2d3548;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .main-subtitle {
        color: #6b7b8d;
        font-size: 1em;
    }
    
    /* Probability bar */
    .prob-bar-container {
        background: #1a1f35;
        border-radius: 10px;
        padding: 3px;
        margin: 10px 0;
    }
    .prob-bar {
        height: 30px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

if 'conflict_history' not in st.session_state:
    st.session_state.conflict_history = collections.deque(maxlen=200)
if 'decision_history' not in st.session_state:
    st.session_state.decision_history = collections.deque(maxlen=50)
if 'stats' not in st.session_state:
    st.session_state.stats = {"holds": 0, "cautions": 0, "allows": 0}
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'running' not in st.session_state:
    st.session_state.running = False


# ═══════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (for demo when no live feed)
# ═══════════════════════════════════════════════════════════════════

def simulate_conflict_tick():
    """
    Simulates a realistic conflict probability signal.
    Creates cyclical patterns that show ALLOW → CAUTION → HOLD → ALLOW
    to demonstrate the system's predictive decision-making.
    """
    t = st.session_state.frame_count
    
    # Base wave: creates periodic conflict scenarios
    base = 0.3 + 0.35 * math.sin(t * 0.05)
    
    # Spike events: occasional high-conflict moments
    spike = 0.0
    if t % 80 > 60:  # Every 80 ticks, a conflict scenario plays out
        phase = (t % 80 - 60) / 20.0
        spike = 0.45 * math.sin(phase * math.pi)
    
    # Noise for realism
    noise = random.gauss(0, 0.03)
    
    prob = max(0.0, min(1.0, base + spike + noise))
    return round(prob, 3)


def get_decision_from_prob(prob):
    """Mirror the same thresholds as the detection pipeline."""
    if prob > 0.70:
        return "HOLD", "🔴"
    elif prob > 0.40:
        return "CAUTION", "🟡"
    else:
        return "ALLOW", "🟢"


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ═══════════════════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ SafeTurn AI</div>
    <div class="main-subtitle">Predictive Free-Left Turn Safety System — HackSETU 2025</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    
    mode = st.radio("Mode", ["🎮 Live Simulation", "📊 Presentation"], index=0)
    
    if mode == "🎮 Live Simulation":
        if st.button("▶️ Start Simulation", use_container_width=True):
            st.session_state.running = True
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.running = False
        if st.button("🔄 Reset Stats", use_container_width=True):
            st.session_state.stats = {"holds": 0, "cautions": 0, "allows": 0}
            st.session_state.conflict_history.clear()
            st.session_state.decision_history.clear()
            st.session_state.frame_count = 0
    
    st.markdown("---")
    st.markdown("## 📐 Thresholds")
    st.markdown("""
    | Level | Threshold |
    |-------|-----------|
    | 🔴 HOLD | > 70% |
    | 🟡 CAUTION | > 40% |
    | 🟢 ALLOW | ≤ 40% |
    """)
    
    st.markdown("---")
    st.markdown("## 🏗️ Tech Stack")
    st.markdown("""
    - YOLOv8 nano (detection)
    - CentroidTracker / DeepSORT
    - Linear trajectory prediction
    - 3-second lookahead window
    - OpenCV visualization
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Three Lines for Judges")
    st.info('"Predictive, not reactive — we act before danger arrives."')
    st.info('"3-second lookahead, 70% threshold, HOLD or ALLOW."')
    st.info('"Existing CCTV, no new hardware, ₹6,000 per junction."')


# ═══════════════════════════════════════════════════════════════════
# PRESENTATION MODE
# ═══════════════════════════════════════════════════════════════════

if mode == "📊 Presentation":
    st.markdown("## 🧠 How SafeTurn AI Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pipeline Architecture")
        
        steps = [
            ("1. INPUT", "CCTV camera feed from free-left junction"),
            ("2. DETECT", "YOLOv8 nano identifies pedestrians & vehicles"),
            ("3. TRACK", "Persistent IDs assigned to each object"),
            ("4. PREDICT", "Trajectory extrapolated 3 seconds ahead"),
            ("5. CONFLICT", "Pairwise collision probability computed"),
            ("6. DECIDE", "HOLD / CAUTION / ALLOW signal fired"),
        ]
        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-step-title">{title}</div>
                <div class="pipeline-step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Key Innovation")
        
        st.markdown("""
        #### 🔮 Predictive vs Reactive
        
        | | SafeTurn AI | Traditional |
        |---|---|---|
        | **Approach** | Predicts 3s ahead | Reacts after event |
        | **Tracking** | Persistent IDs + velocity | Frame-by-frame |
        | **Decision** | Probability-based | Timer-based |
        | **Hardware** | Existing CCTV | New sensors |
        | **Cost** | ₹6,000/junction | ₹50,000+ |
        """)
        
        st.markdown("#### 📊 Conflict Probability Formula")
        st.markdown("""
        ```
        P(conflict) = 0.60 × distance_factor
                    + 0.25 × closing_speed_factor
                    + 0.15 × convergence_factor
        ```
        
        - **Distance**: How close predicted positions are at T+3s
        - **Closing speed**: Are they moving toward each other?
        - **Convergence**: Are their paths intersecting?
        """)
        
        st.markdown("#### 🎯 Decision Thresholds")
        st.progress(0.70, text="🔴 HOLD threshold (70%)")
        st.progress(0.40, text="🟡 CAUTION threshold (40%)")
        
    st.markdown("---")
    st.markdown("### 🏙️ Real-World Impact")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Prediction Window</div>
            <div class="metric-value cyan">3s</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Cost Per Junction</div>
            <div class="metric-value green">₹6,000</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Min FPS on CPU</div>
            <div class="metric-value cyan">8+</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LIVE SIMULATION MODE
# ═══════════════════════════════════════════════════════════════════

if mode == "🎮 Live Simulation":
    
    # Simulate one tick
    if st.session_state.running:
        conflict_prob = simulate_conflict_tick()
        decision, emoji = get_decision_from_prob(conflict_prob)
        
        st.session_state.conflict_history.append(conflict_prob)
        st.session_state.decision_history.append({
            "frame": st.session_state.frame_count,
            "decision": decision,
            "probability": conflict_prob,
            "emoji": emoji,
        })
        
        # Count decision transitions
        prev = st.session_state.decision_history[-2]["decision"] if len(st.session_state.decision_history) > 1 else None
        if decision != prev:
            if decision == "HOLD":
                st.session_state.stats["holds"] += 1
            elif decision == "CAUTION":
                st.session_state.stats["cautions"] += 1
            else:
                st.session_state.stats["allows"] += 1
        
        st.session_state.frame_count += 1
    else:
        conflict_prob = st.session_state.conflict_history[-1] if st.session_state.conflict_history else 0.0
        decision, emoji = get_decision_from_prob(conflict_prob)
    
    # ── Decision Banner ──
    css_class = f"decision-{decision.lower()}"
    st.markdown(f"""
    <div class="{css_class}">
        {emoji} {decision} {emoji}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ── Top metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conflict Probability</div>
            <div class="metric-value {'red' if conflict_prob > 0.7 else 'yellow' if conflict_prob > 0.4 else 'green'}">{conflict_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conflicts Prevented</div>
            <div class="metric-value red">{st.session_state.stats['holds']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cautions Issued</div>
            <div class="metric-value yellow">{st.session_state.stats['cautions']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Turns Allowed</div>
            <div class="metric-value green">{st.session_state.stats['allows']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ── Conflict probability graph ──
    st.markdown("### 📈 Conflict Probability — Live Feed")
    
    if PANDAS_AVAILABLE and len(st.session_state.conflict_history) > 1:
        history = list(st.session_state.conflict_history)
        df = pd.DataFrame({
            "Frame": list(range(len(history))),
            "Conflict Probability": history,
        })
        
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.patch.set_facecolor('#0a0e1a')
            ax.set_facecolor('#0f1729')
            
            # Plot the probability line
            frames = df["Frame"].values
            probs = df["Conflict Probability"].values
            
            ax.plot(frames, probs, color='#00d4ff', linewidth=1.5, alpha=0.9)
            ax.fill_between(frames, probs, alpha=0.15, color='#00d4ff')
            
            # Threshold lines
            ax.axhline(y=0.70, color='#ff4444', linestyle='--', alpha=0.6, label='HOLD (70%)')
            ax.axhline(y=0.40, color='#ffcc00', linestyle='--', alpha=0.6, label='CAUTION (40%)')
            
            # Color zones
            ax.axhspan(0.70, 1.0, alpha=0.05, color='red')
            ax.axhspan(0.40, 0.70, alpha=0.05, color='yellow')
            ax.axhspan(0.0, 0.40, alpha=0.05, color='green')
            
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability", color='#8b95a5', fontsize=10)
            ax.set_xlabel("Frame", color='#8b95a5', fontsize=10)
            ax.tick_params(colors='#8b95a5')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#2d3548')
            ax.spines['left'].set_color('#2d3548')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            # Fallback: Streamlit's built-in chart
            st.line_chart(df.set_index("Frame"))
    else:
        st.info("⏳ Start the simulation to see live conflict probability data.")
    
    # ── Decision History Log ──
    st.markdown("### 📋 Recent Decisions")
    
    if st.session_state.decision_history:
        recent = list(st.session_state.decision_history)[-10:]
        recent.reverse()
        
        if PANDAS_AVAILABLE:
            log_data = []
            for entry in recent:
                log_data.append({
                    "Frame": entry["frame"],
                    "Decision": f"{entry['emoji']} {entry['decision']}",
                    "Probability": f"{entry['probability']:.1%}",
                })
            df_log = pd.DataFrame(log_data)
            st.dataframe(df_log, use_container_width=True, hide_index=True)
        else:
            for entry in recent:
                st.text(f"Frame {entry['frame']}: {entry['emoji']} {entry['decision']} ({entry['probability']:.1%})")
    else:
        st.info("No decisions recorded yet.")
    
    # ── Auto-refresh ──
    if st.session_state.running:
        time.sleep(0.15)  # ~7 updates per second
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #4a5568; font-size: 0.85em;">'
    '🛡️ SafeTurn AI — HackSETU 2025 | '
    '"Predictive, not reactive — we act before danger arrives."'
    '</div>',
    unsafe_allow_html=True,
)
