"""
Production AIOps Dashboard
Real-time monitoring with advanced features
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.anomaly_detector import EnsembleAnomalyDetector
from ingestion.log_ingester import LogIngester
from preprocessing.log_processor import LogProcessor
from utils.alerting import AlertManager
from utils.risk_predictor import RiskPredictor
from utils.drift_detector import DriftDetector
from utils.logger import setup_logger
from config.settings import (
    MODELS_DIR, DASHBOARD_REFRESH_RATE, MAX_HISTORY_POINTS,
    MAX_ALERT_HISTORY, ALERT_LEVELS
)

logger = setup_logger('dashboard')

# Background monitor worker (shared singleton)
from utils.monitor_worker import get_monitor_worker
worker = get_monitor_worker()

# Page configuration
st.set_page_config(
    page_title="AIOps Sentinel - Production",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .terminal {
        background-color: #0E1117;
        color: #00FF41;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #00FF41;
        max-height: 400px;
        overflow-y: auto;
    }
    section[data-testid="stMetric"] {
        background-color: #111827;
        color: #e5e7eb;
        padding: 12px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        border: 1px solid #1f2937;
    }
    section[data-testid="stMetric"] > div {
        color: #e5e7eb;
    }
    section[data-testid="stMetric"] label {
        color: #9ca3af;
        font-weight: 500;
    }
    section[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f9fafb;
        font-weight: 700;
        font-size: 1.6rem;
    }
    .health-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    .health-good { color: #10b981; }
    .health-warning { color: #f59e0b; }
    .health-critical { color: #ef4444; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model_path = MODELS_DIR / "ensemble_model.pkl"
        if not model_path.exists():
            return None, None, None
        
        detector = EnsembleAnomalyDetector()
        detector.load(str(model_path))
        
        feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
        
        # Load event templates if available
        event_templates = {}
        templates_path = MODELS_DIR / "event_templates.pkl"
        if templates_path.exists():
            event_templates = joblib.load(templates_path)
        
        return detector, feature_cols, event_templates
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None, None


# Initialize session state
if 'alert_manager' not in st.session_state:
    st.session_state['alert_manager'] = AlertManager()
if 'risk_predictor' not in st.session_state:
    st.session_state['risk_predictor'] = RiskPredictor()
if 'drift_detector' not in st.session_state:
    st.session_state['drift_detector'] = DriftDetector()
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'server_data' not in st.session_state:
    st.session_state['server_data'] = {}
if 'ingester' not in st.session_state:
    st.session_state['ingester'] = LogIngester()
if 'processor' not in st.session_state:
    st.session_state['processor'] = LogProcessor()

# Load models
detector, feature_cols, event_templates = load_models()

# Sidebar
with st.sidebar:
    st.header("🎮 Control Panel")
    
    # Persist monitor state across pages
    if 'is_running' not in st.session_state:
        st.session_state['is_running'] = False
    st.session_state['is_running'] = st.toggle("🟢 Run Monitor", value=st.session_state.get('is_running', False))
    is_running = st.session_state['is_running']
    # Ensure the worker follows the toggle immediately
    if is_running:
        worker.start()
    else:
        worker.stop()
    
    st.divider()
    
    refresh_rate = st.slider("Refresh Rate (sec)", 0.5, 5.0, DASHBOARD_REFRESH_RATE)
    worker.refresh_rate = refresh_rate
    
    # Add a quick inject button in the dashboard sidebar too for convenience
    if st.button("⚠️ Inject Attack (DDoS)", key="inject_sidebar"):
        worker.inject_attack()
        try:
            st.toast("Injected attack from Stats sidebar", icon="🔥")
        except Exception:
            st.success("Injected attack from Stats sidebar")
        rerun_fn = getattr(st, 'experimental_rerun', None)
        if callable(rerun_fn):
            rerun_fn()
        else:
            rerun_fn2 = getattr(st, 'rerun', None)
            if callable(rerun_fn2):
                rerun_fn2()
            else:
                st.stop()

    st.divider()
    
    st.subheader("Server Selection")
    selected_server = st.selectbox(
        "Monitor Server",
        options=["local", "server-1", "server-2"],
        index=0
    )
    
    st.divider()
    
    st.subheader("Alert Settings")
    show_suppressed = st.checkbox("Show Suppressed Alerts", value=False)
    alert_level_filter = st.selectbox(
        "Filter by Level",
        options=["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        index=0
    )
    
    if not detector:
        st.error("⚠️ Models not loaded. Run train_model.py first.")

# Main UI
st.title("🛡️ AIOps Sentinel - Production Dashboard")
st.caption("Enterprise-grade Infrastructure Anomaly Detection & Monitoring")

if not detector:
    st.error("🚨 Models not found! Please run 'train_model.py' first.")
    st.stop()

# Top Metrics Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

if is_running:
    # Use the background worker's live outputs and skip the inline ingestion loop (background worker handles live updates)
    latest = worker.latest()
    history = worker.get_history()

    # Propagate into session_state for compatibility across pages
    st.session_state['history'] = history

    score = latest['score']
    prediction = latest['prediction']

    # Update drift detector
    try:
        drift_detected = st.session_state['drift_detector'].update([prediction])
    except Exception:
        drift_detected = False

    # Root cause placeholder (requires structured processing)
    root_cause_events = []

    # Generate alert
    alert = st.session_state['alert_manager'].generate_alert(score, selected_server, root_cause_events)

    # Update risk predictor
    st.session_state['risk_predictor'].update_history(score)
    risk_score = st.session_state['risk_predictor'].predict_risk_score()

    # Calculate simplified health score
    error_rate = 0
    log_volume = len(history)
    trend = (np.mean(st.session_state['risk_predictor'].score_history[-5:]) - np.mean(st.session_state['risk_predictor'].score_history[-10:-5])) if len(st.session_state['risk_predictor'].score_history) >= 10 else 0
    health_score = st.session_state['risk_predictor'].calculate_health_score(score, error_rate, log_volume, trend)

    # Update timestamped history used by charts
    timestamp = datetime.now()
    entry = {
        'timestamp': timestamp,
        'score': score,
        'prediction': prediction,
        'risk_score': risk_score,
        'health_score': health_score
    }
    if 'history_ts' not in st.session_state:
        st.session_state['history_ts'] = []
    st.session_state['history_ts'].append(entry)
    if len(st.session_state['history_ts']) > MAX_HISTORY_POINTS:
        st.session_state['history_ts'] = st.session_state['history_ts'][-MAX_HISTORY_POINTS:]

    # Store server data
    st.session_state['server_data'][selected_server] = {
        'score': score,
        'prediction': prediction,
        'health_score': health_score,
        'risk_score': risk_score,
        'timestamp': timestamp
    }

    # Render Metrics
    kpi1.metric("Server Status", "Online", delta_color="normal")
    kpi2.metric("AI Diagnosis", "CRITICAL" if prediction == -1 else "Healthy", "Anomaly" if prediction == -1 else "Stable", delta_color="inverse" if prediction == -1 else "normal")
    kpi3.metric("Anomaly Score", f"{score:.3f}")
    kpi4.metric("Health Score", f"{health_score:.1f}")
    kpi5.metric("Risk Score", f"{risk_score:.1f}%")

    # Alert Banner
    if alert and not alert.suppressed:
        alert_color = ALERT_LEVELS[alert.level]['color']
        st.markdown(
            f'<div style="background-color: {alert_color}20; border-left: 5px solid {alert_color}; padding: 15px; border-radius: 5px; margin: 10px 0;'>
            f'<h3>🚨 {alert.level} ALERT</h3>'
            f'<p>{alert.message}</p>'
            f'<small>Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</small>'
            f'</div>',
            unsafe_allow_html=True
        )

    if drift_detected:
        st.warning("⚠️ Model drift detected! Consider retraining the model.")

    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Monitoring", "🔍 Root Cause", "📈 Analytics", "🚨 Alerts"])

    with tab1:
        st.subheader("Live Anomaly Score Stream")
        if st.session_state.get('history_ts'):
            chart_df = pd.DataFrame(st.session_state['history_ts'])
            chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
            chart_df = chart_df.set_index('timestamp')
            st.area_chart(chart_df[['score']], color="#ff4b4b" if prediction == -1 else "#00FF41", height=300)
        else:
            st.info("Waiting for live data...")

        # Health & Risk
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Infrastructure Health Score")
            health_class = "health-good" if health_score >= 70 else "health-warning" if health_score >= 50 else "health-critical"
            st.markdown(f'<div class="health-score {health_class}">{health_score:.1f}</div>', unsafe_allow_html=True)
        with col2:
            st.subheader("Failure Risk Prediction")
            risk_class = "health-good" if risk_score < 30 else "health-warning" if risk_score < 60 else "health-critical"
            st.markdown(f'<div class="health-score {risk_class}">{risk_score:.1f}%</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Root Cause Analysis")
        st.info("Live root-cause analysis is not available for simulated data. Upload structured logs to populate root-cause responses.")

    with tab3:
        st.subheader("Historical Analytics")
        if st.session_state.get('history_ts'):
            analytics_df = pd.DataFrame(st.session_state['history_ts'])
            analytics_df['timestamp'] = pd.to_datetime(analytics_df['timestamp'])
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(analytics_df.set_index('timestamp')[['health_score']], height=250)
            with col2:
                st.line_chart(analytics_df.set_index('timestamp')[['risk_score']], height=250)

    with tab3:
        st.subheader("Alert History")
        alerts = st.session_state['alert_manager'].get_recent_alerts(
            limit=MAX_ALERT_HISTORY,
            level_filter=alert_level_filter if alert_level_filter != "ALL" else None
        )
        if alerts:
            alert_df = pd.DataFrame([a.to_dict() for a in alerts])
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.info("No alerts in history")

    # Auto-refresh while running
    time.sleep(refresh_rate)
    rerun_fn = getattr(st, 'experimental_rerun', None)
    if callable(rerun_fn):
        rerun_fn()
    else:
        st.rerun()

    # Legacy inline ingestion removed — MonitorWorker handles live data ingestion in background
    all_logs = []
    
    if False:  # legacy ingestion block (no-op - handled by MonitorWorker)
        # Process logs
        feature_matrix, new_feature_cols = st.session_state['processor'].process_logs(
            all_logs, 
            use_time_windows=True
        )
        
        if not feature_matrix.empty:
            # Align features
            for col in feature_cols:
                if col not in feature_matrix.columns:
                    feature_matrix[col] = 0
            
            # Get latest window
            latest_window = feature_matrix.iloc[-1:][feature_cols]
            
            # Predict
            predictions, scores, breakdown = detector.predict(latest_window)
            prediction = predictions[0]
            score = scores[0]
            
            # Update drift detector
            drift_detected = st.session_state['drift_detector'].update(predictions)
            
            # Root cause analysis
            root_cause = st.session_state['risk_predictor'].analyze_root_cause(
                feature_matrix, 
                event_templates
            )
            root_cause_events = [event for event, _ in root_cause[:3]]
            
            # Generate alert
            alert = st.session_state['alert_manager'].generate_alert(
                score, 
                selected_server,
                root_cause_events
            )
            
            # Update risk predictor
            st.session_state['risk_predictor'].update_history(score)
            risk_score = st.session_state['risk_predictor'].predict_risk_score()
            
            # Calculate health score
            error_rate = len([l for l in all_logs if 'ERROR' in l or 'FATAL' in l]) / len(all_logs) if all_logs else 0
            log_volume = len(all_logs)
            trend = np.mean(st.session_state['risk_predictor'].score_history[-5:]) - np.mean(st.session_state['risk_predictor'].score_history[-10:-5]) if len(st.session_state['risk_predictor'].score_history) >= 10 else 0
            
            health_score = st.session_state['risk_predictor'].calculate_health_score(
                score, error_rate, log_volume, trend
            )
            
            # Update history
            timestamp = datetime.now()
            st.session_state['history'].append({
                'timestamp': timestamp,
                'score': score,
                'prediction': prediction,
                'risk_score': risk_score,
                'health_score': health_score
            })
            
            if len(st.session_state['history']) > MAX_HISTORY_POINTS:
                st.session_state['history'] = st.session_state['history'][-MAX_HISTORY_POINTS:]
            
            # Store server data
            st.session_state['server_data'][selected_server] = {
                'score': score,
                'prediction': prediction,
                'health_score': health_score,
                'risk_score': risk_score,
                'timestamp': timestamp
            }
            
            # Render Metrics
            kpi1.metric("Server Status", "Online", delta_color="normal")
            
            if prediction == -1:
                kpi2.metric("AI Diagnosis", "CRITICAL", "Anomaly", delta_color="inverse")
            else:
                kpi2.metric("AI Diagnosis", "Healthy", "Stable", delta_color="normal")
            
            kpi3.metric("Anomaly Score", f"{score:.3f}")
            
            # Health Score with color
            health_color = "normal"
            if health_score < 50:
                health_color = "inverse"
            elif health_score < 70:
                health_color = "off"
            
            kpi4.metric("Health Score", f"{health_score:.1f}", delta_color=health_color)
            kpi5.metric("Risk Score", f"{risk_score:.1f}%")
            
            # Alert Banner
            if alert and not alert.suppressed:
                alert_color = ALERT_LEVELS[alert.level]['color']
                st.markdown(
                    f'<div style="background-color: {alert_color}20; border-left: 5px solid {alert_color}; '
                    f'padding: 15px; border-radius: 5px; margin: 10px 0;">'
                    f'<h3>🚨 {alert.level} ALERT</h3>'
                    f'<p>{alert.message}</p>'
                    f'<small>Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Drift Warning
            if drift_detected:
                st.warning("⚠️ Model drift detected! Consider retraining the model.")
            
            # Main Content Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Monitoring", "🔍 Root Cause", "📈 Analytics", "🚨 Alerts"])
            
            with tab1:
                # Anomaly Score Chart
                st.subheader("Live Anomaly Score Stream")
                if st.session_state['history']:
                    chart_df = pd.DataFrame(st.session_state['history'])
                    chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
                    chart_df = chart_df.set_index('timestamp')
                    
                    chart_color = "#ff4b4b" if prediction == -1 else "#00FF41"
                    st.area_chart(chart_df[['score']], color=chart_color, height=300)
                
                # Health Score Gauge
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Infrastructure Health Score")
                    health_class = "health-good" if health_score >= 70 else "health-warning" if health_score >= 50 else "health-critical"
                    st.markdown(
                        f'<div class="health-score {health_class}">{health_score:.1f}</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.subheader("Failure Risk Prediction")
                    risk_class = "health-good" if risk_score < 30 else "health-warning" if risk_score < 60 else "health-critical"
                    st.markdown(
                        f'<div class="health-score {risk_class}">{risk_score:.1f}%</div>',
                        unsafe_allow_html=True
                    )
            
            with tab2:
                st.subheader("Root Cause Analysis")
                if root_cause:
                    st.write("**Top Contributing Events:**")
                    for i, (event, contribution) in enumerate(root_cause[:5], 1):
                        st.write(f"{i}. **{event}** (count: {int(contribution)})")
                else:
                    st.info("No significant root causes identified")
                
                # Model Breakdown
                st.subheader("Model Ensemble Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Isolation Forest", 
                             "Anomaly" if breakdown['isolation_forest']['prediction'][0] == -1 else "Normal",
                             f"{breakdown['isolation_forest']['score'][0]:.3f}")
                with col2:
                    st.metric("Autoencoder", 
                             "Anomaly" if breakdown['autoencoder']['prediction'][0] == -1 else "Normal",
                             f"{breakdown['autoencoder']['score'][0]:.3f}")
            
            with tab3:
                st.subheader("Historical Analytics")
                if st.session_state['history']:
                    analytics_df = pd.DataFrame(st.session_state['history'])
                    analytics_df['timestamp'] = pd.to_datetime(analytics_df['timestamp'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.line_chart(analytics_df.set_index('timestamp')[['health_score']], height=250)
                    with col2:
                        st.line_chart(analytics_df.set_index('timestamp')[['risk_score']], height=250)
            
            with tab3:
                st.subheader("Alert History")
                alerts = st.session_state['alert_manager'].get_recent_alerts(
                    limit=MAX_ALERT_HISTORY,
                    level_filter=alert_level_filter if alert_level_filter != "ALL" else None
                )
                
                if alerts:
                    alert_df = pd.DataFrame([a.to_dict() for a in alerts])
                    st.dataframe(alert_df, use_container_width=True)
                else:
                    st.info("No alerts in history")
                
                # Alert Stats
                stats = st.session_state['alert_manager'].get_alert_stats()
                st.metric("Total Alerts", stats['total'])
                st.metric("Last 24h", stats['recent_24h'])
            
            # Auto-refresh
            time.sleep(refresh_rate)
            st.rerun()

    # Legacy waiting branch removed — MonitorWorker handles live state independently
else:
    st.info("⏸️ Monitor is PAUSED. Toggle 'Run Monitor' in the sidebar to start.")
    
    # Show static charts if data exists
    if st.session_state['history']:
        st.subheader("Historical Data")
        chart_df = pd.DataFrame(st.session_state['history'])
        chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
        st.area_chart(chart_df.set_index('timestamp')[['score']], height=300)

