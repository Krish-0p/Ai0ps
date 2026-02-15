import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from config.settings import ALERT_LEVELS, DATA_DIR, DASHBOARD_REFRESH_RATE
from ingestion.external_ingest import list_provenance_files

LOG_STRUCTURED = Path("structured_logs.csv")


def compute_risk_level(score: float) -> str:
    # ALERT_LEVELS thresholds are negative (lower = worse); find highest level that applies
    for level, v in sorted(ALERT_LEVELS.items(), key=lambda x: x[1]['threshold']):
        if score <= v['threshold']:
            return level
    return 'LOW'


def simple_forecast(history: list, n_steps: int = 10) -> np.ndarray:
    """Very simple linear forecast using polyfit on recent history"""
    if len(history) < 3:
        return np.array([history[-1]] * n_steps) if history else np.zeros(n_steps)
    y = np.array(history)
    x = np.arange(len(y))
    # Fit a linear model
    coeffs = np.polyfit(x, y, 1)
    next_x = np.arange(len(y), len(y) + n_steps)
    return np.polyval(coeffs, next_x)


def render_stats():
    st.set_page_config(page_title="Server Health Stats", layout="wide")

    # Header with back button
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("← Back to Monitor"):
            st.session_state['page'] = 'main'
            # Safe rerun for Streamlit versions that may not expose experimental_rerun
            rerun_fn = getattr(st, 'experimental_rerun', None)
            if callable(rerun_fn):
                rerun_fn()
            else:
                st.stop()

    st.title("📊 Server Health & Risk Dashboard")

    # Get live data from the shared MonitorWorker when running; otherwise fall back to session_state history
    try:
        from utils.monitor_worker import get_monitor_worker
        worker = get_monitor_worker()
    except Exception:
        worker = None

    if st.session_state.get('is_running') and worker is not None:
        history = worker.get_history()
        latest = worker.latest()
        history_arr = np.array(history)
        latest_score = float(latest.get('score', 0.0))
    else:
        history = st.session_state.get('history', [])
        history_arr = np.array(history)
        latest_score = float(history_arr[-1]) if len(history_arr) > 0 else 0.0

    # Basic metrics
    avg_score = float(np.mean(history_arr)) if len(history_arr) > 0 else 0.0
    anomaly_count = int(np.sum(history_arr < 0)) if len(history_arr) > 0 else 0
    anomaly_rate = (anomaly_count / len(history_arr)) if len(history_arr) > 0 else 0.0

    risk_level = compute_risk_level(latest_score)

    # Top KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk Level", risk_level)
    k2.metric("Latest Anomaly Score", f"{latest_score:.3f}")
    k3.metric("Avg Anomaly Score", f"{avg_score:.3f}")
    k4.metric("Anomaly Rate", f"{anomaly_rate*100:.1f}%")

    # Time series + forecast
    st.subheader("Anomaly Score Timeline")
    col_ts, col_hist = st.columns([3, 1])
    with col_ts:
        if len(history_arr) > 0:
            n_forecast = st.slider("Forecast horizon (steps)", 1, 30, 10)
            forecast = simple_forecast(history_arr.tolist(), n_forecast)

            timeline_df = pd.DataFrame({
                'score': history_arr.tolist() + forecast.tolist(),
                'type': ['history'] * len(history_arr) + ['forecast'] * len(forecast)
            })
            # Plot: two area/line charts joined
            st.line_chart(timeline_df['score'])
        else:
            st.info("No history data yet. Run the monitor to collect scores.")

    with col_hist:
        # Histogram of scores
        if len(history_arr) > 0:
            st.subheader("Score Distribution")
            st.bar_chart(pd.Series(history_arr).value_counts(bins=10).sort_index())

    # Rolling anomaly rate
    st.subheader("Anomaly Rate (Rolling)")
    if len(history_arr) > 1:
        window = st.slider("Rolling window (points)", 5, 100, 20)
        rolling = pd.Series(history_arr).lt(0).astype(int).rolling(window=window, min_periods=1).mean()
        st.line_chart(rolling)

    # Alert breakdown + thresholds
    st.subheader("Alert Thresholds & Current Position")
    alert_levels = ALERT_LEVELS
    thresholds = sorted([(k, v['threshold']) for k, v in alert_levels.items()], key=lambda x: x[1])
    t_df = pd.DataFrame(thresholds, columns=['Level', 'Threshold'])
    st.table(t_df)
    st.write(f"Current score: {latest_score:.3f} → Level: **{risk_level}**")

    # Recent structured logs summary
    st.subheader("Recent Structured Log Summary")
    if LOG_STRUCTURED.exists():
        try:
            s_df = pd.read_csv(LOG_STRUCTURED)
            st.write(f"Total structured rows: {len(s_df)}")
            # Show recent templates frequency
            top_events = s_df['Event_ID'].value_counts().head(10)
            st.bar_chart(top_events)
        except Exception as e:
            st.warning(f"Could not read {LOG_STRUCTURED}: {e}")
    else:
        st.info("No structured logs available yet (processed external logs will append to structured_logs.csv).")

    # Provenance listing
    st.subheader("Recent Ingests (Provenance)")
    history_files = list_provenance_files(limit=10)
    if history_files:
        for item in history_files:
            name = item['name']
            rows = item.get('rows', 'n/a')
            meta_info = item.get('meta') or {}
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(f"{name} — {rows} rows")
                if meta_info:
                    st.write(f"Source: {meta_info.get('source','n/a')} | Uploader: {meta_info.get('uploader','n/a')}")
            with col2:
                if st.button(f"Preview {name}", key=f"stats_preview_{name}"):
                    try:
                        df = pd.read_csv(item['path'])
                        with st.expander(f"Preview: {name}"):
                            st.dataframe(df.head(50))
                    except Exception as e:
                        st.warning(f"Could not preview file {name}: {e}")
            with col3:
                try:
                    with open(item['path'], 'rb') as f:
                        data = f.read()
                    st.download_button("DL", data, file_name=name, key=f"stats_dl_{name}")
                except Exception:
                    st.write("—")
    else:
        st.info("No provenance files yet.")

    # Additional suggestions & actions
    st.subheader("Actions & Tips")
    st.write("- Use the ingestion page to upload external logs and save provenance for audits.")
    st.write("- Manual Retrain: use the button below to retrain models on the current `structured_logs.csv` (manual-only).")

    # Manual retrain button (asynchronous)
    if 'retrain_jobs' not in st.session_state:
        st.session_state['retrain_jobs'] = {}

    def _start_retrain_job():
        import threading, uuid, subprocess, sys
        job_id = uuid.uuid4().hex[:8]
        log_path = Path(DATA_DIR) / 'models' / f'train_log_{job_id}.log'
        job = {
            'id': job_id,
            'status': 'running',
            'log_path': str(log_path),
            'started_at': datetime.utcnow().isoformat() + 'Z',
            'finished_at': None,
            'returncode': None,
            'reloaded': False
        }
        st.session_state['retrain_jobs'][job_id] = job

        def _run():
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'w', encoding='utf-8') as lf:
                    proc = subprocess.Popen([sys.executable, 'train_model.py', '--csv', 'structured_logs.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        lf.write(line)
                        lf.flush()
                    proc.wait()
                    job['returncode'] = proc.returncode
                    job['finished_at'] = datetime.utcnow().isoformat() + 'Z'
                    job['status'] = 'finished' if proc.returncode == 0 else 'error'
            except Exception as e:
                job['status'] = 'error'
                job['finished_at'] = datetime.utcnow().isoformat() + 'Z'
                with open(log_path, 'a', encoding='utf-8') as lf:
                    lf.write(f"\nERROR: {e}\n")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return job_id

    if st.button("🔁 Retrain models now (manual)"):
        job_id = _start_retrain_job()
        st.success(f"Retrain started (job id: {job_id}). Tail the log below to monitor progress.")

    # Display retrain jobs and allow log tailing / reload
    st.subheader("Retrain Jobs")
    jobs = st.session_state.get('retrain_jobs', {})
    if jobs:
        for jid, j in sorted(jobs.items(), key=lambda x: x[1]['started_at'], reverse=True):
            cols = st.columns([3, 2, 2, 1])
            with cols[0]:
                st.write(f"Job {j['id']} — {j['status']} — started: {j['started_at']}")
            with cols[1]:
                if j.get('finished_at'):
                    st.write(f"Finished: {j['finished_at']} (code: {j.get('returncode')})")
            with cols[2]:
                if st.button(f"Tail log {j['id']}", key=f"tail_{j['id']}"):
                    lp = Path(j['log_path'])
                    if lp.exists():
                        try:
                            # Show last 300 lines
                            with open(lp, 'r', encoding='utf-8', errors='replace') as lf:
                                lines = lf.readlines()[-300:]
                            with st.expander(f"Log: {lp.name}"):
                                st.text(''.join(lines[-200:]))
                        except Exception as e:
                            st.warning(f"Could not read log: {e}")
                    else:
                        st.info("Log file not yet created; try again in a few seconds.")
            with cols[3]:
                if j.get('status') == 'finished' and j.get('returncode') == 0 and not j.get('reloaded'):
                    if st.button(f"Reload models {j['id']}", key=f"reload_{j['id']}"):
                        st.session_state['reload_model'] = True
                        j['reloaded'] = True
                        st.success("Model reload scheduled (reload will occur on next app run).")
    else:
        st.info("No retrain jobs yet.")

    st.caption(f"Last updated: {datetime.utcnow().isoformat()} UTC")

    # Auto-refresh the stats page when the Monitor is running
    if st.session_state.get('is_running'):
        st.info("🔴 Live Mode: Monitor is running — auto-refreshing stats")
        # Pull fresh data from the MonitorWorker and trigger a rerun to update charts
        if worker is not None:
            _ = worker.get_history()  # ensure worker has updated state
        time.sleep(DASHBOARD_REFRESH_RATE)
        rerun_fn = getattr(st, 'experimental_rerun', None)
        if callable(rerun_fn):
            rerun_fn()
        else:
            st.stop()
        # Respect the dashboard refresh rate configured globally
        try:
            time.sleep(DASHBOARD_REFRESH_RATE)
        except Exception:
            # Fallback to a short sleep if redirection fails
            time.sleep(1)
        rerun_fn = getattr(st, 'experimental_rerun', None)
        if callable(rerun_fn):
            rerun_fn()
        else:
            # Safe fallback to stop so the app can be resumed by user action
            st.stop()
