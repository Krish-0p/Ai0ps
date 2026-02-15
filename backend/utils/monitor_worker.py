import threading
import time
from collections import deque
from pathlib import Path
import numpy as np
import joblib
from config.settings import DASHBOARD_REFRESH_RATE, MODELS_DIR

try:
    from models.anomaly_detector import EnsembleAnomalyDetector
except Exception:
    EnsembleAnomalyDetector = None


class MonitorWorker:
    """Background worker that maintains live anomaly scores and model state.

    - Runs in a daemon thread and updates internal history at regular intervals.
    - Safe to call from Streamlit UI to start/stop or read history.
    """
    def __init__(self, refresh_rate: float = DASHBOARD_REFRESH_RATE):
        self.refresh_rate = refresh_rate
        self._running = False
        self._lock = threading.Lock()
        self._history = deque(maxlen=1000)
        self._logs = deque(maxlen=200)
        self._last_prediction = 1
        self._last_score = 0.0
        self._inject_next = False
        self._wakeup = threading.Event()  # allow immediate wake for injections
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._force_until = 0.0
        self._forced_score = None
        self._load_model()

    def _load_model(self):
        self.model = None
        self.feature_cols = []
        try:
            if EnsembleAnomalyDetector is not None:
                detector = EnsembleAnomalyDetector()
                detector.load(str(MODELS_DIR / "ensemble_model.pkl"))
                self.model = detector
                try:
                    self.feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
                except Exception:
                    self.feature_cols = []
        except Exception:
            # Keep model None to indicate simulation fallback
            self.model = None
            self.feature_cols = []

    def reload_model(self):
        with self._lock:
            self._load_model()

    def start(self):
        with self._lock:
            self._running = True

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self):
        with self._lock:
            return bool(self._running)

    def inject_attack(self):
        """Trigger an injected attack on the next worker cycle, immediately reflect a strong anomaly in the public state, and wake the loop."""
        with self._lock:
            self._inject_next = True
            # Immediately reflect a visible anomaly so UI can respond instantly
            self._last_prediction = -1
            self._last_score = -0.95
            # Keep the immediate anomaly visible for at least refresh_rate seconds
            try:
                self._history.append(self._last_score)
                self._forced_score = self._last_score
                self._logs.appendleft(f"[{time.strftime('%H:%M:%S')}] [IMMEDIATE] Injected DDoS-like spike | Score: {self._last_score:.3f}")
            except Exception:
                pass
            try:
                self._force_until = time.time() + max(0.5, self.refresh_rate)
            except Exception:
                self._force_until = time.time() + 0.5
        # wake the worker so the attack is processed without waiting for the next sleep interval
        try:
            self._wakeup.set()
        except Exception:
            pass

    def _simulate_step(self):
        # Produce a vector and optionally an injected attack
        # For simplicity, simulate a score in [-1, 1] where negative indicates anomaly
        if self._inject_next:
            self._inject_next = False
            score = -0.9 + np.random.normal(0, 0.05)
            pred = -1
            log = "[SIM] Injected DDoS-like spike"
        else:
            score = float(np.random.normal(0.2, 0.25))
            pred = -1 if score < 0 else 1
            log = "[SIM] Normal tick"

        # Clamp score
        score = max(-1.0, min(1.0, score))

        return pred, float(score), log

    def _loop(self):
        while True:
            if self.is_running():
                try:
                    pred, score, log = None, None, None

                    # If a real model is loaded, use it to predict on a simulated vector
                    if self.model and self.feature_cols:
                        vec = np.zeros((1, len(self.feature_cols)), dtype=int)
                        # add some random noise
                        idxs = np.random.choice(max(1, len(self.feature_cols)), min(2, max(1, len(self.feature_cols))))
                        for i in np.atleast_1d(idxs):
                            vec[0, i] = int(np.random.randint(1, 5))

                        if self._inject_next:
                            vec[0, np.random.randint(0, vec.shape[1])] = 2000
                            self._inject_next = False

                        try:
                            preds, scores, breakdown = self.model.predict(vec)
                            pred = int(np.asarray(preds[0]).item())
                            score = float(np.asarray(scores[0]).item())
                            log = "[MODEL] Predicted via model"
                        except Exception:
                            pred, score, log = self._simulate_step()
                    else:
                        pred, score, log = self._simulate_step()

                    with self._lock:
                        self._last_prediction = pred
                        self._last_score = score
                        self._history.append(score)
                        self._logs.appendleft(f"[{time.strftime('%H:%M:%S')}] {log} | Score: {score:.3f}")
                except Exception:
                    # Don't propagate errors from the background thread
                    pass
            # Wait but allow immediate wake on injection
            try:
                self._wakeup.wait(timeout=self.refresh_rate)
                # clear the wake flag so we don't spin instantly again
                self._wakeup.clear()
            except Exception:
                time.sleep(self.refresh_rate)

    def get_history(self):
        with self._lock:
            return list(self._history)

    def get_logs(self, limit=10):
        with self._lock:
            return list(self._logs)[:limit]

    def latest(self):
        with self._lock:
            # If we've recently injected an attack, force a clear anomaly visible to the UI
            try:
                if getattr(self, '_force_until', 0) > time.time():
                    return {'prediction': -1, 'score': float(getattr(self, '_forced_score', -0.95))}
            except Exception:
                pass
            return {
                'prediction': self._last_prediction,
                'score': self._last_score
            }


# Helper accessor for Streamlit apps to get a single shared resource
_worker_singleton = None

def get_monitor_worker():
    global _worker_singleton
    if _worker_singleton is None:
        _worker_singleton = MonitorWorker()
    return _worker_singleton
