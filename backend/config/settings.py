"""
Production Configuration for AIOps Sentinel
Centralized settings for all components
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Log Processing
WINDOW_SIZE = 100  # Lines per time window
WINDOW_OVERLAP = 0  # Overlap between windows (0 = no overlap)
SLIDING_WINDOW_SECONDS = 60  # Time-based windowing (seconds)

# Model Configuration
ISOLATION_FOREST = {
    'n_estimators': 100,
    'contamination': 0.01,
    'random_state': 42,
    'n_jobs': -1
}

AUTOENCODER = {
    'encoding_dim': 32,
    'hidden_layers': [64, 32],
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'validation_split': 0.2
}

ENSEMBLE_WEIGHTS = {
    'isolation_forest': 0.6,
    'autoencoder': 0.4
}

# Alerting Configuration
ALERT_LEVELS = {
    'LOW': {'threshold': -0.3, 'color': '#fbbf24'},
    'MEDIUM': {'threshold': -0.5, 'color': '#f59e0b'},
    'HIGH': {'threshold': -0.7, 'color': '#ef4444'},
    'CRITICAL': {'threshold': -0.9, 'color': '#dc2626'}
}

ALERT_THROTTLE_SECONDS = 300  # Don't alert same issue within 5 minutes
ALERT_SUPPRESSION_WINDOW = 3600  # Suppress similar alerts for 1 hour

# Health Score Configuration
HEALTH_SCORE_WEIGHTS = {
    'anomaly_score': 0.4,
    'error_rate': 0.3,
    'log_volume': 0.2,
    'trend': 0.1
}

# Log Ingestion
INGESTION_SOURCES = {
    'local_file': {
        'enabled': True,
        'paths': ['live_server.log', 'my_logs.log']
    },
    'ssh': {
        'enabled': False,
        'hosts': [],
        'log_paths': [],
        'username': None,
        'key_path': None
    }
}

# Drain3 Configuration
DRAIN3_CONFIG = {
    'profiling_enabled': False,
    'persistence_type': 'file',
    'persistence_path': str(CACHE_DIR / 'drain3_state.bin')
}

# Noise Filtering
NOISE_PATTERNS = [
    r'heartbeat',
    r'health.*check',
    r'cron.*scheduled',
    r'systemd.*started',
    r'kernel.*log',
    r'^\[.*\]$'  # Empty brackets
]

# Sensitive Data Masking
MASK_PATTERNS = {
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'token': r'[A-Za-z0-9]{32,}',
    'username': r'(?i)\b(user|usr|username|login):\s*\w+'
}

# Dashboard Configuration
DASHBOARD_REFRESH_RATE = 2.0  # seconds
MAX_HISTORY_POINTS = 500
MAX_ALERT_HISTORY = 100

# Model Drift Detection
DRIFT_DETECTION = {
    'enabled': True,
    'window_size': 1000,
    'threshold': 0.15,  # 15% change in prediction distribution
    'retrain_trigger': True
}

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / 'aiops_sentinel.log'

