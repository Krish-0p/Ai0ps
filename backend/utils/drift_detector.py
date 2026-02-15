"""
Model drift detection and retraining trigger
"""
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Tuple
from utils.logger import setup_logger
from config.settings import DRIFT_DETECTION

logger = setup_logger('drift_detector')


class DriftDetector:
    """Detects model drift by monitoring prediction distribution changes"""
    
    def __init__(self):
        self.prediction_history = deque(maxlen=DRIFT_DETECTION['window_size'])
        self.baseline_distribution = None
        self.drift_detected = False
    
    def update_baseline(self, predictions: np.ndarray):
        """Set baseline prediction distribution"""
        normal_count = np.sum(predictions == 1)
        anomaly_count = np.sum(predictions == -1)
        total = len(predictions)
        
        self.baseline_distribution = {
            'normal': normal_count / total if total > 0 else 0,
            'anomaly': anomaly_count / total if total > 0 else 0
        }
        logger.info(f"Baseline distribution set: {self.baseline_distribution}")
    
    def update(self, predictions: np.ndarray):
        """Update prediction history and check for drift"""
        if not DRIFT_DETECTION['enabled']:
            return False
        
        # Add current predictions to history
        for pred in predictions:
            self.prediction_history.append(pred)
        
        if len(self.prediction_history) < DRIFT_DETECTION['window_size']:
            return False
        
        if self.baseline_distribution is None:
            self.update_baseline(predictions)
            return False
        
        # Calculate current distribution
        current_normal = sum(1 for p in self.prediction_history if p == 1)
        current_anomaly = sum(1 for p in self.prediction_history if p == -1)
        total = len(self.prediction_history)
        
        current_distribution = {
            'normal': current_normal / total if total > 0 else 0,
            'anomaly': current_anomaly / total if total > 0 else 0
        }
        
        # Calculate drift: absolute difference in distributions
        normal_drift = abs(current_distribution['normal'] - self.baseline_distribution['normal'])
        anomaly_drift = abs(current_distribution['anomaly'] - self.baseline_distribution['anomaly'])
        total_drift = (normal_drift + anomaly_drift) / 2
        
        if total_drift > DRIFT_DETECTION['threshold']:
            self.drift_detected = True
            logger.warning(
                f"Model drift detected! "
                f"Drift: {total_drift:.3f} (threshold: {DRIFT_DETECTION['threshold']})"
            )
            logger.warning(
                f"Baseline: {self.baseline_distribution}, "
                f"Current: {current_distribution}"
            )
            return True
        
        self.drift_detected = False
        return False
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        return self.drift_detected and DRIFT_DETECTION['retrain_trigger']
    
    def reset(self):
        """Reset drift detection state"""
        self.prediction_history.clear()
        self.baseline_distribution = None
        self.drift_detected = False
        logger.info("Drift detector reset")

