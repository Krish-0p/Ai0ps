"""
Risk prediction and root cause analysis
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger('risk_predictor')


class RiskPredictor:
    """Predicts failure risk based on anomaly trends and patterns"""
    
    def __init__(self, history_window: int = 20):
        self.history_window = history_window
        self.score_history: List[float] = []
        self.timestamps: List[datetime] = []
    
    def update_history(self, score: float, timestamp: datetime = None):
        """Update score history"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.score_history.append(score)
        self.timestamps.append(timestamp)
        
        # Keep only recent history
        if len(self.score_history) > self.history_window:
            self.score_history.pop(0)
            self.timestamps.pop(0)
    
    def predict_risk_score(self) -> float:
        """Predict failure risk score (0-100)"""
        if len(self.score_history) < 3:
            return 0.0  # Not enough data
        
        recent_scores = np.array(self.score_history[-10:])
        
        # Factors:
        # 1. Current score (lower = higher risk)
        current_risk = max(0, -recent_scores[-1] * 50) if recent_scores[-1] < 0 else 0
        
        # 2. Trend (deteriorating = higher risk)
        if len(recent_scores) >= 3:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            trend_risk = max(0, -trend * 30) if trend < 0 else 0
        else:
            trend_risk = 0
        
        # 3. Volatility (high variance = higher risk)
        volatility = np.std(recent_scores)
        volatility_risk = min(30, volatility * 20)
        
        # 4. Anomaly frequency (more anomalies = higher risk)
        anomaly_count = sum(1 for s in recent_scores if s < 0)
        frequency_risk = (anomaly_count / len(recent_scores)) * 20
        
        total_risk = min(100, current_risk + trend_risk + volatility_risk + frequency_risk)
        
        return round(total_risk, 1)
    
    def analyze_root_cause(self, feature_matrix: pd.DataFrame, 
                          event_templates: Dict[int, str],
                          top_n: int = 5) -> List[Tuple[str, float]]:
        """Analyze which events are contributing most to anomaly"""
        if feature_matrix.empty:
            return []
        
        # Get most recent window
        latest_window = feature_matrix.iloc[-1]
        
        # Calculate contribution: events with highest counts in anomalous window
        contributions = []
        for event_id, count in latest_window.items():
            if count > 0:
                event_id_int = int(event_id) if isinstance(event_id, str) else event_id
                template = event_templates.get(event_id_int, f"Event_{event_id}")
                contributions.append((template, float(count)))
        
        # Sort by contribution (count)
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        return contributions[:top_n]
    
    def calculate_health_score(self, anomaly_score: float, error_rate: float,
                               log_volume: float, trend: float) -> float:
        """Calculate overall infrastructure health score (0-100)"""
        from config.settings import HEALTH_SCORE_WEIGHTS
        
        # Normalize components to 0-100 scale
        # Anomaly score: negative = bad, convert to 0-100 (lower anomaly = higher health)
        anomaly_health = max(0, min(100, (anomaly_score + 1) * 50))
        
        # Error rate: 0-1, convert to 0-100 (lower error = higher health)
        error_health = max(0, min(100, (1 - error_rate) * 100))
        
        # Log volume: normalize (assume normal range 0-1000 events)
        volume_health = max(0, min(100, 100 - (log_volume / 10)))
        
        # Trend: positive = improving, negative = deteriorating
        trend_health = max(0, min(100, 50 + (trend * 25)))
        
        # Weighted combination
        health_score = (
            HEALTH_SCORE_WEIGHTS['anomaly_score'] * anomaly_health +
            HEALTH_SCORE_WEIGHTS['error_rate'] * error_health +
            HEALTH_SCORE_WEIGHTS['log_volume'] * volume_health +
            HEALTH_SCORE_WEIGHTS['trend'] * trend_health
        )
        
        return round(max(0, min(100, health_score)), 1)

