"""
Production alerting system with multi-level alerts, throttling, and suppression
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from utils.logger import setup_logger
from config.settings import (
    ALERT_LEVELS, ALERT_THROTTLE_SECONDS, ALERT_SUPPRESSION_WINDOW
)

logger = setup_logger('alerting')


@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    level: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    score: float
    server_id: str
    root_cause: Optional[List[str]] = None
    suppressed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            'score': self.score,
            'server_id': self.server_id,
            'root_cause': self.root_cause or [],
            'suppressed': self.suppressed
        }


class AlertManager:
    """Manages alert generation, throttling, and suppression"""
    
    def __init__(self):
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[str, datetime] = {}  # server_id -> last alert time
        self.suppressed_alerts: Dict[str, datetime] = {}  # alert_key -> suppression_end
    
    def determine_alert_level(self, score: float) -> str:
        """Determine alert level from anomaly score"""
        # Score is negative for anomalies, more negative = worse
        if score <= ALERT_LEVELS['CRITICAL']['threshold']:
            return 'CRITICAL'
        elif score <= ALERT_LEVELS['HIGH']['threshold']:
            return 'HIGH'
        elif score <= ALERT_LEVELS['MEDIUM']['threshold']:
            return 'MEDIUM'
        elif score <= ALERT_LEVELS['LOW']['threshold']:
            return 'LOW'
        return 'NORMAL'
    
    def should_throttle(self, server_id: str) -> bool:
        """Check if alerts should be throttled for this server"""
        if server_id not in self.last_alert_time:
            return False
        
        time_since_last = datetime.now() - self.last_alert_time[server_id]
        return time_since_last.total_seconds() < ALERT_THROTTLE_SECONDS
    
    def should_suppress(self, alert_key: str) -> bool:
        """Check if this alert pattern should be suppressed"""
        if alert_key not in self.suppressed_alerts:
            return False
        
        suppression_end = self.suppressed_alerts[alert_key]
        if datetime.now() > suppression_end:
            # Suppression expired
            del self.suppressed_alerts[alert_key]
            return False
        
        return True
    
    def create_alert_key(self, server_id: str, level: str, root_cause: List[str]) -> str:
        """Create unique key for alert suppression"""
        cause_hash = hash(tuple(sorted(root_cause))) if root_cause else 0
        return f"{server_id}:{level}:{cause_hash}"
    
    def generate_alert(self, score: float, server_id: str, 
                      root_cause: Optional[List[str]] = None) -> Optional[Alert]:
        """Generate alert if conditions are met"""
        level = self.determine_alert_level(score)
        
        if level == 'NORMAL':
            return None
        
        # Check throttling
        if self.should_throttle(server_id):
            logger.debug(f"Alert throttled for {server_id}")
            return None
        
        # Check suppression
        alert_key = self.create_alert_key(server_id, level, root_cause or [])
        if self.should_suppress(alert_key):
            logger.debug(f"Alert suppressed: {alert_key}")
            alert = Alert(
                timestamp=datetime.now(),
                level=level,
                message=f"Anomaly detected on {server_id}",
                score=score,
                server_id=server_id,
                root_cause=root_cause,
                suppressed=True
            )
            self.alert_history.append(alert)
            return alert
        
        # Create alert
        message = self._create_alert_message(level, server_id, score, root_cause)
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            message=message,
            score=score,
            server_id=server_id,
            root_cause=root_cause,
            suppressed=False
        )
        
        # Update tracking
        self.last_alert_time[server_id] = datetime.now()
        self.suppressed_alerts[alert_key] = datetime.now() + timedelta(
            seconds=ALERT_SUPPRESSION_WINDOW
        )
        
        self.alert_history.append(alert)
        
        # Keep history limited
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        logger.warning(f"Alert generated: {level} on {server_id} (score: {score:.3f})")
        return alert
    
    def _create_alert_message(self, level: str, server_id: str, 
                              score: float, root_cause: Optional[List[str]]) -> str:
        """Create human-readable alert message"""
        base_msg = f"[{level}] Anomaly detected on {server_id} (score: {score:.3f})"
        
        if root_cause:
            top_causes = ", ".join(root_cause[:3])
            base_msg += f". Top events: {top_causes}"
        
        return base_msg
    
    def get_recent_alerts(self, limit: int = 50, 
                         level_filter: Optional[str] = None) -> List[Alert]:
        """Get recent alerts, optionally filtered by level"""
        alerts = self.alert_history[-limit:]
        
        if level_filter:
            alerts = [a for a in alerts if a.level == level_filter]
        
        return alerts
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        if not self.alert_history:
            return {
                'total': 0,
                'by_level': {},
                'recent_24h': 0
            }
        
        now = datetime.now()
        recent_24h = [a for a in self.alert_history 
                     if (now - a.timestamp).total_seconds() < 86400]
        
        by_level = {}
        for alert in self.alert_history:
            by_level[alert.level] = by_level.get(alert.level, 0) + 1
        
        return {
            'total': len(self.alert_history),
            'by_level': by_level,
            'recent_24h': len(recent_24h)
        }

