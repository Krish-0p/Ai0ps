"""
Production log preprocessing pipeline
Handles Drain3 template mining, noise filtering, and windowing
"""
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from utils.logger import setup_logger
from config.settings import (
    DRAIN3_CONFIG, NOISE_PATTERNS, WINDOW_SIZE, 
    SLIDING_WINDOW_SECONDS, CACHE_DIR
)

logger = setup_logger('preprocessing')


class LogProcessor:
    """Production log preprocessing with Drain3 and windowing"""
    
    def __init__(self):
        self.template_miner = self._init_drain3()
        self.noise_patterns = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]
        self.event_cache = {}  # Cache event IDs for templates
    
    def _init_drain3(self) -> TemplateMiner:
        """Initialize Drain3 template miner with persistence"""
        config = TemplateMinerConfig()
        config.profiling_enabled = DRAIN3_CONFIG['profiling_enabled']
        
        if DRAIN3_CONFIG['persistence_type'] == 'file':
            config.persistence_type = 'file'
            config.persistence_path = DRAIN3_CONFIG['persistence_path']
        
        return TemplateMiner(config=config)
    
    def filter_noise(self, lines: List[str]) -> List[str]:
        """Remove heartbeat, cron, and other noise patterns"""
        filtered = []
        for line in lines:
            is_noise = any(pattern.search(line) for pattern in self.noise_patterns)
            if not is_noise:
                filtered.append(line)
        
        if len(lines) != len(filtered):
            logger.debug(f"Filtered {len(lines) - len(filtered)} noise lines")
        
        return filtered
    
    def extract_templates(self, lines: List[str]) -> pd.DataFrame:
        """Extract log templates using Drain3"""
        structured_data = []
        
        for line in lines:
            if not line.strip():
                continue
            
            try:
                result = self.template_miner.add_log_message(line)
                template_id = result['cluster_id']
                template_str = result['template_mined']
                
                structured_data.append({
                    'Raw_Log': line,
                    'Event_ID': template_id,
                    'Event_Template': template_str,
                    'Timestamp': self._extract_timestamp(line)
                })
            except Exception as e:
                logger.warning(f"Error processing line: {line[:100]}: {e}")
                continue
        
        if structured_data:
            logger.info(f"Extracted {len(structured_data)} structured log events")
            return pd.DataFrame(structured_data)
        else:
            return pd.DataFrame(columns=['Raw_Log', 'Event_ID', 'Event_Template', 'Timestamp'])
    
    def _extract_timestamp(self, line: str) -> str:
        """Extract ISO timestamp from log line"""
        # Look for ISO format timestamp
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
        if iso_match:
            return iso_match.group(1)
        return pd.Timestamp.now().isoformat()
    
    def create_time_windows(self, df: pd.DataFrame, 
                           window_seconds: int = None) -> pd.DataFrame:
        """Create time-based windows from structured logs"""
        if df.empty:
            return pd.DataFrame()
        
        window_seconds = window_seconds or SLIDING_WINDOW_SECONDS
        
        # Convert timestamps
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        
        if df.empty:
            return pd.DataFrame()
        
        # Create time windows
        df = df.sort_values('Timestamp')
        start_time = df['Timestamp'].min()
        df['Window_ID'] = ((df['Timestamp'] - start_time).dt.total_seconds() // window_seconds).astype(int)
        
        return df
    
    def create_line_windows(self, df: pd.DataFrame, 
                           window_size: int = None) -> pd.DataFrame:
        """Create line-based windows (fallback for logs without timestamps)"""
        if df.empty:
            return pd.DataFrame()
        
        window_size = window_size or WINDOW_SIZE
        df['Window_ID'] = df.index // window_size
        return df
    
    def build_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Build feature matrix: windows x event types"""
        if df.empty or 'Window_ID' not in df.columns:
            logger.warning("Cannot build feature matrix: empty or missing Window_ID")
            return pd.DataFrame(), []
        
        # Create crosstab: windows x events
        feature_matrix = pd.crosstab(df['Window_ID'], df['Event_ID'])
        feature_matrix = feature_matrix.fillna(0)
        
        # Ensure all columns are numeric and string-named
        feature_matrix.columns = [str(col) for col in feature_matrix.columns]
        
        feature_columns = list(feature_matrix.columns)
        logger.info(f"Built feature matrix: {len(feature_matrix)} windows x {len(feature_columns)} event types")
        
        return feature_matrix, feature_columns
    
    def sanitize_structured_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize structured DataFrame: ensure Timestamp, Event_ID, and clean rows

        - Fill missing Event_ID with 'unknown'
        - Ensure Timestamp parsed to datetime (or fill with now)
        - Drop rows with empty Raw_Log
        - Remove duplicates
        """
        if df.empty:
            return df

        # Drop rows without Raw_Log
        if 'Raw_Log' in df.columns:
            df = df.dropna(subset=['Raw_Log'])
            df = df[df['Raw_Log'].str.strip() != '']

        # Ensure Event_ID and Event_Template exist
        if 'Event_ID' not in df.columns:
            df['Event_ID'] = 'unknown'
        else:
            df['Event_ID'] = df['Event_ID'].fillna('unknown')

        if 'Event_Template' not in df.columns:
            df['Event_Template'] = 'unknown'
        else:
            df['Event_Template'] = df['Event_Template'].fillna('unknown')

        # Ensure Timestamp exists and is parsed
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.Timestamp.now().isoformat()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Timestamp'] = df['Timestamp'].fillna(pd.Timestamp.now())

        # Remove duplicates
        df = df.drop_duplicates(subset=['Raw_Log', 'Event_ID'])

        return df


    def process_logs(self, lines: List[str], use_time_windows: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Complete preprocessing pipeline"""
        # 1. Filter noise
        filtered_lines = self.filter_noise(lines)
        
        if not filtered_lines:
            logger.warning("No lines remaining after noise filtering")
            return pd.DataFrame(), []
        
        # 2. Extract templates
        structured_df = self.extract_templates(filtered_lines)
        
        if structured_df.empty:
            return pd.DataFrame(), []

        # 2.5 Sanitize structured data
        structured_df = self.sanitize_structured_df(structured_df)
        
        # 3. Create windows
        if use_time_windows and 'Timestamp' in structured_df.columns:
            windowed_df = self.create_time_windows(structured_df)
        else:
            windowed_df = self.create_line_windows(structured_df)
        
        if windowed_df.empty:
            return pd.DataFrame(), []
        
        # 4. Build feature matrix
        feature_matrix, feature_columns = self.build_feature_matrix(windowed_df)
        
        # Ensure numeric and fill any NaNs resulting from processing
        if not feature_matrix.empty:
            feature_matrix = feature_matrix.fillna(0).astype(int)
        
        return feature_matrix, feature_columns

