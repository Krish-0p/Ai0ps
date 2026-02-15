"""
Production log ingestion module
Supports local files, SSH, and future cloud integrations
"""
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Iterator, Optional, Dict
from utils.logger import setup_logger
from config.settings import INGESTION_SOURCES, MASK_PATTERNS

logger = setup_logger('ingestion')


class LogIngester:
    """Unified log ingestion interface"""
    
    def __init__(self):
        self.mask_patterns = {k: re.compile(v) for k, v in MASK_PATTERNS.items()}
    
    def mask_sensitive_data(self, line: str) -> str:
        """Mask IPs, emails, tokens, usernames in log lines"""
        masked = line
        for pattern_name, pattern in self.mask_patterns.items():
            if pattern_name == 'ip_address':
                masked = pattern.sub('[IP_MASKED]', masked)
            elif pattern_name == 'email':
                masked = pattern.sub('[EMAIL_MASKED]', masked)
            elif pattern_name == 'token':
                masked = pattern.sub('[TOKEN_MASKED]', masked)
            elif pattern_name == 'username':
                masked = pattern.sub(r'\1: [USER_MASKED]', masked)
        return masked
    
    def normalize_timestamp(self, line: str) -> str:
        """Normalize timestamps to UTC ISO format if present"""
        # Common timestamp patterns
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})',
            r'(\d{10,13})',  # Unix timestamp
        ]
        
        # If no timestamp found, prepend current UTC
        for pattern in timestamp_patterns:
            if re.search(pattern, line):
                return line  # Already has timestamp
        
        # Prepend UTC timestamp
        utc_now = datetime.now(timezone.utc).isoformat()
        return f"{utc_now} {line}"
    
    def ingest_local_file(self, file_path: str, last_position: int = 0) -> tuple[List[str], int]:
        """Read new lines from local file since last position"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Log file not found: {file_path}")
                return [], 0
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                new_position = f.tell()
            
            # Normalize and mask
            processed_lines = []
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                line = self.normalize_timestamp(line)
                line = self.mask_sensitive_data(line)
                processed_lines.append(line)
            
            logger.debug(f"Ingested {len(processed_lines)} lines from {file_path}")
            return processed_lines, new_position
            
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return [], last_position
    
    def ingest_ssh(self, host: str, log_path: str, username: str, 
                   key_path: Optional[str] = None) -> List[str]:
        """Ingest logs from remote server via SSH"""
        try:
            import paramiko
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Use key or password auth
            if key_path:
                ssh.connect(host, username=username, key_filename=key_path)
            else:
                # In production, use environment variables or secrets manager
                password = os.getenv(f'SSH_PASSWORD_{host}', '')
                ssh.connect(host, username=username, password=password)
            
            # Tail last 100 lines (adjust as needed)
            stdin, stdout, stderr = ssh.exec_command(f'tail -n 100 {log_path}')
            lines = stdout.readlines()
            ssh.close()
            
            processed_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    line = self.normalize_timestamp(line)
                    line = self.mask_sensitive_data(line)
                    processed_lines.append(line)
            
            logger.info(f"Ingested {len(processed_lines)} lines from {host}:{log_path}")
            return processed_lines
            
        except ImportError:
            logger.error("paramiko not installed. Install with: pip install paramiko")
            return []
        except Exception as e:
            logger.error(f"SSH ingestion error for {host}:{log_path}: {e}")
            return []
    
    def get_ingestion_sources(self) -> List[Dict]:
        """Get configured ingestion sources"""
        sources = []
        
        if INGESTION_SOURCES['local_file']['enabled']:
            for path in INGESTION_SOURCES['local_file']['paths']:
                if os.path.exists(path):
                    sources.append({
                        'type': 'local_file',
                        'path': path,
                        'last_position': 0
                    })
        
        if INGESTION_SOURCES['ssh']['enabled']:
            for i, host in enumerate(INGESTION_SOURCES['ssh']['hosts']):
                sources.append({
                    'type': 'ssh',
                    'host': host,
                    'log_path': INGESTION_SOURCES['ssh']['log_paths'][i],
                    'username': INGESTION_SOURCES['ssh']['username'],
                    'key_path': INGESTION_SOURCES['ssh'].get('key_path')
                })
        
        return sources

