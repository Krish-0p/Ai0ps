"""
Legacy log parser - now uses production preprocessing module
Maintained for backward compatibility
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.log_processor import LogProcessor
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger('log_parser')


def parse_logs(log_file_path: str) -> pd.DataFrame:
    """Parse logs using production preprocessing pipeline"""
    logger.info(f"Parsing logs from: {log_file_path}")
    
    # Read log file
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {log_file_path}")
        return pd.DataFrame()
    
    if not lines:
        logger.warning("No lines found in log file")
        return pd.DataFrame()
    
    # Use production processor
    processor = LogProcessor()
    structured_df = processor.extract_templates(lines)
    
    if structured_df.empty:
        logger.warning("No structured data extracted")
        return pd.DataFrame()
    
    logger.info(f"Successfully parsed {len(structured_df)} log entries")
    return structured_df


if __name__ == "__main__":
    import os
    
    log_filename = "my_logs.log"
    
    if os.path.exists(log_filename):
        print(f"Found file: {log_filename}. Processing...")
        df = parse_logs(log_filename)
        
        if not df.empty:
            print(f"\nSuccess! Processed {len(df)} lines.")
            print("Preview of Structured Data:")
            print(df.head())
            df.to_csv("structured_logs.csv", index=False)
            print("\nSaved to 'structured_logs.csv'")
        else:
            print("No data extracted from logs.")
    else:
        print(f"ERROR: Could not find '{log_filename}' in this folder.")
        print("Please make sure the file exists.")
