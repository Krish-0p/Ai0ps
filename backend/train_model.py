"""
Production model training script
Trains Isolation Forest + Autoencoder ensemble
"""
import pandas as pd
import sys
from pathlib import Path
from preprocessing.log_processor import LogProcessor
from models.anomaly_detector import EnsembleAnomalyDetector
from utils.logger import setup_logger
from config.settings import MODELS_DIR, DATA_DIR

logger = setup_logger('training')


def train_models(csv_path: str = "structured_logs.csv"):
    """Train ensemble anomaly detection model"""
    logger.info("=" * 60)
    logger.info("Starting model training pipeline")
    logger.info("=" * 60)
    
    # 1. Load structured data
    logger.info(f"Loading structured logs from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} log entries")
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        logger.error("Please run log_parser.py first to generate structured_logs.csv")
        sys.exit(1)
    
    if df.empty:
        logger.error("No data to train on")
        sys.exit(1)
    
    # 2. Create windows and feature matrix
    logger.info("Creating time windows and feature matrix...")
    processor = LogProcessor()
    
    # Create windows
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        windowed_df = processor.create_time_windows(df)
    else:
        windowed_df = processor.create_line_windows(df)
    
    if windowed_df.empty:
        logger.error("No windows created from data")
        sys.exit(1)
    
    # Build feature matrix
    feature_matrix, feature_columns = processor.build_feature_matrix(windowed_df)
    
    if feature_matrix.empty:
        logger.error("Feature matrix is empty")
        sys.exit(1)
    
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"Number of event types: {len(feature_columns)}")
    
    # 3. Train ensemble model
    logger.info("Training ensemble model (Isolation Forest + Autoencoder)...")
    detector = EnsembleAnomalyDetector()
    
    try:
        detector.train(feature_matrix, feature_columns)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Save models
    model_path = MODELS_DIR / "ensemble_model.pkl"
    logger.info(f"Saving models to {model_path}")
    detector.save(str(model_path))
    
    # Save feature columns separately for easy loading
    import joblib
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.pkl")
    
    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--csv", default="structured_logs.csv",
                       help="Path to structured logs CSV")
    
    args = parser.parse_args()
    train_models(args.csv)
