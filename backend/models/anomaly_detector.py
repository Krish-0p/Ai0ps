"""
Production anomaly detection models
Isolation Forest + Autoencoder ensemble
"""
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
# TensorFlow is optional at import time (some environments may not have it)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    keras = None
    layers = None

from utils.logger import setup_logger
from config.settings import (
    ISOLATION_FOREST, AUTOENCODER, ENSEMBLE_WEIGHTS,
    MODELS_DIR
)

logger = setup_logger('models')


class IsolationForestDetector:
    """Isolation Forest anomaly detector"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def train(self, X: pd.DataFrame, feature_columns: List[str]):
        """Train Isolation Forest model"""
        self.feature_columns = feature_columns
        
        # Ensure X has all required columns
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X_aligned = X[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_aligned)
        
        # Train model
        self.model = IsolationForest(**ISOLATION_FOREST)
        self.model.fit(X_scaled)
        
        logger.info(f"Isolation Forest trained on {len(X)} samples with {len(feature_columns)} features")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies: returns (predictions, scores)"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Align features
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X_aligned = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_aligned)
        
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        return predictions, scores
    
    def save(self, path: str):
        """Save model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        logger.info(f"Isolation Forest saved to {path}")
    
    def load(self, path: str):
        """Load model and scaler"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        logger.info(f"Isolation Forest loaded from {path}")


class AutoencoderDetector:
    """Autoencoder-based anomaly detector"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.threshold = None
    
    def _build_model(self, input_dim: int) -> 'keras.Model':
        """Build autoencoder architecture"""
        encoding_dim = AUTOENCODER['encoding_dim']
        hidden_layers = AUTOENCODER['hidden_layers']
        
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for hidden_dim in hidden_layers:
            encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = encoded
        for hidden_dim in reversed(hidden_layers):
            decoded = layers.Dense(hidden_dim, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=AUTOENCODER['learning_rate']),
            loss='mse'
        )
        
        return autoencoder
    
    def train(self, X: pd.DataFrame, feature_columns: List[str]):
        """Train autoencoder model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot train Autoencoder.")
        self.feature_columns = feature_columns
        
        # Align features
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X_aligned = X[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_aligned)
        
        # Build and train model
        self.model = self._build_model(len(feature_columns))
        
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=AUTOENCODER['epochs'],
            batch_size=AUTOENCODER['batch_size'],
            validation_split=AUTOENCODER['validation_split'],
            verbose=0
        )
        
        # Set threshold based on reconstruction error
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile as threshold
        
        logger.info(f"Autoencoder trained. Threshold: {self.threshold:.4f}")
        return history
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies: returns (predictions, scores)"""
        # If the autoencoder model isn't available, fall back to neutral outputs
        if self.model is None:
            # Return normal predictions and zeroed scores so AE doesn't affect ensemble
            preds = np.ones((len(X),), dtype=int)
            scores = np.zeros((len(X),), dtype=float)
            return preds, scores
        
        # Align features
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X_aligned = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_aligned)
        
        # Get reconstruction errors
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        
        # Convert to anomaly scores (negative = anomaly, positive = normal)
        # Normalize to similar scale as Isolation Forest
        scores = -mse / self.threshold if self.threshold > 0 else -mse
        
        # Predictions: -1 if above threshold, 1 otherwise
        predictions = np.where(mse > self.threshold, -1, 1)
        
        return predictions, scores
    
    def save(self, path: str):
        """Save model, scaler, and threshold"""
        self.model.save(path.replace('.pkl', '_model.h5'))
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold
        }, path)
        logger.info(f"Autoencoder saved to {path}")
    
    def load(self, path: str):
        """Load model, scaler, and threshold"""
        self.model = keras.models.load_model(path.replace('.pkl', '_model.h5'))
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.threshold = data['threshold']
        logger.info(f"Autoencoder loaded from {path}")


class EnsembleAnomalyDetector:
    """Ensemble of Isolation Forest + Autoencoder"""
    
    def __init__(self):
        self.if_detector = IsolationForestDetector()
        self.ae_detector = AutoencoderDetector()
        self.feature_columns = []
        self.weights = ENSEMBLE_WEIGHTS
    
    def train(self, X: pd.DataFrame, feature_columns: List[str]):
        """Train both models"""
        self.feature_columns = feature_columns
        
        logger.info("Training Isolation Forest...")
        self.if_detector.train(X, feature_columns)
        
        logger.info("Training Autoencoder...")
        self.ae_detector.train(X, feature_columns)
        
        logger.info("Ensemble training complete")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Ensemble prediction with detailed breakdown"""
        # Get predictions from both models
        if_pred, if_scores = self.if_detector.predict(X)
        ae_pred, ae_scores = ae_scores = self.ae_detector.predict(X)
        
        # Normalize scores to [-1, 1] range for fair weighting
        if_scores_norm = np.tanh(if_scores)
        ae_scores_norm = np.tanh(ae_scores)
        
        # Weighted ensemble score
        ensemble_scores = (
            self.weights['isolation_forest'] * if_scores_norm +
            self.weights['autoencoder'] * ae_scores_norm
        )
        
        # Ensemble prediction: -1 if score < 0, 1 otherwise
        ensemble_predictions = np.where(ensemble_scores < 0, -1, 1)
        
        # Detailed breakdown
        breakdown = {
            'isolation_forest': {
                'prediction': if_pred,
                'score': if_scores,
                'weight': self.weights['isolation_forest']
            },
            'autoencoder': {
                'prediction': ae_pred,
                'score': ae_scores,
                'weight': self.weights['autoencoder']
            },
            'ensemble': {
                'prediction': ensemble_predictions,
                'score': ensemble_scores
            }
        }
        
        return ensemble_predictions, ensemble_scores, breakdown
    
    def save(self, base_path: str):
        """Save both models"""
        self.if_detector.save(base_path.replace('.pkl', '_if.pkl'))
        self.ae_detector.save(base_path.replace('.pkl', '_ae.pkl'))
        joblib.dump({
            'feature_columns': self.feature_columns,
            'weights': self.weights
        }, base_path)
        logger.info(f"Ensemble model saved to {base_path}")
    
    def load(self, base_path: str):
        """Load both models"""
        # Load Isolation Forest (required)
        self.if_detector.load(base_path.replace('.pkl', '_if.pkl'))

        # Load Autoencoder (optional)
        try:
            self.ae_detector.load(base_path.replace('.pkl', '_ae.pkl'))
        except Exception as e:
            logger.warning(f"Autoencoder failed to load: {e}. Continuing with Isolation Forest only.")
            # Disable autoencoder contribution if not available
            self.ae_detector.model = None
            self.weights['autoencoder'] = 0.0

        # Load metadata
        data = joblib.load(base_path)
        self.feature_columns = data.get('feature_columns', [])
        self.weights = data.get('weights', self.weights)
        logger.info(f"Ensemble model loaded from {base_path}")

