"""
Prediction Service
Loads trained model and generates forecasts.
"""
import pandas as pd
import xgboost as xgb
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

from ..features.feature_builder import FeatureBuilder

class TrafficPredictor:
    """
    Service for traffic congestion forecasting.
    Uses trained XGBoost model.
    """
    
    def __init__(self, model_path: str = "src/models/traffic_model.json"):
        self.model_path = model_path
        self.model = None
        self.feature_builder = FeatureBuilder()
        self.load_model()
        
    def load_model(self):
        """Load XGBoost model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = xgb.XGBRegressor()
                self.model.load_model(self.model_path)
                logger.info(f"Loaded traffic model from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Prediction will return mock data.")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict(self, segment_id: str, timestamp: datetime, weather_forecast: Dict[str, Any] = None) -> float:
        """
        Predict congestion level for a specific time and location.
        
        Args:
            segment_id: Unique ID of the road segment
            timestamp: Future time to predict for
            weather_forecast: Optional dict with 'temperature', 'is_raining', etc.
            
        Returns:
            Predicted congestion ratio (0.0 to 1.0+)
        """
        # Create single-row dataframe for feature building
        data = {
            'timestamp': timestamp,
            'segment_id': segment_id,
            # Mock weather if not provided
            'temperature': weather_forecast.get('temperature') if weather_forecast else None,
            'is_raining': weather_forecast.get('is_raining') if weather_forecast else None
        }
        
        # Build features
        X, _ = self.feature_builder.create_features([data], training=False)
        
        # Predict
        if self.model:
            try:
                prediction = self.model.predict(X)[0]
                return max(0.0, float(prediction)) # Ensure non-negative
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return 0.0
        else:
            # Fallback mock logic if no model trained yet
            hour = timestamp.hour
            # Rush hour logic (8-9 AM, 5-6 PM)
            if 7 <= hour <= 9 or 16 <= hour <= 18:
                return 0.85
            return 0.3
