"""
Feature Engineering Pipeline
Transforms raw traffic data into features for ML models.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

class FeatureBuilder:
    """
    Builds features from raw traffic data.
    """
    
    def __init__(self):
        # Define feature columns
        self.feature_columns = [
            'hour_sin', 'hour_cos',
            'day_of_week', 'is_weekend', 
            'temperature', 'is_raining', 'is_snowing',
            'segment_id'
        ]
        
    def create_features(self, data: List[Dict[str, Any]], training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create X (features) and y (target) from data.
        
        Args:
            data: List of dictionaries containing raw data
            training: If True, returns y (target). If False, y is None.
            
        Returns:
            X (DataFrame), y (Series or None)
        """
        df = pd.DataFrame(data)
        
        if df.empty:
            return pd.DataFrame(columns=self.feature_columns), None
            
        # 1. Temporal Features (Cyclical)
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
        # Hour sin/cos
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Weekend
        if 'is_weekend' not in df.columns:
             df['is_weekend'] = df['day_of_week'] >= 5
        df['is_weekend'] = df['is_weekend'].astype(int)
        
        # 2. Weather Features (Simulation/Mock if missing)
        # Ideally this comes from the database or external API
        if 'temperature' not in df.columns or df['temperature'].isna().all():
            # Mock: Cooler in morning/evening
            df['temperature'] = 65 - 5 * np.cos(2 * np.pi * (df['hour'] - 14) / 24)
            # Add some noise
            df['temperature'] += np.random.normal(0, 2, size=len(df))
            
        if 'is_raining' not in df.columns or df['is_raining'].isna().all():
            # Mock: Random 10% chance
            df['is_raining'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
            
        if 'is_snowing' not in df.columns or df['is_snowing'].isna().all():
            df['is_snowing'] = 0
            
        # 3. Select Features and Ensure Numeric
        X = df[self.feature_columns].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # 4. Target
        y = None
        if training:
            if 'congestion_level' in df.columns:
                y = df['congestion_level']
            elif 'current_speed' in df.columns and 'expected_speed' in df.columns:
                # Calculate manually if missing
                y = df['current_speed'] / df['expected_speed'].replace(0, 1)
            else:
                raise ValueError("Target column 'congestion_level' missing for training")
                
        # 5. Clean NaNs/Infs (CRITICAL for XGBoost)
        mask = X.notna().all(axis=1)
        if y is not None:
            # Ensure y is numeric
            y = pd.to_numeric(y, errors='coerce')
            mask = mask & y.notna() & np.isfinite(y)
            y = y[mask]
        
        X = X[mask]
        
        return X, y
