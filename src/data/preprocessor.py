"""Data preprocessing and feature engineering module."""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficDataPreprocessor:
    """Preprocessor for traffic data with feature engineering."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'speed'
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw traffic data.
        
        Args:
            df: Raw traffic DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing")
        df = df.copy()
        
        # Convert timestamp if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Add time-based features
        df = self._add_temporal_features(df)
        
        # Add lag features
        df = self._add_lag_features(df)
        
        # Add rolling statistics
        df = self._add_rolling_features(df)
        
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with handled missing values
        """
        # Forward fill for time series data
        if 'timestamp' in df.columns:
            df = df.ffill()
        
        # Fill remaining with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from timestamp.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with additional temporal features
        """
        if 'timestamp' not in df.columns:
            return df
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Rush hour indicator
        df['is_rush_hour'] = df['hour'].isin(Config.RUSH_HOURS).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 24, 168]) -> pd.DataFrame:
        """Add lag features for time series prediction.
        
        Args:
            df: DataFrame with time series data
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        if 'speed' not in df.columns:
            return df
        
        for lag in lags:
            df[f'speed_lag_{lag}'] = df['speed'].shift(lag)
            
        if 'traffic_volume' in df.columns:
            for lag in lags:
                df[f'volume_lag_{lag}'] = df['traffic_volume'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Add rolling statistics features.
        
        Args:
            df: DataFrame with time series data
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling features
        """
        if 'speed' not in df.columns:
            return df
        
        for window in windows:
            df[f'speed_rolling_mean_{window}'] = df['speed'].rolling(window=window, min_periods=1).mean()
            df[f'speed_rolling_std_{window}'] = df['speed'].rolling(window=window, min_periods=1).std()
            df[f'speed_rolling_min_{window}'] = df['speed'].rolling(window=window, min_periods=1).min()
            df[f'speed_rolling_max_{window}'] = df['speed'].rolling(window=window, min_periods=1).max()
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 168,
        forecast_horizon: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting.
        
        Args:
            df: Preprocessed DataFrame
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'segment_id', 'street', 'direction']]
        data = df[feature_cols].values
        
        X, y = [], []
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:i + sequence_length])
            # Predict speed for next forecast_horizon steps
            if 'speed' in df.columns:
                speed_idx = df.columns.get_loc('speed')
                y.append(data[i + sequence_length:i + sequence_length + forecast_horizon, speed_idx])
        
        return np.array(X), np.array(y)
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using StandardScaler.
        
        Args:
            X: Input features
            fit: Whether to fit the scaler
            
        Returns:
            Normalized features
        """
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            X_normalized = self.scaler.transform(X_reshaped)
        
        return X_normalized.reshape(original_shape)
