"""Tests for data preprocessing module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.preprocessor import TrafficDataPreprocessor
from src.data.collector import ChicagoTrafficCollector


@pytest.fixture
def sample_data():
    """Create sample traffic data for testing."""
    collector = ChicagoTrafficCollector()
    return collector.generate_synthetic_data(num_days=10, locations=2)


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = TrafficDataPreprocessor()
    assert preprocessor is not None
    assert preprocessor.scaler is not None


def test_preprocess(sample_data):
    """Test basic preprocessing."""
    preprocessor = TrafficDataPreprocessor()
    processed = preprocessor.preprocess(sample_data)
    
    assert isinstance(processed, pd.DataFrame)
    assert len(processed) > 0
    
    # Check that new features were added
    assert 'hour_sin' in processed.columns
    assert 'hour_cos' in processed.columns
    assert 'is_rush_hour' in processed.columns
    assert 'is_weekend' in processed.columns


def test_temporal_features(sample_data):
    """Test temporal feature engineering."""
    preprocessor = TrafficDataPreprocessor()
    processed = preprocessor.preprocess(sample_data)
    
    # Check cyclical encoding
    assert 'hour_sin' in processed.columns
    assert 'hour_cos' in processed.columns
    assert 'dow_sin' in processed.columns
    assert 'dow_cos' in processed.columns
    
    # Check value ranges for cyclical features
    assert processed['hour_sin'].min() >= -1
    assert processed['hour_sin'].max() <= 1
    assert processed['hour_cos'].min() >= -1
    assert processed['hour_cos'].max() <= 1


def test_lag_features(sample_data):
    """Test lag feature creation."""
    preprocessor = TrafficDataPreprocessor()
    processed = preprocessor.preprocess(sample_data)
    
    # Check that lag features exist
    assert 'speed_lag_1' in processed.columns
    assert 'speed_lag_24' in processed.columns
    
    # Check that lag features have correct values
    # (first few rows will be NaN)
    assert processed['speed_lag_1'].isna().sum() >= 1


def test_rolling_features(sample_data):
    """Test rolling statistics features."""
    preprocessor = TrafficDataPreprocessor()
    processed = preprocessor.preprocess(sample_data)
    
    # Check that rolling features exist
    assert 'speed_rolling_mean_3' in processed.columns
    assert 'speed_rolling_std_3' in processed.columns
    assert 'speed_rolling_min_3' in processed.columns
    assert 'speed_rolling_max_3' in processed.columns


def test_create_sequences(sample_data):
    """Test sequence creation for time series."""
    preprocessor = TrafficDataPreprocessor()
    processed = preprocessor.preprocess(sample_data)
    
    # Drop rows with NaN values for this test
    processed = processed.dropna()
    
    if len(processed) > 200:  # Only test if we have enough data
        X, y = preprocessor.create_sequences(
            processed,
            sequence_length=24,
            forecast_horizon=6
        )
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X.shape) == 3  # (samples, sequence_length, features)
        assert len(y.shape) == 2  # (samples, forecast_horizon)
        assert X.shape[0] == y.shape[0]  # Same number of samples
        assert X.shape[1] == 24  # Correct sequence length
        assert y.shape[1] == 6  # Correct forecast horizon


def test_normalize_features():
    """Test feature normalization."""
    preprocessor = TrafficDataPreprocessor()
    
    # Create dummy data
    X = np.random.randn(100, 10, 5)
    
    # Fit and transform
    X_normalized = preprocessor.normalize_features(X, fit=True)
    
    assert X_normalized.shape == X.shape
    assert isinstance(X_normalized, np.ndarray)
    
    # Check that scaler was fitted
    assert preprocessor.scaler.mean_ is not None
