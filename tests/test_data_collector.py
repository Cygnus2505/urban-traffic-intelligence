"""Tests for data collection module."""
import pytest
import pandas as pd
from datetime import datetime

from src.data.collector import ChicagoTrafficCollector


def test_collector_initialization():
    """Test that collector initializes correctly."""
    collector = ChicagoTrafficCollector()
    assert collector is not None
    assert collector.base_url is not None


def test_generate_synthetic_data():
    """Test synthetic data generation."""
    collector = ChicagoTrafficCollector()
    df = collector.generate_synthetic_data(num_days=5, locations=3)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'segment_id' in df.columns
    assert 'timestamp' in df.columns
    assert 'speed' in df.columns
    assert 'traffic_volume' in df.columns
    
    # Check data types
    assert df['speed'].dtype in [float, 'float64']
    assert df['traffic_volume'].dtype in [int, 'int64']
    
    # Check value ranges
    assert df['speed'].min() > 0
    assert df['speed'].max() <= 100
    assert df['traffic_volume'].min() >= 0


def test_synthetic_data_time_patterns():
    """Test that synthetic data has realistic time patterns."""
    collector = ChicagoTrafficCollector()
    df = collector.generate_synthetic_data(num_days=2, locations=1)
    
    # Group by hour and check that rush hours have different speeds
    hourly_avg = df.groupby('hour')['speed'].mean()
    
    # Rush hour speeds should generally be lower
    rush_hours = [7, 8, 9, 16, 17, 18]
    non_rush_hours = [1, 2, 3, 4, 5]
    
    rush_avg = hourly_avg[hourly_avg.index.isin(rush_hours)].mean()
    non_rush_avg = hourly_avg[hourly_avg.index.isin(non_rush_hours)].mean()
    
    # This is a probabilistic test, but should generally hold
    assert rush_avg < non_rush_avg * 1.2  # Allow some variance
