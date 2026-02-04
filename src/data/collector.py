"""Data collection module for Chicago traffic data."""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from pathlib import Path

from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChicagoTrafficCollector:
    """Collector for Chicago traffic data from the City of Chicago Data Portal."""
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize the traffic data collector.
        
        Args:
            api_token: Optional API token for Chicago Data Portal
        """
        self.api_token = api_token or Config.CHICAGO_DATA_PORTAL_TOKEN
        self.base_url = Config.CHICAGO_TRAFFIC_API
        
    def fetch_traffic_data(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch traffic data from Chicago Data Portal.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame containing traffic data
        """
        logger.info(f"Fetching traffic data from {start_date} to {end_date}")
        
        params = {
            "$limit": limit,
            "$order": "_last_updt DESC"
        }
        
        if self.api_token:
            params["$$app_token"] = self.api_token
        
        if start_date:
            params["$where"] = f"_last_updt >= '{start_date.isoformat()}'"
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning("No data returned from API")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching traffic data: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_data(
        self, 
        num_days: int = 30,
        locations: int = 10
    ) -> pd.DataFrame:
        """Generate synthetic traffic data for testing purposes.
        
        Args:
            num_days: Number of days to generate data for
            locations: Number of traffic locations
            
        Returns:
            DataFrame with synthetic traffic data
        """
        logger.info(f"Generating synthetic data for {num_days} days and {locations} locations")
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=num_days),
            end=datetime.now(),
            freq='H'
        )
        
        data = []
        for location_id in range(1, locations + 1):
            for date in dates:
                # Simulate traffic patterns
                hour = date.hour
                day_of_week = date.weekday()
                
                # Base traffic (higher during day)
                base_traffic = 50 + 30 * np.sin((hour - 6) * np.pi / 12)
                
                # Weekday vs weekend
                weekday_factor = 1.3 if day_of_week < 5 else 0.7
                
                # Rush hour peaks
                rush_hour_factor = 1.0
                if hour in Config.RUSH_HOURS:
                    rush_hour_factor = 1.5
                
                # Add some randomness
                noise = np.random.normal(0, 10)
                
                speed = max(5, base_traffic * weekday_factor * rush_hour_factor + noise)
                volume = int(max(10, 100 + 50 * (100 - speed) / 100 + np.random.normal(0, 20)))
                
                data.append({
                    'segment_id': location_id,
                    'timestamp': date,
                    'speed': round(speed, 2),
                    'traffic_volume': volume,
                    'street': f'Street_{location_id}',
                    'direction': 'NB' if location_id % 2 == 0 else 'SB',
                    'hour': hour,
                    'day_of_week': day_of_week
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic records")
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: Path):
        """Save traffic data to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load traffic data from CSV file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            DataFrame with traffic data
        """
        if filepath.exists():
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
