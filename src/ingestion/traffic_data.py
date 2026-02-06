"""
Traffic Congestion Data Ingestion from Chicago Data Portal
Dataset: Chicago Traffic Tracker - Historical Congestion Estimates
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert

from ..config import get_settings
from ..database.models import TrafficCongestion, init_database
from ..guardrails.validators import TrafficDataValidator
from ..monitoring.metrics import DATA_VALIDATION_ERRORS

settings = get_settings()


class TrafficCongestionIngester:
    """Ingest traffic congestion data from Chicago Data Portal"""
    
    BASE_URL = "https://data.cityofchicago.org/resource"
    
    def __init__(self):
        self.dataset_id = settings.traffic_congestion_dataset_id
        self.app_token = settings.chicago_data_portal_app_token
        self.batch_size = settings.ingestion_batch_size
        
    def _build_url(self, limit: int, offset: int, where_clause: Optional[str] = None) -> str:
        """Build Socrata API URL with pagination"""
        url = f"{self.BASE_URL}/{self.dataset_id}.json"
        params = [f"$limit={limit}", f"$offset={offset}"]
        
        if where_clause:
            params.append(f"$where={where_clause}")
        
        if self.app_token:
            params.append(f"$$app_token={self.app_token}")
            
        return url + "?" + "&".join(params)
    
    def fetch_data(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50000
    ) -> pd.DataFrame:
        """Fetch traffic congestion data from API"""
        
        where_clauses = []
        
        if start_date:
            where_clauses.append(f"time >= '{start_date.isoformat()}'")
        if end_date:
            where_clauses.append(f"time <= '{end_date.isoformat()}'")
            
        where_clause = " AND ".join(where_clauses) if where_clauses else None
        
        all_records = []
        offset = 0
        
        while True:
            url = self._build_url(
                limit=min(limit, self.batch_size),
                offset=offset,
                where_clause=where_clause
            )
            
            logger.info(f"Fetching traffic data: offset={offset}")
            
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                records = response.json()
                
                if not records:
                    break
                    
                all_records.extend(records)
                offset += len(records)
                
                if len(records) < self.batch_size or offset >= limit:
                    break
                    
            except requests.RequestException as e:
                logger.error(f"Failed to fetch traffic data: {e}")
                break
        
        logger.info(f"Fetched {len(all_records)} traffic records")
        
        if not all_records:
            return pd.DataFrame()
            
        return self._transform_data(all_records)
    
    def _transform_data(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform raw API data to structured format"""
        
        df = pd.DataFrame(records)
        
        # Rename columns to match our schema (Chicago Data Portal sxs8-h27x keys)
        column_mapping = {
            'segment_id': 'segment_id',
            'street': 'street',
            'direction': 'direction',
            'from_street': 'from_street',
            'to_street': 'to_street',
            'speed': 'current_speed',  # Historical dataset uses 'speed'
            'start_latitude': 'start_lat',
            'start_longitude': 'start_lon',
            'end_latitude': 'end_lat',
            'end_longitude': 'end_lon',
            'time': 'timestamp'
        }
        
        # Only rename columns that exist
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Convert numeric columns
        numeric_cols = ['current_speed', 'expected_speed', 'start_lat', 'start_lon', 
                       'end_lat', 'end_lon']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate congestion level
        if 'current_speed' in df.columns:
            if 'expected_speed' not in df.columns or df['expected_speed'].isnull().all():
                # If expected_speed is missing, assume a default of 30 mph for normalization
                # or just use current_speed directly for training if logic allows.
                # Let's set expected_speed to 30 as a placeholder if missing.
                df['expected_speed'] = 30.0
            
            df['congestion_level'] = df['current_speed'] / df['expected_speed'].replace(0, 1)
        
        # Select only columns we need
        final_cols = [
            'segment_id', 'street', 'direction', 'from_street', 'to_street',
            'current_speed', 'expected_speed', 'congestion_level',
            'start_lat', 'start_lon', 'end_lat', 'end_lon',
            'timestamp', 'hour', 'day_of_week', 'is_weekend'
        ]
        
        df = df[[c for c in final_cols if c in df.columns]]
        
        # Apply guardrails
        records_before = len(df)
        mask = df.apply(lambda x: TrafficDataValidator.validate_congestion(x.to_dict()), axis=1)
        df = df[mask].copy()
        
        records_after = len(df)
        if records_before > records_after:
            diff = records_before - records_after
            logger.warning(f"Guardrails filtered out {diff} invalid records")
            DATA_VALIDATION_ERRORS.labels(source="traffic_congestion").inc(diff)
            
        return df
    
    def save_to_database(self, df: pd.DataFrame, database_url: str) -> int:
        """Save transformed data to PostgreSQL"""
        
        if df.empty:
            logger.warning("No data to save")
            return 0
            
        engine = create_engine(database_url)
        
        # Convert DataFrame to list of dicts
        records = df.to_dict('records')
        
        # Use upsert to handle duplicates
        with engine.begin() as conn:
            for i in range(0, len(records), 1000):
                batch = records[i:i+1000]
                
                stmt = insert(TrafficCongestion).values(batch)
                stmt = stmt.on_conflict_do_nothing()
                conn.execute(stmt)
                
        logger.info(f"Saved {len(records)} traffic congestion records")
        return len(records)


def ingest_traffic_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    database_url: Optional[str] = None
) -> int:
    """Main function to ingest traffic congestion data"""
    
    if database_url is None:
        database_url = settings.database_url
        
    # Initialize database
    init_database(database_url)
    
    # Default to last 7 days if no dates specified
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()
    
    ingester = TrafficCongestionIngester()
    
    logger.info(f"Ingesting traffic data from {start_date} to {end_date}")
    
    df = ingester.fetch_data(start_date=start_date, end_date=end_date)
    
    if df.empty:
        logger.warning("No traffic data fetched")
        return 0
        
    records_saved = ingester.save_to_database(df, database_url)
    
    return records_saved


if __name__ == "__main__":
    # Test ingestion
    from datetime import datetime, timedelta
    
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()
    
    count = ingest_traffic_data(start_date=start, end_date=end)
    print(f"Ingested {count} records")
