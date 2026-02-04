"""
Traffic Crashes Data Ingestion from Chicago Data Portal
Dataset: Traffic Crashes - Crashes
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert

from ..config import get_settings
from ..database.models import TrafficCrash, Document, init_database

settings = get_settings()


class TrafficCrashIngester:
    """Ingest traffic crash/incident data from Chicago Data Portal"""
    
    BASE_URL = "https://data.cityofchicago.org/resource"
    
    def __init__(self):
        self.dataset_id = settings.traffic_crashes_dataset_id
        self.app_token = settings.chicago_data_portal_app_token
        self.batch_size = settings.ingestion_batch_size
        
    def _build_url(self, limit: int, offset: int, where_clause: Optional[str] = None) -> str:
        """Build Socrata API URL"""
        url = f"{self.BASE_URL}/{self.dataset_id}.json"
        params = [f"$limit={limit}", f"$offset={offset}", "$order=crash_date DESC"]
        
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
        """Fetch traffic crash data from API"""
        
        where_clauses = []
        
        if start_date:
            where_clauses.append(f"crash_date >= '{start_date.strftime('%Y-%m-%d')}'")
        if end_date:
            where_clauses.append(f"crash_date <= '{end_date.strftime('%Y-%m-%d')}'")
            
        where_clause = " AND ".join(where_clauses) if where_clauses else None
        
        all_records = []
        offset = 0
        
        while True:
            url = self._build_url(
                limit=min(limit, self.batch_size),
                offset=offset,
                where_clause=where_clause
            )
            
            logger.info(f"Fetching crash data: offset={offset}")
            
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
                logger.error(f"Failed to fetch crash data: {e}")
                break
        
        logger.info(f"Fetched {len(all_records)} crash records")
        
        if not all_records:
            return pd.DataFrame()
            
        return self._transform_data(all_records)
    
    def _transform_data(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform raw API data to structured format"""
        
        df = pd.DataFrame(records)
        
        # Column mapping
        column_mapping = {
            'crash_record_id': 'crash_record_id',
            'crash_date': 'crash_date',
            'street_name': 'street_name',
            'street_direction': 'street_direction',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'beat_of_occurrence': 'beat_of_occurrence',
            'crash_type': 'crash_type',
            'prim_contributory_cause': 'primary_cause',
            'sec_contributory_cause': 'secondary_cause',
            'weather_condition': 'weather_condition',
            'lighting_condition': 'lighting_condition',
            'roadway_surface_cond': 'road_condition',
            'traffic_control_device': 'traffic_control_device',
            'injuries_total': 'injuries_total',
            'injuries_fatal': 'injuries_fatal',
            'damage': 'damage'
        }
        
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        # Parse date
        if 'crash_date' in df.columns:
            df['crash_date'] = pd.to_datetime(df['crash_date'])
            df['hour'] = df['crash_date'].dt.hour
            df['day_of_week'] = df['crash_date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Convert numeric columns
        numeric_cols = ['latitude', 'longitude', 'injuries_total', 'injuries_fatal']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Generate crash description for RAG
        df['crash_description'] = df.apply(self._generate_description, axis=1)
        
        # Select final columns
        final_cols = [
            'crash_record_id', 'street_name', 'street_direction', 
            'latitude', 'longitude', 'beat_of_occurrence',
            'crash_date', 'crash_type', 'primary_cause', 'secondary_cause',
            'weather_condition', 'lighting_condition', 'road_condition',
            'traffic_control_device', 'injuries_total', 'injuries_fatal',
            'damage', 'crash_description', 'hour', 'day_of_week', 'is_weekend'
        ]
        
        df = df[[c for c in final_cols if c in df.columns]]
        
        return df
    
    def _generate_description(self, row) -> str:
        """Generate human-readable description of crash for RAG"""
        
        parts = []
        
        # Date and location
        if pd.notna(row.get('crash_date')):
            date_str = row['crash_date'].strftime('%B %d, %Y at %I:%M %p')
            parts.append(f"Traffic incident occurred on {date_str}")
        
        if pd.notna(row.get('street_name')):
            location = row['street_name']
            if pd.notna(row.get('street_direction')):
                location = f"{row['street_direction']} {location}"
            parts.append(f"on {location}")
        
        # Crash type
        if pd.notna(row.get('crash_type')):
            parts.append(f"Crash type: {row['crash_type']}")
        
        # Cause
        if pd.notna(row.get('primary_cause')):
            parts.append(f"Primary cause: {row['primary_cause']}")
        
        # Conditions
        conditions = []
        if pd.notna(row.get('weather_condition')):
            conditions.append(f"weather: {row['weather_condition']}")
        if pd.notna(row.get('lighting_condition')):
            conditions.append(f"lighting: {row['lighting_condition']}")
        if pd.notna(row.get('road_condition')):
            conditions.append(f"road surface: {row['road_condition']}")
            
        if conditions:
            parts.append(f"Conditions: {', '.join(conditions)}")
        
        # Impact
        injuries = row.get('injuries_total', 0) or 0
        fatal = row.get('injuries_fatal', 0) or 0
        
        if injuries > 0:
            parts.append(f"Total injuries: {int(injuries)}")
        if fatal > 0:
            parts.append(f"Fatal injuries: {int(fatal)}")
            
        if pd.notna(row.get('damage')):
            parts.append(f"Damage level: {row['damage']}")
        
        return ". ".join(parts) + "."
    
    def save_to_database(self, df: pd.DataFrame, database_url: str) -> int:
        """Save crash data to PostgreSQL"""
        
        if df.empty:
            logger.warning("No crash data to save")
            return 0
            
        engine = create_engine(database_url)
        records = df.to_dict('records')
        
        saved_count = 0
        
        with engine.begin() as conn:
            for i in range(0, len(records), 1000):
                batch = records[i:i+1000]
                
                # Insert crashes
                stmt = insert(TrafficCrash).values(batch)
                stmt = stmt.on_conflict_do_nothing(index_elements=['crash_record_id'])
                conn.execute(stmt)
                saved_count += len(batch)
                
        logger.info(f"Saved {saved_count} traffic crash records")
        return saved_count
    
    def create_documents(self, df: pd.DataFrame, database_url: str) -> int:
        """Create document records for RAG from crash descriptions"""
        
        if df.empty:
            return 0
            
        engine = create_engine(database_url)
        
        documents = []
        for _, row in df.iterrows():
            if pd.notna(row.get('crash_description')) and row['crash_description']:
                doc = {
                    'doc_type': 'incident',
                    'source_id': row.get('crash_record_id'),
                    'title': f"Traffic Incident - {row.get('street_name', 'Unknown Location')}",
                    'content': row['crash_description'],
                    'created_date': row.get('crash_date'),
                    'location': row.get('street_name'),
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                    'is_embedded': False,
                    'chunk_count': 0
                }
                documents.append(doc)
        
        with engine.begin() as conn:
            for i in range(0, len(documents), 1000):
                batch = documents[i:i+1000]
                stmt = insert(Document).values(batch)
                stmt = stmt.on_conflict_do_nothing()
                conn.execute(stmt)
                
        logger.info(f"Created {len(documents)} documents for RAG")
        return len(documents)


def ingest_crash_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    database_url: Optional[str] = None,
    create_docs: bool = True
) -> Dict[str, int]:
    """Main function to ingest traffic crash data"""
    
    if database_url is None:
        database_url = settings.database_url
        
    init_database(database_url)
    
    # Default to last 30 days
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    ingester = TrafficCrashIngester()
    
    logger.info(f"Ingesting crash data from {start_date} to {end_date}")
    
    df = ingester.fetch_data(start_date=start_date, end_date=end_date)
    
    result = {'crashes': 0, 'documents': 0}
    
    if df.empty:
        logger.warning("No crash data fetched")
        return result
        
    result['crashes'] = ingester.save_to_database(df, database_url)
    
    if create_docs:
        result['documents'] = ingester.create_documents(df, database_url)
    
    return result


if __name__ == "__main__":
    start = datetime.now() - timedelta(days=7)
    end = datetime.now()
    
    result = ingest_crash_data(start_date=start, end_date=end)
    print(f"Ingested {result['crashes']} crashes, created {result['documents']} documents")
