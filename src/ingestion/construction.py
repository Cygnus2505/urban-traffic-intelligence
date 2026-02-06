"""
Road Construction Data Ingestion from Chicago Data Portal
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert

from ..config import get_settings
from ..database.models import RoadConstruction, Document, init_database

settings = get_settings()


class RoadConstructionIngester:
    """Ingest road construction/closure data from Chicago Data Portal"""
    
    BASE_URL = "https://data.cityofchicago.org/resource"
    
    def __init__(self):
        self.dataset_id = settings.road_construction_dataset_id
        self.app_token = settings.chicago_data_portal_app_token
        self.batch_size = settings.ingestion_batch_size
        
    def _build_url(self, limit: int, offset: int, where_clause: Optional[str] = None) -> str:
        """Build Socrata API URL"""
        url = f"{self.BASE_URL}/{self.dataset_id}.json"
        params = [f"$limit={limit}", f"$offset={offset}"]
        
        if where_clause:
            params.append(f"$where={where_clause}")
        
        if self.app_token:
            params.append(f"$$app_token={self.app_token}")
        
        # Order by most recent first
        params.append("$order=applicationstartdate DESC")
            
        return url + "?" + "&".join(params)
    
    def fetch_data(
        self, 
        active_only: bool = True,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch road construction data from API"""
        
        where_clauses = []
        
        # Only get records with street location info
        where_clauses.append("streetname IS NOT NULL")
        
        if active_only:
            today = datetime.now().strftime('%Y-%m-%d')
            where_clauses.append(f"applicationexpiredate >= '{today}'")
            
        where_clause = " AND ".join(where_clauses) if where_clauses else None
        
        all_records = []
        offset = 0
        
        while True:
            url = self._build_url(
                limit=min(limit, self.batch_size),
                offset=offset,
                where_clause=where_clause
            )
            
            logger.info(f"Fetching construction data: offset={offset}")
            
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
                logger.error(f"Failed to fetch construction data: {e}")
                break
        
        logger.info(f"Fetched {len(all_records)} construction records")
        
        if not all_records:
            return pd.DataFrame()
            
        return self._transform_data(all_records)
    
    def _transform_data(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform raw API data"""
        
        df = pd.DataFrame(records)
        
        # Column mapping for Transportation Department Permits dataset (pubx-yq2d)
        column_mapping = {
            'applicationnumber': 'permit_id',
            'streetname': 'street_name',
            'streetnumberfrom': 'from_street',
            'streetnumberto': 'to_street',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'worktypedescription': 'work_type',
            'comments': 'work_description',
            'primarycontactlast': 'contractor',
            'applicationstartdate': 'start_date',
            'applicationexpiredate': 'end_date',
            'direction': 'direction',
            'suffix': 'suffix'
        }
        
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        # Parse dates
        date_cols = ['start_date', 'end_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert coordinates
        for col in ['latitude', 'longitude']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Mark active status
        today = datetime.now()
        if 'end_date' in df.columns:
            df['is_active'] = df['end_date'] >= today
        else:
            df['is_active'] = True
        
        # Generate description for RAG
        df['work_description'] = df.apply(self._generate_description, axis=1)
        
        final_cols = [
            'permit_id', 'street_name', 'from_street', 'to_street',
            'latitude', 'longitude', 'work_type', 'work_description',
            'contractor', 'start_date', 'end_date', 'is_active'
        ]
        
        df = df[[c for c in final_cols if c in df.columns]]
        
        return df
    
    def _generate_description(self, row) -> str:
        """Generate description for RAG"""
        
        parts = []
        
        # Location - build full street address
        if pd.notna(row.get('street_name')):
            location = ""
            if pd.notna(row.get('from_street')):
                location += str(row['from_street']) + " "
            if pd.notna(row.get('direction')):
                location += str(row['direction']) + " "
            location += str(row['street_name'])
            if pd.notna(row.get('suffix')):
                location += " " + str(row['suffix'])
            parts.append(f"Road work at {location}")
        
        # Work type
        if pd.notna(row.get('work_type')):
            parts.append(f"Type of work: {row['work_type']}")
        
        # Original description/comments
        if pd.notna(row.get('work_description')) and row.get('work_description'):
            original_desc = str(row['work_description'])
            if original_desc and original_desc != row.get('work_type', ''):
                parts.append(f"Details: {original_desc}")
        
        # Timeline
        if pd.notna(row.get('start_date')) and pd.notna(row.get('end_date')):
            try:
                start = row['start_date'].strftime('%B %d, %Y')
                end = row['end_date'].strftime('%B %d, %Y')
                parts.append(f"Duration: {start} to {end}")
            except:
                pass
        
        # Contractor
        if pd.notna(row.get('contractor')):
            parts.append(f"Contractor: {row['contractor']}")
        
        return ". ".join(parts) + "." if parts else ""
    
    def save_to_database(self, df: pd.DataFrame, database_url: str) -> int:
        """Save construction data to PostgreSQL"""
        
        if df.empty:
            return 0
            
        engine = create_engine(database_url)
        records = df.to_dict('records')
        
        with engine.begin() as conn:
            for i in range(0, len(records), 1000):
                batch = records[i:i+1000]
                stmt = insert(RoadConstruction).values(batch)
                stmt = stmt.on_conflict_do_nothing()
                conn.execute(stmt)
                
        logger.info(f"Saved {len(records)} construction records")
        return len(records)
    
    def create_documents(self, df: pd.DataFrame, database_url: str) -> int:
        """Create document records for RAG"""
        
        if df.empty:
            return 0
            
        engine = create_engine(database_url)
        
        documents = []
        for _, row in df.iterrows():
            if pd.notna(row.get('work_description')) and row['work_description']:
                doc = {
                    'doc_type': 'construction',
                    'source_id': row.get('permit_id'),
                    'title': f"Road Construction - {row.get('street_name', 'Unknown')}",
                    'content': row['work_description'],
                    'created_date': row.get('start_date'),
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
                
        logger.info(f"Created {len(documents)} construction documents")
        return len(documents)


def ingest_construction_data(
    database_url: Optional[str] = None,
    active_only: bool = True,
    create_docs: bool = True
) -> Dict[str, int]:
    """Main function to ingest road construction data"""
    
    if database_url is None:
        database_url = settings.database_url
        
    init_database(database_url)
    
    ingester = RoadConstructionIngester()
    
    logger.info("Ingesting road construction data")
    
    df = ingester.fetch_data(active_only=active_only)
    
    result = {'construction': 0, 'documents': 0}
    
    if df.empty:
        logger.warning("No construction data fetched")
        return result
        
    result['construction'] = ingester.save_to_database(df, database_url)
    
    if create_docs:
        result['documents'] = ingester.create_documents(df, database_url)
    
    return result


if __name__ == "__main__":
    result = ingest_construction_data()
    print(f"Ingested {result['construction']} construction records, created {result['documents']} documents")
