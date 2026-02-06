from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
import pandas as pd
from typing import List, Dict, Any

from ...config import get_settings

router = APIRouter(tags=["Data"])
settings = get_settings()

@router.get("/summary")
async def get_traffic_summary():
    """Get high-level statistics for the dashboard"""
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        # Get counts
        traffic_count = conn.execute(text("SELECT count(*) FROM traffic_congestion")).scalar()
        incident_count = conn.execute(text("SELECT count(*) FROM traffic_crashes")).scalar()
        doc_count = conn.execute(text("SELECT count(*) FROM documents")).scalar()
        
        # Get average congestion
        avg_congestion = conn.execute(text("SELECT AVG(congestion_level) FROM traffic_congestion")).scalar() or 0
        
        # Get recent segments for map
        recent_query = text("""
            SELECT DISTINCT ON (segment_id) 
                segment_id, street, direction, current_speed, expected_speed, 
                congestion_level, start_lat, start_lon, end_lat, end_lon, timestamp
            FROM traffic_congestion
            ORDER BY segment_id, timestamp DESC
            LIMIT 200
        """)
        segments = pd.read_sql(recent_query, conn).to_dict('records')
        
        # Get recent incidents (crashes)
        incidents_query = text("""
            SELECT street_name, crash_date, damage, latitude, longitude
            FROM traffic_crashes
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY crash_date DESC
            LIMIT 50
        """)
        incidents = pd.read_sql(incidents_query, conn).to_dict('records')
        
    return {
        "metrics": {
            "total_records": traffic_count,
            "active_incidents": incident_count,
            "indexed_documents": doc_count,
            "avg_congestion": round(float(avg_congestion), 2)
        },
        "segments": segments,
        "incidents": incidents
    }
@router.get("/segments")
async def get_all_segments():
    """Get list of all unique segments with street names for search"""
    engine = create_engine(settings.database_url)
    query = text("""
        SELECT DISTINCT ON (segment_id) 
            segment_id, street, direction
        FROM traffic_congestion
        ORDER BY segment_id, street
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df.to_dict('records')
