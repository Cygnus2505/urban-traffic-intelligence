"""
Database models for Urban Traffic Intelligence Platform
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, 
    ForeignKey, Index, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


class TrafficCongestion(Base):
    """Historical traffic congestion data by segment"""
    __tablename__ = "traffic_congestion"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(String(50), nullable=False, index=True)
    street = Column(String(255))
    direction = Column(String(50))
    from_street = Column(String(255))
    to_street = Column(String(255))
    
    # Traffic metrics
    current_speed = Column(Float)
    expected_speed = Column(Float)
    congestion_level = Column(Float)  # Ratio: current/expected
    
    # Location
    start_lat = Column(Float)
    start_lon = Column(Float)
    end_lat = Column(Float)
    end_lon = Column(Float)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False, index=True)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_segment_timestamp', 'segment_id', 'timestamp'),
    )


class TrafficCrash(Base):
    """Traffic crash/incident data"""
    __tablename__ = "traffic_crashes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crash_record_id = Column(String(50), unique=True, index=True)
    
    # Location
    street_name = Column(String(255))
    street_direction = Column(String(50))
    latitude = Column(Float)
    longitude = Column(Float)
    beat_of_occurrence = Column(String(20))
    
    # Crash details
    crash_date = Column(DateTime, nullable=False, index=True)
    crash_type = Column(String(100))
    primary_cause = Column(String(255))
    secondary_cause = Column(String(255))
    
    # Conditions
    weather_condition = Column(String(100))
    lighting_condition = Column(String(100))
    road_condition = Column(String(100))
    traffic_control_device = Column(String(100))
    
    # Impact
    injuries_total = Column(Integer, default=0)
    injuries_fatal = Column(Integer, default=0)
    damage = Column(String(100))
    
    # Description (for RAG)
    crash_description = Column(Text)
    
    # Time features
    hour = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RoadConstruction(Base):
    """Road construction and closure data"""
    __tablename__ = "road_construction"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    permit_id = Column(String(50), index=True)
    
    # Location
    street_name = Column(String(255))
    from_street = Column(String(255))
    to_street = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Details
    work_type = Column(String(255))
    work_description = Column(Text)
    contractor = Column(String(255))
    
    # Timeline
    start_date = Column(DateTime, index=True)
    end_date = Column(DateTime, index=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class WeatherData(Base):
    """Weather data for Chicago"""
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Weather metrics
    temperature = Column(Float)  # Fahrenheit
    feels_like = Column(Float)
    humidity = Column(Integer)
    pressure = Column(Integer)
    visibility = Column(Integer)  # meters
    wind_speed = Column(Float)
    wind_direction = Column(Integer)
    
    # Conditions
    weather_main = Column(String(50))  # Rain, Snow, Clear, etc.
    weather_description = Column(String(255))
    clouds_percentage = Column(Integer)
    
    # Precipitation
    rain_1h = Column(Float, default=0)
    snow_1h = Column(Float, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Documents for RAG (incident reports, bulletins, etc.)"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    doc_type = Column(String(50), index=True)  # incident, construction, bulletin
    source_id = Column(String(100))  # Reference to original record
    
    title = Column(String(500))
    content = Column(Text, nullable=False)
    
    # Metadata
    created_date = Column(DateTime)
    location = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Processing status
    is_embedded = Column(Boolean, default=False)
    chunk_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionLog(Base):
    """Log of predictions for monitoring"""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Request info
    request_id = Column(String(50), index=True)
    endpoint = Column(String(100))
    
    # Prediction details
    segment_id = Column(String(50))
    forecast_timestamp = Column(DateTime)
    predicted_value = Column(Float)
    actual_value = Column(Float)  # Filled in later
    
    # Model info
    model_version = Column(String(100))
    
    # Metrics
    latency_ms = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RAGQueryLog(Base):
    """Log of RAG queries for monitoring"""
    __tablename__ = "rag_query_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Request
    request_id = Column(String(50), index=True)
    query = Column(Text)
    
    # Response
    answer = Column(Text)
    sources_count = Column(Integer)
    
    # Quality metrics
    retrieval_score = Column(Float)
    has_citations = Column(Boolean)
    abstained = Column(Boolean, default=False)
    
    # Performance
    latency_ms = Column(Float)
    tokens_used = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)


def init_database(database_url: str):
    """Initialize database and create all tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(database_url: str):
    """Get database session"""
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()
