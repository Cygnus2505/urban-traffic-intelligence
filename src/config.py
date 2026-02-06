"""
Configuration settings for Urban Traffic Intelligence Platform
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Chicago Data Portal
    chicago_data_portal_app_token: Optional[str] = None
    
    # Chicago Data Portal Dataset IDs
    traffic_congestion_dataset_id: str = "sxs8-h27x"  # Historical Congestion
    traffic_crashes_dataset_id: str = "85ca-t3if"     # Traffic Crashes
    road_construction_dataset_id: str = "pubx-yq2d"   # Transportation Department Permits
    
    # Weather API
    openweather_api_key: Optional[str] = None
    chicago_lat: float = 41.8781
    chicago_lon: float = -87.6298
    
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/traffic_db"
    
    # Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "traffic_documents"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "traffic_forecasting"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Ingestion Settings
    ingestion_batch_size: int = 10000
    max_records_per_request: int = 50000
    
    # Model Settings
    forecast_horizon_hours: int = 24
    retrain_threshold_mae: float = 5.0
    
    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    min_retrieval_score: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
