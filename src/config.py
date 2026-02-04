"""Configuration management for the traffic forecasting system."""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the ML traffic forecasting system."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    VECTOR_DB_PATH = DATA_DIR / "vector_db"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    CHICAGO_DATA_PORTAL_TOKEN = os.getenv("CHICAGO_DATA_PORTAL_TOKEN", "")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Model Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    
    # RAG Configuration
    VECTOR_DB_COLLECTION = "traffic_knowledge"
    TOP_K_RETRIEVAL = 5
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_EXPLANATION_CONTENT_LENGTH = 200  # Character limit for context snippets
    
    # Forecasting Configuration
    FORECAST_HORIZON = 24  # hours
    SEQUENCE_LENGTH = 168  # 1 week of hourly data
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Traffic patterns
    RUSH_HOUR_MORNING = [7, 8, 9]
    RUSH_HOUR_EVENING = [16, 17, 18]
    RUSH_HOURS = [7, 8, 9, 16, 17, 18]
    WEEKDAY_COUNT = 5  # Monday (0) through Friday (4)
    
    # Data sources
    CHICAGO_TRAFFIC_API = "https://data.cityofchicago.org/resource/8v9j-bter.json"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_PATH.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
