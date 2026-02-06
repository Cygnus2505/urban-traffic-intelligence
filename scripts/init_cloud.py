"""
Initialize Cloud Infrastructure
Creates tables in Supabase and collections in Qdrant Cloud.
"""
import os
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from loguru import logger
from dotenv import load_dotenv
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.database.models import init_database
from src.config import get_settings

load_dotenv()

def init_cloud():
    cloud_db_url = os.getenv("CLOUD_DATABASE_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    if not cloud_db_url:
        logger.error("CLOUD_DATABASE_URL not found!")
        return

    logger.info("--- Initializing Supabase Schema ---")
    try:
        init_database(cloud_db_url)
        logger.info("Supabase schema initialized successfully.")
    except Exception as e:
        logger.error(f"Supabase init failed: {e}")

    logger.info("\n--- Initializing Qdrant Cloud ---")
    if not qdrant_url or not qdrant_key:
        logger.error("Qdrant cloud credentials missing!")
        return

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        from qdrant_client.http import models
        collection_name = "traffic_documents"
        
        logger.info(f"Checking collection {collection_name}...")
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )
            logger.info("Qdrant collection created.")
        else:
            logger.info("Qdrant collection already exists.")
    except Exception as e:
        logger.error(f"Qdrant init failed: {e}")

if __name__ == "__main__":
    init_cloud()
