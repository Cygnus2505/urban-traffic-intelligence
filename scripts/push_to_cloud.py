"""
Data Migration Script
Migrates local PostgreSQL and Qdrant data to Supabase and Qdrant Cloud.
"""
import os
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

# Load local .env
load_dotenv()

def migrate_database():
    """Migrate PostgreSQL data to Supabase"""
    local_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/traffic_db")
    cloud_url = os.getenv("CLOUD_DATABASE_URL")
    
    if not cloud_url:
        logger.error("CLOUD_DATABASE_URL not found in .env")
        return

    logger.info("Connecting to local and cloud databases...")
    local_engine = create_engine(local_url)
    cloud_engine = create_engine(cloud_url)
    
    tables = [
        'traffic_congestion', 
        'traffic_crashes', 
        'road_construction', 
        'weather_data', 
        'documents', 
        'prediction_logs', 
        'rag_query_logs'
    ]
    
    for table in tables:
        try:
            logger.info(f"Checking table: {table}")
            # Use chunks for potentially large tables
            rows_migrated = 0
            for chunk in pd.read_sql_table(table, local_engine, chunksize=5000):
                if not chunk.empty:
                    chunk.to_sql(table, cloud_engine, if_exists='append', index=False)
                    rows_migrated += len(chunk)
            
            if rows_migrated > 0:
                logger.info(f"Successfully migrated {rows_migrated} rows for {table}")
            else:
                logger.warning(f"Table {table} had no data to migrate.")
        except Exception as e:
            logger.error(f"Failed to migrate {table}: {e}")

def migrate_vectors():
    """Migrate Qdrant vectors to Qdrant Cloud"""
    local_client = QdrantClient(host="localhost", port=6333)
    
    cloud_url = os.getenv("QDRANT_URL")
    cloud_key = os.getenv("QDRANT_API_KEY")
    collection_name = "traffic_documents"
    
    if not cloud_url or not cloud_key:
        logger.error("QDRANT_URL or QDRANT_API_KEY not found in .env")
        return

    logger.info("Connecting to Qdrant Cloud...")
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_key)
    
    try:
        # 1. Get points from local
        logger.info(f"Fetching points from local collection: {collection_name}")
        points, _ = local_client.scroll(collection_name=collection_name, limit=10000, with_vectors=True)
        
        if not points:
            logger.warning("No vectors found locally to migrate.")
            return

        # 2. Re-create collection on cloud (simplification)
        logger.info(f"Creating collection {collection_name} on cloud...")
        from qdrant_client.http import models
        cloud_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        
        # 3. Upsert to cloud
        logger.info(f"Upserting {len(points)} points to cloud...")
        cloud_client.upsert(collection_name=collection_name, points=points)
        logger.info("Vector migration complete!")
        
    except Exception as e:
        logger.error(f"Vector migration failed: {e}")

if __name__ == "__main__":
    import sys
    
    print("\n[MIGRATION] Urban Traffic Intelligence - Cloud Migration Utility")
    print("=====================================================")
    print("1. Migrate Database (PostgreSQL -> Supabase)")
    print("2. Migrate Vectors (Local Qdrant -> Qdrant Cloud)")
    print("3. Migrate All")
    print("q. Quit")
    
    choice = input("\nSelect an option: ")
    
    if choice == '1':
        migrate_database()
    elif choice == '2':
        migrate_vectors()
    elif choice == '3':
        migrate_database()
        migrate_vectors()
    elif choice == 'q':
        sys.exit()
    else:
        print("Invalid choice.")
