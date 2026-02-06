"""
Verify Cloud Data
Checks row counts and collections in cloud providers.
"""
import os
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def verify_cloud():
    cloud_db_url = os.getenv("CLOUD_DATABASE_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    print("\n--- Supabase Verification ---")
    if cloud_db_url:
        try:
            engine = create_engine(cloud_db_url)
            with engine.connect() as conn:
                tables = [
                    'traffic_congestion', 
                    'traffic_crashes', 
                    'road_construction', 
                    'weather_data', 
                    'documents', 
                    'rag_query_logs'
                ]
                for table in tables:
                    try:
                        res = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = res.scalar()
                        print(f"Table {table}: {count} rows")
                    except Exception as e:
                        print(f"Table {table} check failed: {e}")
        except Exception as e:
            print(f"Supabase check failed: {e}")
    else:
        print("CLOUD_DATABASE_URL missing")

    print("\n--- Qdrant Cloud Verification ---")
    if qdrant_url and qdrant_key:
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
            collection_name = "traffic_documents"
            if client.collection_exists(collection_name):
                info = client.get_collection(collection_name)
                print(f"Collection {collection_name}: {info.points_count} vectors")
            else:
                print(f"Collection {collection_name} does not exist")
        except Exception as e:
            print(f"Qdrant check failed: {e}")
    else:
        print("Qdrant credentials missing")

if __name__ == "__main__":
    verify_cloud()
