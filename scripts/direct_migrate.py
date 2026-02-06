"""
Direct Data Migration
Synchronous migration with direct prints for visibility.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

def run_migration():
    local_db = os.getenv("DATABASE_URL")
    cloud_db = os.getenv("CLOUD_DATABASE_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    print(f"Local DB: {local_db}")
    print(f"Cloud DB: {cloud_db[:50]}...")

    local_engine = create_engine(local_db)
    cloud_engine = create_engine(cloud_db)

    tables = [
        'traffic_congestion', 
        'traffic_crashes', 
        'road_construction', 
        'weather_data', 
        'documents', 
        'prediction_logs', 
        'rag_query_logs'
    ]

    print("\n--- Migrating Database ---")
    for table in tables:
        try:
            print(f"Migrating {table}...")
            # Supabase can be slow, using small chunks
            rows = 0
            for chunk in pd.read_sql_table(table, local_engine, chunksize=2000):
                chunk.to_sql(table, cloud_engine, if_exists='append', index=False)
                rows += len(chunk)
            print(f"SUCCESS: Migrated {rows} rows for {table}")
        except Exception as e:
            print(f"ERROR: Failed {table}: {e}")

    print("\n--- Migrating Vectors ---")
    try:
        local_q = QdrantClient(host="localhost", port=6333)
        cloud_q = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        collection = "traffic_documents"
        
        print(f"Fetching vectors from {collection}...")
        points, _ = local_q.scroll(collection_name=collection, limit=10000, with_vectors=True)
        if points:
            print(f"Uploading {len(points)} vectors to Cloud...")
            # Transform Record to PointStruct
            cloud_points = [
                models.PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=p.payload
                ) for p in points
            ]
            cloud_q.upsert(collection_name=collection, points=cloud_points)
            print("SUCCESS: Vectors migrated.")
        else:
            print("No vectors found locally.")
    except Exception as e:
        print(f"ERROR: Vector migration failed: {e}")

if __name__ == "__main__":
    run_migration()
