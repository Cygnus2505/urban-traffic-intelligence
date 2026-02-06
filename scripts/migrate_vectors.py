"""
Vector-Only Migration
Migrates local Qdrant vectors to Qdrant Cloud.
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

def migrate_vectors():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    collection = "traffic_documents"

    print(f"Connecting to Qdrant Cloud: {qdrant_url}")
    
    local_q = QdrantClient(host="localhost", port=6333)
    cloud_q = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    try:
        print(f"Fetching vectors from local {collection}...")
        # Get all points (paging if needed)
        limit = 5000
        offset = None
        total_migrated = 0
        
        while True:
            points, next_offset = local_q.scroll(
                collection_name=collection, 
                limit=limit, 
                with_vectors=True,
                offset=offset
            )
            
            if not points:
                break
                
            print(f"Uploading batch of {len(points)} vectors...")
            cloud_points = [
                models.PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=p.payload
                ) for p in points
            ]
            
            cloud_q.upsert(collection_name=collection, points=cloud_points)
            total_migrated += len(points)
            
            if not next_offset:
                break
            offset = next_offset

        print(f"SUCCESS: Total {total_migrated} vectors migrated.")
        
    except Exception as e:
        print(f"ERROR: Vector migration failed: {e}")

if __name__ == "__main__":
    migrate_vectors()
