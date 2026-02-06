"""
Vector Retrieval Module
Handles interaction with Qdrant vector database for storing and searching vectors.
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger
import uuid

from ..config import get_settings

settings = get_settings()


class QdrantRetriever:
    """Manages vector storage and retrieval using Qdrant"""
    
    def __init__(self):
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            self.client = QdrantClient(
                host=settings.api_host if settings.qdrant_host == "qdrant" else settings.qdrant_host,
                port=settings.qdrant_port
            )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = 1536  # OpenAI text-embedding-3-small dimension
        
        self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Failed to check/create collection: {e}")
            # Don't raise here, allow app to start even if Qdrant is down temporarily
            
    def index_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Save chunks and their embeddings to Qdrant.
        
        Args:
            chunks: List of chunk dicts (must contain 'content' and 'metadata')
            embeddings: corresponding vectors
        """
        if not chunks or not embeddings:
            return
            
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),  # Generate unique ID
                vector=vector,
                payload={
                    "content": chunk["content"],
                    "parent_id": str(chunk.get("parent_id", "")),
                    **chunk.get("metadata", {})
                }
            ))
            
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Indexed {len(points)} chunks to Qdrant")
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            raise

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most similar chunks to the query vector.
        
        Args:
            query_vector: Embedding of the search query
            top_k: Number of results to return
            
        Returns:
            List of dicts containing 'content' and 'score'
        """
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=settings.min_retrieval_score
            ).points
            
            return [
                {
                    "content": hit.payload.get("content", ""),
                    "metadata": {k:v for k,v in hit.payload.items() if k != "content"},
                    "score": hit.score
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
