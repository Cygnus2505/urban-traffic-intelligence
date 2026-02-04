"""Vector database management for traffic knowledge storage and retrieval."""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficVectorDB:
    """Vector database for storing and retrieving traffic-related knowledge."""
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """Initialize the vector database.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory or Config.VECTOR_DB_PATH
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=Config.VECTOR_DB_COLLECTION,
            metadata={"description": "Traffic knowledge and insights"}
        )
        
        logger.info(f"Vector database initialized with {self.collection.count()} documents")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to the vector database.
        
        Args:
            documents: List of text documents to add
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to vector database")
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the vector database for relevant documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary containing query results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get all documents from the collection.
        
        Returns:
            Dictionary containing all documents
        """
        return self.collection.get()
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(Config.VECTOR_DB_COLLECTION)
        logger.info("Collection deleted")
    
    def initialize_traffic_knowledge(self):
        """Initialize the database with traffic domain knowledge."""
        logger.info("Initializing traffic knowledge base")
        
        # Traffic patterns and insights
        traffic_knowledge = [
            {
                "content": "Rush hour traffic in Chicago typically occurs between 7-9 AM and 4-7 PM on weekdays. During these times, major highways like I-90, I-94, and I-290 experience significant congestion.",
                "metadata": {"category": "patterns", "type": "rush_hour"}
            },
            {
                "content": "Weekend traffic patterns in Chicago differ significantly from weekdays. Saturday afternoons see increased traffic near shopping districts and entertainment venues, while Sunday evenings experience congestion from returning weekend travelers.",
                "metadata": {"category": "patterns", "type": "weekend"}
            },
            {
                "content": "Weather conditions significantly impact traffic flow. Snow and ice can reduce average speeds by 30-50%, while heavy rain typically reduces speeds by 10-20%. Traffic volume often decreases during severe weather events.",
                "metadata": {"category": "factors", "type": "weather"}
            },
            {
                "content": "Special events like Chicago Cubs games at Wrigley Field, concerts at United Center, and festivals in Grant Park can cause localized traffic surges. Plan for 2-3x normal travel times near event venues.",
                "metadata": {"category": "events", "type": "special"}
            },
            {
                "content": "Construction zones in Chicago often reduce lane capacity and create bottlenecks. Major construction projects can persist for months, requiring commuters to plan alternate routes.",
                "metadata": {"category": "factors", "type": "construction"}
            },
            {
                "content": "Public transit usage in Chicago peaks during rush hours, with the CTA Red and Blue Lines experiencing maximum capacity. Traffic tends to be lighter near major CTA stations as commuters opt for public transit.",
                "metadata": {"category": "transit", "type": "public"}
            },
            {
                "content": "Traffic speed predictions should account for historical patterns, current conditions, and known upcoming events. Machine learning models perform best when combining multiple data sources.",
                "metadata": {"category": "modeling", "type": "prediction"}
            },
            {
                "content": "Chicago's grid system allows for multiple route alternatives. During congestion, side streets may offer faster travel times than major arterials, though this varies by time of day.",
                "metadata": {"category": "routing", "type": "alternatives"}
            },
            {
                "content": "Lake Shore Drive experiences unique traffic patterns influenced by recreational activities, weather, and Chicago's lakefront attractions. Summer weekends see increased northbound traffic in the afternoon.",
                "metadata": {"category": "locations", "type": "lake_shore_drive"}
            },
            {
                "content": "Traffic volume typically decreases by 20-30% during major holidays like Thanksgiving, Christmas, and New Year's Day. The days immediately before and after holidays often see increased traffic.",
                "metadata": {"category": "patterns", "type": "holidays"}
            }
        ]
        
        # Add documents to database
        documents = [item["content"] for item in traffic_knowledge]
        metadatas = [item["metadata"] for item in traffic_knowledge]
        
        self.add_documents(documents, metadatas)
        logger.info(f"Initialized knowledge base with {len(documents)} documents")
