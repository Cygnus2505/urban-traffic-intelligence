"""RAG system for traffic forecasting with retrieval-augmented generation."""
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

from .vector_db import TrafficVectorDB
from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficRAGSystem:
    """RAG system that combines retrieval with traffic predictions."""
    
    def __init__(self, vector_db: Optional[TrafficVectorDB] = None):
        """Initialize the RAG system.
        
        Args:
            vector_db: Optional vector database instance
        """
        self.vector_db = vector_db or TrafficVectorDB()
        self.llm_available = bool(Config.OPENAI_API_KEY)
        
        if not self.llm_available:
            logger.warning("OpenAI API key not configured. LLM features will be disabled.")
    
    def retrieve_relevant_knowledge(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant traffic knowledge for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        results = self.vector_db.query(query, n_results=top_k)
        
        relevant_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                relevant_docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return relevant_docs
    
    def generate_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate context from retrieved documents.
        
        Args:
            relevant_docs: List of relevant documents
            
        Returns:
            Formatted context string
        """
        if not relevant_docs:
            return "No relevant traffic knowledge found."
        
        context_parts = ["Relevant traffic knowledge:"]
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"{i}. {doc['content']}")
        
        return "\n".join(context_parts)
    
    def augment_prediction_with_context(
        self,
        prediction: float,
        location: str,
        timestamp: datetime,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Augment a traffic prediction with relevant context.
        
        Args:
            prediction: Predicted traffic speed
            location: Location identifier
            timestamp: Timestamp for the prediction
            additional_context: Optional additional context
            
        Returns:
            Dictionary with prediction and context
        """
        # Create query based on time and location
        hour = timestamp.hour
        day_of_week = timestamp.strftime('%A')
        
        query = f"Traffic patterns for {location} on {day_of_week} at {hour}:00"
        if additional_context:
            query += f" {additional_context}"
        
        # Retrieve relevant knowledge
        relevant_docs = self.retrieve_relevant_knowledge(query)
        context = self.generate_context(relevant_docs)
        
        # Determine conditions based on prediction
        if prediction < 20:
            condition = "severe congestion"
        elif prediction < 35:
            condition = "heavy traffic"
        elif prediction < 50:
            condition = "moderate traffic"
        else:
            condition = "light traffic"
        
        return {
            'prediction': prediction,
            'location': location,
            'timestamp': timestamp.isoformat(),
            'condition': condition,
            'context': context,
            'relevant_factors': [doc['metadata'].get('type', 'general') for doc in relevant_docs]
        }
    
    def explain_prediction(
        self,
        prediction: float,
        features: Dict[str, Any]
    ) -> str:
        """Generate an explanation for a traffic prediction.
        
        Args:
            prediction: Predicted value
            features: Feature values used in prediction
            
        Returns:
            Human-readable explanation
        """
        explanation_parts = []
        
        # Basic prediction statement
        speed = prediction
        if speed < 20:
            explanation_parts.append("Heavy congestion is expected with very slow traffic speeds.")
        elif speed < 35:
            explanation_parts.append("Moderate to heavy traffic is expected with below-average speeds.")
        elif speed < 50:
            explanation_parts.append("Normal traffic flow is expected with average speeds.")
        else:
            explanation_parts.append("Light traffic is expected with good flow.")
        
        # Add feature-based explanations
        if 'is_rush_hour' in features and features['is_rush_hour']:
            explanation_parts.append("This is during rush hour, which typically sees increased congestion.")
        
        if 'is_weekend' in features and features['is_weekend']:
            explanation_parts.append("Weekend traffic patterns apply, which differ from weekday patterns.")
        
        if 'hour' in features:
            hour = features['hour']
            if 7 <= hour <= 9:
                explanation_parts.append("Morning rush hour contributes to slower traffic.")
            elif 16 <= hour <= 18:
                explanation_parts.append("Evening rush hour contributes to slower traffic.")
        
        # Retrieve contextual knowledge
        query = f"Traffic speed {speed:.1f} mph factors"
        relevant_docs = self.retrieve_relevant_knowledge(query, top_k=2)
        
        if relevant_docs:
            explanation_parts.append("\nRelevant insights:")
            for doc in relevant_docs[:2]:
                explanation_parts.append(f"- {doc['content'][:200]}...")
        
        return " ".join(explanation_parts)
    
    def query_traffic_knowledge(self, question: str) -> str:
        """Query the traffic knowledge base with a natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            Answer based on retrieved knowledge
        """
        relevant_docs = self.retrieve_relevant_knowledge(question, top_k=3)
        
        if not relevant_docs:
            return "I don't have specific information about that topic in my knowledge base."
        
        # Format response
        response_parts = ["Based on traffic knowledge:"]
        for doc in relevant_docs:
            response_parts.append(f"\n• {doc['content']}")
        
        return "\n".join(response_parts)
