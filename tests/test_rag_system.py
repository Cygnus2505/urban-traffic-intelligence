"""Tests for RAG system."""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.rag.vector_db import TrafficVectorDB
from src.rag.rag_system import TrafficRAGSystem


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the database."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_vector_db_initialization(temp_db_path):
    """Test vector database initialization."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    assert db is not None
    assert db.collection is not None
    assert db.embedding_model is not None


def test_add_documents(temp_db_path):
    """Test adding documents to vector database."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    
    documents = [
        "Rush hour traffic occurs from 7-9 AM.",
        "Weekend traffic is lighter than weekday traffic."
    ]
    
    initial_count = db.collection.count()
    db.add_documents(documents)
    
    assert db.collection.count() == initial_count + len(documents)


def test_query_documents(temp_db_path):
    """Test querying documents from vector database."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    
    documents = [
        "Rush hour traffic occurs from 7-9 AM and 4-7 PM.",
        "Weekend traffic patterns differ from weekdays.",
        "Weather conditions significantly impact traffic flow."
    ]
    
    db.add_documents(documents)
    
    # Query for rush hour information
    results = db.query("When is rush hour?", n_results=2)
    
    assert 'documents' in results
    assert len(results['documents']) > 0
    assert len(results['documents'][0]) <= 2


def test_initialize_traffic_knowledge(temp_db_path):
    """Test initialization of traffic knowledge base."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    
    initial_count = db.collection.count()
    db.initialize_traffic_knowledge()
    
    # Should have added multiple documents
    assert db.collection.count() > initial_count


def test_rag_system_initialization(temp_db_path):
    """Test RAG system initialization."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    rag_system = TrafficRAGSystem(vector_db=db)
    
    assert rag_system is not None
    assert rag_system.vector_db is not None


def test_retrieve_relevant_knowledge(temp_db_path):
    """Test knowledge retrieval."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    db.initialize_traffic_knowledge()
    
    rag_system = TrafficRAGSystem(vector_db=db)
    
    relevant_docs = rag_system.retrieve_relevant_knowledge(
        "What are rush hour patterns?",
        top_k=3
    )
    
    assert isinstance(relevant_docs, list)
    assert len(relevant_docs) <= 3
    if relevant_docs:
        assert 'content' in relevant_docs[0]
        assert 'metadata' in relevant_docs[0]


def test_query_traffic_knowledge(temp_db_path):
    """Test natural language knowledge query."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    db.initialize_traffic_knowledge()
    
    rag_system = TrafficRAGSystem(vector_db=db)
    
    answer = rag_system.query_traffic_knowledge(
        "How does weather affect traffic?"
    )
    
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_explain_prediction(temp_db_path):
    """Test prediction explanation generation."""
    db = TrafficVectorDB(persist_directory=temp_db_path)
    db.initialize_traffic_knowledge()
    
    rag_system = TrafficRAGSystem(vector_db=db)
    
    features = {
        'hour': 8,
        'is_rush_hour': 1,
        'is_weekend': 0
    }
    
    explanation = rag_system.explain_prediction(
        prediction=25.0,
        features=features
    )
    
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert 'rush hour' in explanation.lower() or 'congestion' in explanation.lower()
