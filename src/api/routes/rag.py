"""
RAG API Routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime

from ...rag.chunker import DocumentChunker
from ...rag.embedder import OpenAIEmbedder
from ...rag.retriever import QdrantRetriever
from ...rag.generator import RAGGenerator
from ...monitoring.metrics import RAG_QUERY_TOTAL, RAG_FAITHFULNESS_SCORE
from ...guardrails.validators import RAGGuardrail
from ...config import get_settings

settings = get_settings()

router = APIRouter(tags=["RAG"])

# Pydantic models for request/response
class RAGQuery(BaseModel):
    query: str

class Source(BaseModel):
    content_snippet: str
    score: float
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: float
    is_relevant: bool = True
    faithfulness_score: float = 1.0

class IngestionResponse(BaseModel):
    message: str
    # documents_processed: int


async def process_documents_task(limit: int = 100):
    """Background task to process pending documents"""
    logger.info("Starting document processing task...")
    try:
        # Initialize RAG components
        chunker = DocumentChunker()
        embedder = OpenAIEmbedder()
        retriever = QdrantRetriever()
        
        # Get unprocessed documents
        session = get_session(settings.database_url)
        docs = session.query(Document).filter(Document.is_embedded == False).limit(limit).all()
        
        if not docs:
            logger.info("No pending documents to process")
            return
            
        # Convert to format for chunker
        doc_dicts = [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": {
                    "source_id": doc.source_id,
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                    "location": doc.location,
                    "created_date": doc.created_date.isoformat() if doc.created_date else None
                }
            }
            for doc in docs
        ]
        
        # 1. Chunk
        chunks = chunker.chunk_documents(doc_dicts)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
        
        if not chunks:
            return

        # 2. Embed
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed_texts(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # 3. Index
        retriever.index_chunks(chunks, embeddings)
        
        # 4. Update status
        for doc in docs:
            doc.is_embedded = True
            doc.updated_at = datetime.utcnow()
            
        session.commit()
        logger.info("Document processing complete")
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
    finally:
        session.close()


@router.post("/ask", response_model=RAGResponse)
async def ask_question(request: RAGQuery):
    """
    Ask a natural language question about traffic.
    """
    start_time = datetime.now()
    
    # 1. Relevance Guardrail
    if not RAGGuardrail.is_relevant(request.query):
        RAG_QUERY_TOTAL.labels(status="blocked", is_relevant="false").inc()
        return {
            "answer": "I'm sorry, I can only answer questions related to Chicago traffic, incidents, and roadwork. Please ask something about traffic conditions or crashes.",
            "sources": [],
            "latency_ms": 10,
            "is_relevant": False,
            "faithfulness_score": 0.0
        }
        
    try:
        # Initialize components
        embedder = OpenAIEmbedder()
        retriever = QdrantRetriever()
        generator = RAGGenerator()
        
        # 2. Embed query
        query_embedding = embedder.embed_texts([request.query])[0]
        
        # 3. Retrieve relevant chunks
        context_chunks = retriever.search(query_embedding, top_k=5)
        
        # 4. Generate answer
        result = generator.generate_answer(request.query, context_chunks)
        
        # 5. Faithfulness Guardrail
        source_texts = [s["content_snippet"] for s in result["sources"]]
        faithfulness = RAGGuardrail.validate_response_faithfulness(result["answer"], source_texts)
        
        # Track metrics
        RAG_QUERY_TOTAL.labels(status="success", is_relevant="true").inc()
        RAG_FAITHFULNESS_SCORE.observe(faithfulness)
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log query to database
        try:
            session = get_session(settings.database_url)
            log = RAGQueryLog(
                query=request.query,
                answer=result["answer"],
                sources_count=len(result["sources"]),
                latency_ms=latency_ms,
                is_relevant=True,
                faithfulness_score=faithfulness
            )
            session.add(log)
            session.commit()
            session.close()
        except Exception as log_err:
            logger.error(f"Failed to log RAG query: {log_err}")
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "latency_ms": latency_ms,
            "is_relevant": True,
            "faithfulness_score": round(faithfulness, 2)
        }
        
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks, limit: int = 100):
    """
    Trigger background ingestion of pending documents.
    """
    background_tasks.add_task(process_documents_task, limit)
    return {"message": "Ingestion task started in background"}
