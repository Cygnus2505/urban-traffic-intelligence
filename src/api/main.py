"""FastAPI application for traffic forecasting system."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import logging

from ..forecaster import HybridTrafficForecaster
from ..data.collector import ChicagoTrafficCollector
from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chicago Traffic Forecasting API",
    description="ML system combining RAG and predictive models for traffic forecasting",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize forecaster (global instance)
forecaster = None


# Pydantic models
class TrafficDataPoint(BaseModel):
    """Model for a single traffic data point."""
    segment_id: int
    timestamp: str
    speed: float
    traffic_volume: int
    street: str
    direction: str


class ForecastRequest(BaseModel):
    """Model for forecast request."""
    location: str
    historical_data: List[TrafficDataPoint]
    include_context: bool = True


class KnowledgeQuery(BaseModel):
    """Model for knowledge base query."""
    question: str


class AddKnowledgeRequest(BaseModel):
    """Model for adding knowledge to the system."""
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the forecaster on startup."""
    global forecaster
    logger.info("Initializing traffic forecasting system...")
    Config.ensure_directories()
    forecaster = HybridTrafficForecaster(use_rag=True)
    logger.info("System initialized successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Chicago Traffic Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "query": "/query-knowledge",
            "add_knowledge": "/add-knowledge",
            "generate_demo_data": "/generate-demo-data"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_enabled": forecaster.use_rag if forecaster else False,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict_traffic(request: ForecastRequest):
    """Make traffic predictions for a location.
    
    Args:
        request: Forecast request with location and historical data
        
    Returns:
        Predictions with optional RAG context
    """
    if forecaster is None or forecaster.ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train the model first or use demo mode."
        )
    
    try:
        # Convert request data to DataFrame
        data_dicts = [point.dict() for point in request.historical_data]
        df = pd.DataFrame(data_dicts)
        
        # Make prediction
        result = forecaster.predict(
            df,
            location=request.location,
            include_context=request.include_context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-knowledge")
async def query_knowledge(query: KnowledgeQuery):
    """Query the traffic knowledge base.
    
    Args:
        query: Natural language question
        
    Returns:
        Answer from knowledge base
    """
    if forecaster is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        answer = forecaster.query_knowledge(query.question)
        return {
            "question": query.question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-knowledge")
async def add_knowledge(request: AddKnowledgeRequest):
    """Add new knowledge to the RAG system.
    
    Args:
        request: Documents and metadata to add
        
    Returns:
        Success message
    """
    if forecaster is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        forecaster.add_traffic_knowledge(
            documents=request.documents,
            metadatas=request.metadatas
        )
        return {
            "message": f"Successfully added {len(request.documents)} documents",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Add knowledge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate-demo-data")
async def generate_demo_data():
    """Generate synthetic traffic data for demonstration.
    
    Returns:
        Sample of generated data
    """
    try:
        collector = ChicagoTrafficCollector()
        df = collector.generate_synthetic_data(num_days=7, locations=5)
        
        # Save for later use
        data_path = Config.DATA_DIR / "demo_traffic_data.csv"
        collector.save_data(df, data_path)
        
        # Return sample
        sample = df.head(20).to_dict(orient='records')
        
        return {
            "message": "Demo data generated successfully",
            "total_records": len(df),
            "sample": sample,
            "saved_to": str(data_path)
        }
    except Exception as e:
        logger.error(f"Demo data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-stats")
async def knowledge_stats():
    """Get statistics about the knowledge base.
    
    Returns:
        Knowledge base statistics
    """
    if forecaster is None or not forecaster.use_rag:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        all_docs = forecaster.vector_db.get_all_documents()
        return {
            "total_documents": len(all_docs['ids']) if all_docs['ids'] else 0,
            "collection_name": Config.VECTOR_DB_COLLECTION,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
