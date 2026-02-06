"""
Prediction API Routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from loguru import logger

from ...monitoring.metrics import PREDICTION_LATENCY, PREDICTION_VALUE
from ...models.predictor import TrafficPredictor
from ...models.trainer import TrafficModelTrainer

router = APIRouter(tags=["Forecasting"])

# Initialize predictor globally (lazy loading)
predictor = TrafficPredictor()

class PredictionRequest(BaseModel):
    segment_id: str
    timestamp: Optional[datetime] = None  # Default to now if None
    
class PredictionResponse(BaseModel):
    segment_id: str
    timestamp: datetime
    congestion_level: float
    description: str

class TrainResponse(BaseModel):
    message: str
    status: str

def get_congestion_description(level: float) -> str:
    """Convert numerical score to human description"""
    if level < 0.3: return "Free Flow"
    if level < 0.6: return "Moderate"
    if level < 0.8: return "Heavy"
    return "Severe Congestion"

async def train_model_task():
    """Background task to retrain model"""
    logger.info("Starting model training task...")
    try:
        trainer = TrafficModelTrainer()
        trainer.train()
        # Reload predictor
        predictor.load_model()
        logger.info("Training complete and model reloaded.")
    except Exception as e:
        logger.error(f"Training failed: {e}")

@router.post("/predict", response_model=PredictionResponse)
async def predict_traffic(request: PredictionRequest):
    """
    Predict future traffic congestion.
    """
    target_time = request.timestamp or datetime.utcnow()
    
    # Ensure time is in future or now (model might support past, but conceptual goal is future)
    
    start_time = datetime.now()
    try:
        score = predictor.predict(request.segment_id, target_time)
        
        # Track metrics
        latency = (datetime.now() - start_time).total_seconds()
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_VALUE.observe(score)
        
        return {
            "segment_id": request.segment_id,
            "timestamp": target_time,
            "congestion_level": round(score, 2),
            "description": get_congestion_description(score)
        }
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=TrainResponse)
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in background.
    """
    background_tasks.add_task(train_model_task)
    return {"message": "Model training started in background", "status": "processing"}
