#!/usr/bin/env python
"""Example script demonstrating the traffic forecasting system."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.forecaster import HybridTrafficForecaster
from src.data.collector import ChicagoTrafficCollector
from src.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the example."""
    logger.info("=" * 80)
    logger.info("Chicago Traffic Forecasting System - Example")
    logger.info("=" * 80)
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Step 1: Generate synthetic data
    logger.info("\n1. Generating synthetic traffic data...")
    collector = ChicagoTrafficCollector()
    data = collector.generate_synthetic_data(num_days=30, locations=10)
    
    data_path = Config.DATA_DIR / "example_traffic_data.csv"
    collector.save_data(data, data_path)
    logger.info(f"Generated {len(data)} records, saved to {data_path}")
    
    # Step 2: Initialize the hybrid forecasting system
    logger.info("\n2. Initializing Hybrid Traffic Forecasting System...")
    forecaster = HybridTrafficForecaster(use_rag=True)
    logger.info("System initialized with RAG enabled")
    
    # Step 3: Query the knowledge base
    logger.info("\n3. Querying Traffic Knowledge Base...")
    questions = [
        "What are typical rush hour patterns in Chicago?",
        "How does weather affect traffic?",
        "What should I know about weekend traffic?"
    ]
    
    for question in questions:
        logger.info(f"\nQ: {question}")
        answer = forecaster.query_knowledge(question)
        logger.info(f"A: {answer}")
    
    # Step 4: Add custom knowledge
    logger.info("\n4. Adding custom traffic knowledge...")
    custom_knowledge = [
        "The Eisenhower Expressway (I-290) is one of Chicago's most congested highways, "
        "particularly during morning rush hour heading inbound to the city.",
        "Construction on the Jane Byrne Interchange has created additional bottlenecks "
        "for traffic traveling between I-90, I-94, and I-290."
    ]
    
    forecaster.add_traffic_knowledge(
        documents=custom_knowledge,
        metadatas=[
            {"category": "locations", "type": "i290"},
            {"category": "factors", "type": "construction"}
        ]
    )
    logger.info(f"Added {len(custom_knowledge)} custom knowledge entries")
    
    # Step 5: Train a simple model (with small subset for demonstration)
    logger.info("\n5. Training prediction model...")
    logger.info("Note: Using small dataset for demonstration. In production, use more data.")
    
    # Use a subset for quick training
    train_data = data[data['segment_id'] == 1].head(1000)
    
    try:
        history = forecaster.train_model(
            train_data=train_data,
            model_type='lstm',
            epochs=5,  # Just 5 epochs for demo
            save_path=Config.MODEL_DIR / "example_model.pth"
        )
        logger.info("Model training completed!")
        logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")
    except Exception as e:
        logger.warning(f"Model training skipped due to: {e}")
        logger.info("This is expected if you don't have all dependencies installed.")
        logger.info("The RAG system and knowledge base are still fully functional!")
    
    # Step 6: Make predictions (if model was trained)
    if forecaster.ml_model is not None:
        logger.info("\n6. Making predictions with RAG context...")
        
        # Use recent data for prediction
        recent_data = data[data['segment_id'] == 1].tail(200)
        
        try:
            predictions = forecaster.predict(
                input_data=recent_data,
                location="I-290 Eisenhower Expressway",
                include_context=True
            )
            
            logger.info(f"\nForecast for {predictions['location']}:")
            logger.info(f"Forecast starts at: {predictions['forecast_start']}")
            
            # Show first 5 predictions
            for pred in predictions['predictions'][:5]:
                logger.info(f"  {pred['timestamp']}: {pred['predicted_speed']:.1f} mph - {pred.get('condition', 'N/A')}")
            
            if 'explanation' in predictions:
                logger.info(f"\nExplanation: {predictions['explanation']}")
        
        except Exception as e:
            logger.warning(f"Prediction skipped: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Example completed successfully!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Start the API server: python -m src.api.main")
    logger.info("2. Access API docs at: http://localhost:8000/docs")
    logger.info("3. Query knowledge base via API")
    logger.info("4. Train models with more data")


if __name__ == "__main__":
    main()
