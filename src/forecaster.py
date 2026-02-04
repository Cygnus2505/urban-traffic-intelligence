"""Integrated traffic forecasting system combining RAG and predictive models."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .models.trainer import TrafficModelTrainer
from .rag.rag_system import TrafficRAGSystem
from .rag.vector_db import TrafficVectorDB
from .data.preprocessor import TrafficDataPreprocessor
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridTrafficForecaster:
    """Hybrid forecasting system combining ML predictions with RAG insights."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_rag: bool = True
    ):
        """Initialize the hybrid forecasting system.
        
        Args:
            model_path: Path to pre-trained model
            use_rag: Whether to use RAG for context-aware predictions
        """
        self.preprocessor = TrafficDataPreprocessor()
        self.use_rag = use_rag
        
        # Initialize RAG system
        if use_rag:
            self.vector_db = TrafficVectorDB()
            # Initialize traffic knowledge if database is empty
            if self.vector_db.collection.count() == 0:
                self.vector_db.initialize_traffic_knowledge()
            self.rag_system = TrafficRAGSystem(self.vector_db)
        else:
            self.rag_system = None
        
        # Initialize ML model (will be loaded or trained later)
        self.ml_model = None
        self.model_path = model_path
        
        logger.info("Hybrid traffic forecasting system initialized")
    
    def train_model(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        model_type: str = 'lstm',
        epochs: int = None,
        save_path: Optional[Path] = None
    ):
        """Train the predictive model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            model_type: Type of model to train
            epochs: Number of training epochs
            save_path: Path to save the trained model
        """
        logger.info(f"Training {model_type} model")
        
        # Preprocess data
        train_processed = self.preprocessor.preprocess(train_data)
        
        # Create sequences
        X_train, y_train = self.preprocessor.create_sequences(
            train_processed,
            sequence_length=Config.SEQUENCE_LENGTH,
            forecast_horizon=Config.FORECAST_HORIZON
        )
        
        # Normalize features
        X_train_normalized = self.preprocessor.normalize_features(X_train, fit=True)
        
        # Prepare validation data if provided
        X_val_normalized, y_val = None, None
        if val_data is not None:
            val_processed = self.preprocessor.preprocess(val_data)
            X_val, y_val = self.preprocessor.create_sequences(
                val_processed,
                sequence_length=Config.SEQUENCE_LENGTH,
                forecast_horizon=Config.FORECAST_HORIZON
            )
            X_val_normalized = self.preprocessor.normalize_features(X_val, fit=False)
        
        # Initialize and train model
        input_size = X_train_normalized.shape[-1]
        self.ml_model = TrafficModelTrainer(
            model_type=model_type,
            input_size=input_size,
            forecast_horizon=Config.FORECAST_HORIZON
        )
        
        history = self.ml_model.train(
            X_train_normalized, y_train,
            X_val_normalized, y_val,
            epochs=epochs
        )
        
        # Save model
        if save_path:
            self.ml_model.save_model(save_path)
        
        logger.info("Model training completed")
        return history
    
    def load_model(self, model_path: Path):
        """Load a pre-trained model.
        
        Args:
            model_path: Path to the model file
        """
        # This requires knowing the input size, which should be saved with the model
        # For now, we'll assume a default
        logger.info(f"Loading model from {model_path}")
        # Implementation would require saving/loading model metadata
        pass
    
    def predict(
        self,
        input_data: pd.DataFrame,
        location: str,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """Make traffic predictions with optional RAG context.
        
        Args:
            input_data: Input data for prediction
            location: Location identifier
            include_context: Whether to include RAG context
            
        Returns:
            Dictionary with predictions and context
        """
        if self.ml_model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Preprocess input
        processed = self.preprocessor.preprocess(input_data)
        
        # Create sequences
        X, _ = self.preprocessor.create_sequences(
            processed,
            sequence_length=Config.SEQUENCE_LENGTH,
            forecast_horizon=Config.FORECAST_HORIZON
        )
        
        # Normalize
        X_normalized = self.preprocessor.normalize_features(X, fit=False)
        
        # Make predictions
        predictions = self.ml_model.predict(X_normalized)
        
        # Get the most recent prediction
        latest_prediction = predictions[-1]  # Shape: (forecast_horizon,)
        
        # Prepare timestamps for forecast
        if 'timestamp' in input_data.columns:
            last_timestamp = pd.to_datetime(input_data['timestamp'].iloc[-1])
        else:
            last_timestamp = datetime.now()
        
        forecast_timestamps = [
            last_timestamp + timedelta(hours=i+1) 
            for i in range(Config.FORECAST_HORIZON)
        ]
        
        result = {
            'location': location,
            'forecast_start': last_timestamp.isoformat(),
            'predictions': []
        }
        
        # Add predictions with context if RAG is enabled
        for i, (timestamp, speed) in enumerate(zip(forecast_timestamps, latest_prediction)):
            pred_dict = {
                'timestamp': timestamp.isoformat(),
                'hour': timestamp.hour,
                'predicted_speed': float(speed)
            }
            
            if include_context and self.use_rag:
                # Get RAG context for this specific prediction
                augmented = self.rag_system.augment_prediction_with_context(
                    prediction=float(speed),
                    location=location,
                    timestamp=timestamp
                )
                pred_dict['condition'] = augmented['condition']
                pred_dict['relevant_factors'] = augmented['relevant_factors']
                
                # Add context only for first few predictions to avoid verbosity
                if i < 3:
                    pred_dict['context'] = augmented['context']
            
            result['predictions'].append(pred_dict)
        
        # Add overall summary if RAG is enabled
        if include_context and self.use_rag:
            avg_speed = float(np.mean(latest_prediction))
            features = {
                'hour': forecast_timestamps[0].hour,
                'is_rush_hour': forecast_timestamps[0].hour in Config.RUSH_HOURS,
                'is_weekend': forecast_timestamps[0].weekday() >= 5
            }
            result['explanation'] = self.rag_system.explain_prediction(avg_speed, features)
        
        return result
    
    def forecast_multi_location(
        self,
        locations_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Make predictions for multiple locations.
        
        Args:
            locations_data: Dictionary mapping location names to their data
            
        Returns:
            Dictionary of predictions for each location
        """
        results = {}
        for location, data in locations_data.items():
            logger.info(f"Predicting for location: {location}")
            results[location] = self.predict(data, location)
        
        return results
    
    def query_knowledge(self, question: str) -> str:
        """Query the traffic knowledge base.
        
        Args:
            question: Natural language question
            
        Returns:
            Answer from knowledge base
        """
        if not self.use_rag:
            return "RAG system is not enabled."
        
        return self.rag_system.query_traffic_knowledge(question)
    
    def add_traffic_knowledge(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add new traffic knowledge to the RAG system.
        
        Args:
            documents: List of documents to add
            metadatas: Optional metadata for each document
        """
        if not self.use_rag:
            logger.warning("RAG system is not enabled")
            return
        
        self.vector_db.add_documents(documents, metadatas)
        logger.info(f"Added {len(documents)} documents to knowledge base")
