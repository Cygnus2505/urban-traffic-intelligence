"""Model trainer for traffic forecasting models."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
from tqdm import tqdm

from .forecasting_models import TrafficLSTM, TrafficGRU, AttentionLSTM
from ..config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficModelTrainer:
    """Trainer for traffic forecasting models."""
    
    def __init__(
        self,
        model_type: str = 'lstm',
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24,
        device: Optional[str] = None
    ):
        """Initialize the model trainer.
        
        Args:
            model_type: Type of model ('lstm', 'gru', 'attention_lstm')
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            forecast_horizon: Number of time steps to predict
            device: Device to use for training
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        if model_type == 'lstm':
            self.model = TrafficLSTM(
                input_size, hidden_size, num_layers, dropout, forecast_horizon
            )
        elif model_type == 'gru':
            self.model = TrafficGRU(
                input_size, hidden_size, num_layers, dropout, forecast_horizon
            )
        elif model_type == 'attention_lstm':
            self.model = AttentionLSTM(
                input_size, hidden_size, num_layers, dropout, forecast_horizon
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        
        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training history
        """
        epochs = epochs or Config.EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE
        
        logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self.evaluate(X_val, y_val, batch_size)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(Config.MODEL_DIR / "best_model.pth")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        return self.history
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = None
    ) -> float:
        """Evaluate the model.
        
        Args:
            X: Input data
            y: Target data
            batch_size: Batch size for evaluation
            
        Returns:
            Average loss
        """
        batch_size = batch_size or Config.BATCH_SIZE
        
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        total_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def save_model(self, path: Path):
        """Save the model.
        
        Args:
            path: Path to save the model
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load the model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        logger.info(f"Model loaded from {path}")
