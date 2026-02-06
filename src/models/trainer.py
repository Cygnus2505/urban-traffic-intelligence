"""
Model Training Script
Trains XGBoost model for traffic congestion prediction.
"""
import pandas as pd
import xgboost as xgb
import joblib
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from loguru import logger

from ..config import get_settings
from ..features.feature_builder import FeatureBuilder

settings = get_settings()

class TrafficModelTrainer:
    def __init__(self):
        self.feature_builder = FeatureBuilder()
        self.model_path = "src/models/traffic_model.json"
        
    def load_data(self, limit: int = 50000) -> pd.DataFrame:
        """Load recent training data from database"""
        logger.info(f"Loading last {limit} records from database...")
        engine = create_engine(settings.database_url)
        query = f"""
        SELECT * FROM traffic_congestion 
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """
        return pd.read_sql(query, engine)
        
    def train(self):
        """Execute full training pipeline"""
        # 1. Load Data
        df = self.load_data()
        if df.empty:
            logger.error("No data found for training!")
            return
            
        logger.info(f"Loaded {len(df)} records.")
        
        # 2. Build Features
        X, y = self.feature_builder.create_features(df.to_dict('records'), training=True)
        
        if y is None:
            logger.error("Could not generate target variable.")
            return

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Train XGBoost
        logger.info("Training XGBoost Regressor...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Model Evaluation Results:")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R2 Score: {r2:.4f}")
        
        # 6. Save Model
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save_model(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

if __name__ == "__main__":
    trainer = TrafficModelTrainer()
    trainer.train()
