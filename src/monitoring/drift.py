"""
ML Model Drift Detection
Monitors changes in data distributions to detect when models may need retraining.
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from loguru import logger
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine

from ..config import get_settings

settings = get_settings()

class DriftDetector:
    """
    Detects statistical drift in traffic data.
    """
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url)
        
    def get_reference_data(self, days_back: int = 7, limit: int = 5000) -> pd.DataFrame:
        """Load data from a past window to act as a reference"""
        start_date = datetime.now() - timedelta(days=days_back + 7)
        end_date = datetime.now() - timedelta(days=days_back)
        
        query = f"""
            SELECT congestion_level, hour, day_of_week 
            FROM traffic_congestion 
            WHERE timestamp >= '{start_date.isoformat()}' 
            AND timestamp <= '{end_date.isoformat()}'
            LIMIT {limit}
        """
        return pd.read_sql(query, self.engine)

    def get_current_data(self, limit: int = 1000) -> pd.DataFrame:
        """Load the most recent data to check for drift"""
        query = f"""
            SELECT congestion_level, hour, day_of_week 
            FROM traffic_congestion 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        return pd.read_sql(query, self.engine)

    def calculate_drift(self) -> Dict[str, Any]:
        """
        Perform K-S test to detect drift in congestion levels.
        """
        try:
            ref_df = self.get_reference_data()
            curr_df = self.get_current_data()
            
            if ref_df.empty or curr_df.empty:
                return {"status": "insufficient_data", "is_drifted": False}
                
            # Perform K-S Test on congestion_level
            statistic, p_value = ks_2samp(
                ref_df['congestion_level'].dropna(), 
                curr_df['congestion_level'].dropna()
            )
            
            # If p-value is very small (e.g. < 0.05), the distributions are significantly different
            is_drifted = p_value < 0.05
            
            drift_report = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "is_drifted": is_drifted,
                "p_value": round(float(p_value), 4),
                "ks_statistic": round(float(statistic), 4),
                "ref_count": len(ref_df),
                "curr_count": len(curr_df)
            }
            
            if is_drifted:
                logger.warning(f"MODEL DRIFT DETECTED: p-value={p_value:.4f}")
            else:
                logger.info(f"No significant drift detected: p-value={p_value:.4f}")
                
            return drift_report
            
        except Exception as e:
            logger.error(f"Drift calculation failed: {e}")
            return {"status": "error", "message": str(e), "is_drifted": False}
