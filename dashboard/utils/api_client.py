import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime

class TrafficAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def get_health(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def ask_rag(self, query: str) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/rag/ask",
                json={"query": query}
            )
            return response.json()
        except Exception as e:
            return {"answer": f"Error connecting to AI Assistant: {str(e)}", "sources": []}

    def get_prediction(self, segment_id: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        try:
            payload = {"segment_id": segment_id}
            if timestamp:
                payload["timestamp"] = timestamp.isoformat()
            
            response = requests.post(
                f"{self.base_url}/models/predict",
                json=payload
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def trigger_training(self) -> Dict[str, Any]:
        try:
            response = requests.post(f"{self.base_url}/models/train")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    def get_summary(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/data/summary")
            return response.json()
        except Exception as e:
            return {"metrics": {}, "segments": [], "error": str(e)}

    def get_all_segments(self) -> List[Dict[str, Any]]:
        try:
            response = requests.get(f"{self.base_url}/data/segments")
            return response.json()
        except Exception as e:
            return []

    def get_raw_traffic(self, limit: int = 100) -> pd.DataFrame:
        # Since we don't have a direct /traffic endpoint yet, we might need to add one or query DB
        # For now, let's assume we might add a simple one in routes/health or a new one.
        # Alternatively, we can use the metrics or just mock for the demo if needed.
        # But for a real dashboard, we need a list of segments.
        return pd.DataFrame()
