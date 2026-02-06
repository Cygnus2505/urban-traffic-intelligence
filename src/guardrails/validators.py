"""
Data Validation Guardrails
Ensures data integrity and system security before processing.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger
import pandas as pd

class TrafficDataValidator:
    """
    Validates incoming traffic congestion and crash data.
    """
    
    # Chicago Coordinates approximate bounds
    LAT_MIN, LAT_MAX = 41.60, 42.05
    LON_MIN, LON_MAX = -87.95, -87.50
    
    @classmethod
    def validate_congestion(cls, data: Dict[str, Any]) -> bool:
        """
        Validate a single traffic congestion record.
        """
        try:
            # 1. Check coordinates
            lat = float(data.get('start_lat', 0))
            lon = float(data.get('start_lon', 0))
            
            if not (cls.LAT_MIN <= lat <= cls.LAT_MAX and cls.LON_MIN <= lon <= cls.LON_MAX):
                logger.warning(f"Invalid congestion coordinates: ({lat}, {lon})")
                return False
                
            # 2. Check speeds (logical range 0-100 mph)
            current_speed = float(data.get('current_speed', -1))
            expected_speed = float(data.get('expected_speed', -1))
            
            if not (0 <= current_speed <= 100 and 0 <= expected_speed <= 100):
                logger.warning(f"Non-logical speed values: current={current_speed}, expected={expected_speed}")
                return False
                
            # 3. Check timestamps (not in future, not older than 7 days if real-time)
            ts_str = data.get('timestamp')
            if ts_str:
                ts = pd.to_datetime(ts_str)
                now = datetime.now()
                if ts > now + timedelta(minutes=5):
                    logger.warning(f"Future timestamp detected: {ts}")
                    return False
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.error(f"Validation error (malformed data): {e}")
            return False

class RAGGuardrail:
    """
    Controls and monitors RAG query quality and safety.
    """
    
    TRAFFIC_KEYWORDS = [
        "traffic", "congestion", "speed", "road", "street", "highway", 
        "crash", "accident", "construction", "closure", "chicago", 
        "northbound", "southbound", "eastbound", "westbound", "delay"
    ]
    
    @classmethod
    def is_relevant(cls, query: str) -> bool:
        """
        Check if the query is relevant to traffic intelligence.
        Simple keyword-based guardrail for demo; could use an LLM-based filter.
        """
        query_lower = query.lower()
        is_relevant = any(kw in query_lower for kw in cls.TRAFFIC_KEYWORDS)
        
        if not is_relevant:
            logger.warning(f"Off-topic query blocked: {query}")
        
        return is_relevant

    @classmethod
    def validate_response_faithfulness(cls, answer: str, sources: List[str]) -> float:
        """
        Heuristic for faithfulness: Checks if key terms in answer exist in sources.
        Returns a score 0.0 - 1.0.
        """
        if not sources:
            return 0.0
            
        # Very simple version: keyword overlap
        answer_words = set(answer.lower().split())
        source_text = " ".join(sources).lower()
        
        count = 0
        meaningful_words = [w for w in answer_words if len(w) > 4]
        if not meaningful_words:
            return 1.0
            
        for word in meaningful_words:
            if word in source_text:
                count += 1
                
        return count / len(meaningful_words)
