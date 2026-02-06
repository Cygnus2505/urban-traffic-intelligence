"""
Health and Metrics Endpoints
"""
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from prometheus_client import (
    generate_latest, 
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge
)
from datetime import datetime

from ...config import get_settings
from ...monitoring.drift import DriftDetector

router = APIRouter(tags=["Health & Monitoring"])

settings = get_settings()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

DB_CONNECTION_STATUS = Gauge(
    "database_connection_status",
    "Database connection status (1=connected, 0=disconnected)"
)


def check_database_connection() -> dict:
    """Check if database is reachable"""
    try:
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        DB_CONNECTION_STATUS.set(1)
        return {"status": "connected", "latency_ms": None}
    except ModuleNotFoundError as e:
        DB_CONNECTION_STATUS.set(0)
        return {"status": "driver_not_installed", "error": str(e)}
    except SQLAlchemyError as e:
        DB_CONNECTION_STATUS.set(0)
        return {"status": "disconnected", "error": str(e)}


@router.get("/health")
async def health_check():
    """
    Health check endpoint - verifies API is running and database is reachable.
    
    Returns:
        - status: "healthy" or "degraded"
        - timestamp: current UTC time
        - database: connection status
        - version: API version
    """
    db_status = check_database_connection()
    
    overall_status = "healthy" if db_status["status"] == "connected" else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": db_status
        }
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/monitoring/drift")
async def get_model_drift():
    """
    Check for statistical drift in traffic data.
    """
    detector = DriftDetector()
    report = detector.calculate_drift()
    return report
