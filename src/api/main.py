"""
Urban Traffic Intelligence API - Main Application
"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from .routes.health import router as health_router, REQUEST_COUNT, REQUEST_LATENCY
from .routes.rag import router as rag_router
from .routes.predict import router as predict_router
from .routes.data import router as data_router
from ..config import get_settings
from ..database.models import init_database

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    - Startup: Initialize database tables
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting Urban Traffic Intelligence API...")
    try:
        init_database(settings.database_url)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Urban Traffic Intelligence API...")


# Create FastAPI application
app = FastAPI(
    title="Urban Traffic Intelligence API",
    description="""
    A production-grade ML system combining RAG and predictive models 
    for traffic forecasting, incident analysis, and explainable insights for Chicago.
    
    ## Features
    - **Ask Questions (RAG)**: Query traffic patterns with natural language
    - **Forecast (ML)**: Predict traffic conditions by segment
    - **Monitoring**: Model drift detection and RAG quality metrics
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware - allow dashboard and development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit dashboard
        "http://localhost:3000",  # Grafana
        "http://localhost:8000",  # API itself (for testing)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware to track request metrics for Prometheus.
    Records request count and latency for each endpoint.
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics (skip /metrics endpoint to avoid recursion)
    if request.url.path != "/metrics":
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
    
    return response


# Include routers
app.include_router(health_router)
app.include_router(rag_router, prefix="/rag")
app.include_router(predict_router, prefix="/models")
app.include_router(data_router, prefix="/data")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")


# Future routers will be added here:
# from .routes.rag import router as rag_router
# from .routes.predict import router as predict_router
# app.include_router(rag_router, prefix="/rag")
# app.include_router(predict_router, prefix="/predict")
