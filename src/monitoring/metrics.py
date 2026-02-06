"""
Custom Monitoring Metrics
Defines Prometheus metrics for RAG and ML performance.
"""
from prometheus_client import Histogram, Counter, Gauge

# --- RAG Metrics ---
RAG_QUERY_TOTAL = Counter(
    "rag_queries_total",
    "Total number of RAG queries",
    ["status", "is_relevant"]
)

RAG_FAITHFULNESS_SCORE = Histogram(
    "rag_faithfulness_score",
    "Faithfulness score of RAG responses",
    buckets=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# --- ML Forecasting Metrics ---
PREDICTION_LATENCY = Histogram(
    "traffic_prediction_latency_seconds",
    "Latency of traffic congestion predictions in seconds"
)

PREDICTION_VALUE = Histogram(
    "traffic_prediction_value",
    "Distribution of predicted congestion levels",
    buckets=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
)

MODEL_DRIFT_SCORE = Gauge(
    "traffic_model_drift_p_value",
    "P-value from model drift detection (lower = more drift)"
)

# --- Data Quality Metrics ---
DATA_VALIDATION_ERRORS = Counter(
    "data_validation_errors_total",
    "Total records rejected by guardrails",
    ["source"]
)
