# Urban Traffic Intelligence Platform

A production-grade ML system that combines **RAG (Retrieval-Augmented Generation)** and **predictive models** to provide traffic forecasting, incident analysis, and explainable insights for Chicago.

##  Features

- **Ask Questions (RAG)**: Query traffic patterns with natural language
  - "Why is congestion high near downtown today?"
  - "What were the top incident causes last week?"
  
- **Forecast + Charts (ML)**: Predict traffic conditions
  - Congestion forecasting by zone/segment
  - Travel time predictions
  - Trend visualization

- **Monitoring & Alerts**: Production-grade observability
  - Model drift detection
  - RAG quality metrics (hallucination rate, citation coverage)
  - System health dashboards

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface                               │
│                 (Streamlit Dashboard)                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Service                             │
│    /rag/ask    /predict    /charts    /metrics    /health       │
└──────┬──────────────┬──────────────┬────────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌────────────┐  ┌───────────┐  ┌──────────────┐
│  RAG       │  │   ML      │  │  Monitoring  │
│  Pipeline  │  │  Models   │  │  & Metrics   │
└──────┬─────┘  └─────┬─────┘  └──────┬───────┘
       │              │               │
       ▼              ▼               ▼
┌────────────┐  ┌───────────┐  ┌──────────────┐
│  Qdrant    │  │  MLflow   │  │  Prometheus  │
│ (Vectors)  │  │ (Models)  │  │  + Grafana   │
└────────────┘  └───────────┘  └──────────────┘
       │              │
       └──────┬───────┘
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PostgreSQL                                  │
│   Traffic Data │ Incidents │ Weather │ Documents │ Logs         │
└─────────────────────────────────────────────────────────────────┘
              ▲
              │
┌─────────────────────────────────────────────────────────────────┐
│                   Ingestion Pipeline                             │
│    Chicago Data Portal │ OpenWeatherMap │ Scheduled Tasks       │
└─────────────────────────────────────────────────────────────────┘
```

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| API | FastAPI |
| Database | PostgreSQL |
| Vector DB | Qdrant |
| ML Framework | XGBoost, scikit-learn |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Experiment Tracking | MLflow |
| Orchestration | Prefect |
| Monitoring | Prometheus + Grafana |
| Dashboard | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |

##  Data Sources

| Source | Type | Description |
|--------|------|-------------|
| Chicago Traffic Tracker | Structured | Historical congestion by segment |
| Traffic Crashes | Structured + Text | Incident records with descriptions |
| Road Construction | Structured + Text | Active roadwork permits |
| OpenWeatherMap | Structured | Current + forecast weather |

##  Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- OpenAI API key (for embeddings/LLM)
- Chicago Data Portal app token (optional, increases rate limits)

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/urban-traffic-intelligence.git
cd urban-traffic-intelligence

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
```

### 2. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 3. Initialize Database & Ingest Data

```bash
# Run ingestion pipeline
docker-compose exec api python -m src.ingestion.pipeline --days 7
```

### 4. Access Services

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

##  Project Structure

```
urban-traffic-intelligence/
├── src/
│   ├── ingestion/          # Data ingestion modules
│   │   ├── traffic_data.py
│   │   ├── incidents.py
│   │   ├── weather.py
│   │   ├── construction.py
│   │   └── pipeline.py
│   ├── features/           # Feature engineering
│   │   └── feature_pipeline.py
│   ├── models/             # ML model training
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── rag/                # RAG pipeline
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── guardrails/         # Quality controls
│   │   └── validators.py
│   ├── api/                # FastAPI service
│   │   ├── main.py
│   │   └── routes/
│   ├── monitoring/         # Drift & metrics
│   │   ├── drift.py
│   │   └── metrics.py
│   ├── database/           # Database models
│   └── config.py
├── dashboard/              # Streamlit app
├── tests/                  # Test suite
├── data/                   # Data storage
├── .github/workflows/      # CI/CD
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

##  Development

### Local Setup (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Qdrant (via Docker)
docker-compose up -d postgres qdrant

# Run API locally
uvicorn src.api.main:app --reload
```

### Running Tests

```bash
pytest tests/ -v
```

### Ingestion Commands

```bash
# Full ingestion (last 7 days)
python -m src.ingestion.pipeline

# Custom date range
python -m src.ingestion.pipeline --start-date 2025-01-01 --end-date 2025-01-31

# Skip specific sources
python -m src.ingestion.pipeline --skip-weather --skip-construction

# Use synthetic weather (for testing)
python -m src.ingestion.pipeline --synthetic-weather
```

##  API Endpoints

### RAG

```bash
# Ask a question
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is there congestion on I-90?"}'
```

### Predictions

```bash
# Get traffic forecast
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"segment_id": "123", "horizon_hours": 6}'
```

### Charts

```bash
# Get chart data
curl "http://localhost:8000/charts/zone?zone_id=5&days=30"
```

##  Learning Outcomes

Building this project teaches:

- **MLOps**: Model versioning, experiment tracking, automated retraining
- **RAG**: Document chunking, vector search, prompt engineering, guardrails
- **Data Engineering**: ETL pipelines, data quality, feature engineering
- **Production ML**: API design, monitoring, drift detection
- **DevOps**: Docker, CI/CD, infrastructure as code

##  License

MIT License - see [LICENSE](LICENSE) for details.

##  Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
