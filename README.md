---
title: Urban Traffic Intelligence
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Urban Traffic Intelligence Platform

The Urban Traffic Intelligence Platform is a comprehensive system designed to provide real-time traffic monitoring, incident analysis, and predictive forecasting for the city of Chicago. It integrates Retrieval-Augmented Generation (RAG) with machine learning models to deliver actionable insights and data-driven predictions.

## Core Capabilities

- **Traffic Analysis (RAG)**: Utilize natural language processing to query traffic patterns, incident causes, and historical trends.
- **Congestion Forecasting**: Generate forecasts for specific road segments using trained machine learning models.
- **Data Visualization**: Interactive map and charts for real-time traffic monitoring and trend analysis.
- **System Monitoring**: Performance tracking for both machine learning models and API health.

## System Architecture

The platform follows a modular architecture:

- **Frontend**: Streamlit-based dashboard for user interaction and visualization.
- **Backend API**: FastAPI service managing data retrieval, model predictions, and RAG operations.
- **AI/ML Layer**: 
  - XGBoost for congestion forecasting.
  - OpenAI GPT-4 and Qdrant for retrieval-augmented generation.
- **Storage Layer**: PostgreSQL for structured data; Qdrant for vector storage.
- **Data Pipeline**: Automated ingestion from Chicago Data Portal and weather services.

## Technical Configuration

### Prerequisites
- Python 3.11 or higher
- Docker and Docker Compose
- OpenAI API Key

### Initial Setup

1. **Repository Configuration**
   ```bash
   git clone https://github.com/Cygnus2505/urban-traffic-intelligence.git
   cd urban-traffic-intelligence
   ```

2. **Environment Variables**
   Create a `.env` file based on the provided `.env.example`. Ensure all database connections and API keys are specified.

3. **Service Deployment**
   ```bash
   docker-compose up -d
   ```

4. **Data Ingestion**
   Initialize the data pipeline to populate the databases:
   ```bash
   python -m src.ingestion.pipeline --days 7
   ```

## Application Access

| Component | Access URL |
|-----------|------------|
| Dashboard | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| Monitoring | http://localhost:3000 (Grafana) |

## Development and Testing

### Testing
Run the test suite using pytest:
```bash
pytest tests/
```

### Manual Ingestion
Specific data ranges can be ingested manually via the pipeline script:
```bash
python -m src.ingestion.pipeline --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

## Project Structure

- `src/`: Core application logic (API, Ingestion, Models, RAG).
- `dashboard/`: Streamlit application files.
- `data/`: Local storage for raw and processed datasets.
- `tests/`: Integration and unit tests.

## Acknowledgments
Data provided by the Chicago Data Portal.
