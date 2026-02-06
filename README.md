---
title: Urban Traffic Intelligence
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Urban Traffic Intelligence Platform 🚦

[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=Streamlit)](https://chicago-traffic-ai.streamlit.app/)
[![API Status](https://img.shields.io/badge/Hugging%20Face-Active%20API-yellow?style=for-the-badge&logo=HuggingFace)](https://huggingface.co/spaces/Cygnus2505/UrbanTraffic)

The Urban Traffic Intelligence Platform is a comprehensive system designed to provide real-time traffic monitoring, incident analysis, and predictive forecasting for the city of Chicago. It integrates Retrieval-Augmented Generation (RAG) with machine learning models to deliver actionable insights and data-driven predictions.

## 🚀 Core Capabilities

- **Traffic Analysis (RAG)**: Natural language querying of traffic patterns and incident causes.
- **Congestion Forecasting**: Street-specific forecasts (24h horizon) using XGBoost.
- **Real-time Map**: Interactive topographical view with live congestion overlays.
- **System Guardrails**: Automated data validation and model drift monitoring.

## ☁️ Cloud Deployment (Free Tier)

This project is optimized for deployment on free-tier cloud services:

### 1. Database & Vector Store
- **Supabase (PostgreSQL)**: Host for structured traffic data. Use the **Session Pooler** (port 6543) for IPv4 compatibility.
- **Qdrant Cloud (Vector DB)**: Host for RAG document embeddings.

### 2. Backend API
- **Hugging Face Spaces**: Containerized deployment of the FastAPI backend.
- **Configuration**: Add your `.env` variables as "Secrets" in HF Space settings.

### 3. Frontend Dashboard
- **Streamlit Community Cloud**: Host for the interactive dashboard.
- **Connection**: Set `TRAFFIC_API_BASE_URL` in Streamlit Secrets to point to your HF Space.

## 🛠️ Local Development

1. **Setup Repository**
   ```bash
   git clone https://github.com/Cygnus2505/urban-traffic-intelligence.git
   cd urban-traffic-intelligence
   ```

2. **Environment Configuration**
   Copy `.env.example` to `.env` and fill in:
   - `OPENAI_API_KEY` (Required for RAG)
   - Database credentials (Local Postgres or Supabase)

3. **Run with Docker**
   ```bash
   docker-compose up -d
   ```

4. **Initialize Data**
   ```bash
   python -m src.ingestion.pipeline --days 7
   python scripts/push_to_cloud.py  # To sync local data to cloud
   ```

## 📊 Performance & Monitoring
- **Forecasting**: XGBoost Regressor with street-level sensitivity (R2: 0.33).
- **RAG Knowledge**: 20,467 processed documents including crashes and construction alerts.
- **Observability**: Prometheus metrics integrated for latency and drift tracking.

---
Data provided by the [Chicago Data Portal](https://data.cityofchicago.org/).
