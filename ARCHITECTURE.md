# System Architecture

## Overview

The Chicago Traffic Forecasting System is a hybrid ML system that combines Retrieval-Augmented Generation (RAG) with predictive time series models to provide intelligent, context-aware traffic forecasts.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Applications                        │
│                   (Web, Mobile, APIs, Dashboards)                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FastAPI REST API                           │
│               (Prediction, Query, Knowledge Management)             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Hybrid Traffic Forecaster                         │
│                    (Core Integration Layer)                         │
└───────────────┬────────────────────────────────┬────────────────────┘
                │                                │
                ▼                                ▼
┌──────────────────────────┐      ┌──────────────────────────────────┐
│   ML Prediction Models   │      │      RAG System                  │
│  ┌──────────────────┐    │      │  ┌────────────────────────┐     │
│  │  LSTM Model      │    │      │  │  Vector Database       │     │
│  │  GRU Model       │    │      │  │  (ChromaDB)            │     │
│  │  Attention LSTM  │    │      │  │  - Traffic Knowledge   │     │
│  └──────────────────┘    │      │  │  - Historical Insights │     │
│                          │      │  └────────────────────────┘     │
│  ┌──────────────────┐    │      │                                  │
│  │  Model Trainer   │    │      │  ┌────────────────────────┐     │
│  │  - Training Loop │    │      │  │  Embedding Model       │     │
│  │  - Optimization  │    │      │  │  (Sentence Transformer)│     │
│  └──────────────────┘    │      │  └────────────────────────┘     │
└──────────────────────────┘      └──────────────────────────────────┘
                │                                │
                └────────────────┬───────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                            │
│  ┌───────────────┐  ┌──────────────────┐  ┌────────────────────┐   │
│  │  Data         │  │  Feature         │  │  Preprocessor      │   │
│  │  Collector    │  │  Engineering     │  │  - Normalization   │   │
│  │  - Chicago    │  │  - Temporal      │  │  - Sequence        │   │
│  │    Data API   │  │  - Lag Features  │  │    Creation        │   │
│  │  - Synthetic  │  │  - Rolling Stats │  │                    │   │
│  └───────────────┘  └──────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. FastAPI REST API

**Purpose:** Provides HTTP endpoints for external access to the system.

**Key Features:**
- RESTful API design
- Automatic OpenAPI documentation
- CORS support
- Async request handling

**Endpoints:**
- `/predict` - Make traffic predictions
- `/query-knowledge` - Query traffic knowledge base
- `/add-knowledge` - Add custom traffic insights
- `/generate-demo-data` - Generate synthetic data
- `/knowledge-stats` - Get knowledge base statistics

### 2. Hybrid Traffic Forecaster

**Purpose:** Core integration layer that combines ML predictions with RAG insights.

**Responsibilities:**
- Orchestrate predictions across models
- Combine ML outputs with retrieved context
- Manage model training pipeline
- Coordinate knowledge base updates

**Key Methods:**
- `predict()` - Generate forecasts with context
- `train_model()` - Train prediction models
- `query_knowledge()` - Query traffic insights
- `add_traffic_knowledge()` - Extend knowledge base

### 3. ML Prediction Models

**Architecture:**

#### LSTM (Long Short-Term Memory)
```
Input Layer → LSTM Layer 1 → LSTM Layer 2 → FC Layer → Output
(features)    (128 units)    (128 units)    (64→24)   (24 hours)
```

#### GRU (Gated Recurrent Unit)
```
Input Layer → GRU Layer 1 → GRU Layer 2 → FC Layer → Output
(features)    (128 units)   (128 units)   (64→24)   (24 hours)
```

#### Attention LSTM
```
Input → LSTM → Attention → Context Vector → FC → Output
           ↓       ↑
           └───────┘
```

**Training Pipeline:**
1. Sequence creation (168-hour lookback)
2. Feature normalization
3. Batch processing
4. Adam optimization
5. Learning rate scheduling
6. Early stopping

### 4. RAG System

**Components:**

#### Vector Database (ChromaDB)
- Stores traffic domain knowledge as embeddings
- Enables semantic similarity search
- Persistent storage
- Collection management

#### Embedding Model
- Sentence Transformers (all-MiniLM-L6-v2)
- Converts text to 384-dimensional vectors
- Fast inference
- Good semantic understanding

#### Knowledge Base
Pre-loaded with:
- Rush hour patterns
- Weather impacts
- Special events
- Construction effects
- Public transit interactions
- Route alternatives

**Query Flow:**
```
User Question
     ↓
Embed Query
     ↓
Similarity Search (Vector DB)
     ↓
Retrieve Top-K Documents
     ↓
Format Context
     ↓
Return to Forecaster
```

### 5. Data Processing Layer

#### Data Collector
- Fetches from Chicago Data Portal API
- Generates synthetic data for testing
- Handles rate limiting and errors
- Saves/loads data efficiently

#### Feature Engineering
**Temporal Features:**
- Hour, day, month (cyclical encoding)
- Weekend indicator
- Rush hour indicator
- Sine/cosine transformations

**Lag Features:**
- Previous 1, 2, 3 hours
- Previous day (24 hours)
- Previous week (168 hours)

**Rolling Statistics:**
- Mean, std, min, max
- Windows: 3, 6, 12, 24 hours

#### Preprocessor
- Handles missing values
- Normalizes features
- Creates sequences
- Manages scaling

## Data Flow

### Prediction Flow

```
1. Input Data (Historical Traffic)
         ↓
2. Preprocess & Feature Engineering
         ↓
3. Create Sequences (168 hours → 24 hours)
         ↓
4. Normalize Features
         ↓
5. ML Model Prediction
         ↓
6. Query RAG for Context (Parallel)
         ↓
7. Combine Prediction + Context
         ↓
8. Return Augmented Forecast
```

### Training Flow

```
1. Collect Historical Data
         ↓
2. Preprocess & Engineer Features
         ↓
3. Split Train/Validation
         ↓
4. Create Sequences
         ↓
5. Initialize Model & Optimizer
         ↓
6. Training Loop
   - Forward pass
   - Loss calculation
   - Backward pass
   - Parameter update
         ↓
7. Validation & Early Stopping
         ↓
8. Save Best Model
```

### Knowledge Query Flow

```
1. Natural Language Question
         ↓
2. Embed Question (Sentence Transformer)
         ↓
3. Vector Similarity Search
         ↓
4. Retrieve Top-K Documents
         ↓
5. Format & Return Answer
```

## Technology Stack

### Core ML/AI
- **PyTorch**: Deep learning framework
- **Sentence Transformers**: Text embeddings
- **ChromaDB**: Vector database
- **Scikit-learn**: Data preprocessing

### API & Web
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Requests**: HTTP client

### Optional/Future
- **OpenAI API**: For advanced LLM features
- **Prophet**: Additional forecasting
- **Statsmodels**: Statistical models

## Deployment Considerations

### Development
```bash
python -m src.api.main
# or
uvicorn src.api.main:app --reload
```

### Production
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Future)
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0"]
```

## Performance Characteristics

### Latency
- **API Response**: < 100ms (without model inference)
- **RAG Query**: 50-200ms (depending on collection size)
- **Model Prediction**: 10-50ms (CPU), <10ms (GPU)
- **End-to-End Forecast**: 100-300ms

### Throughput
- **API**: 100+ requests/second
- **RAG System**: 50+ queries/second
- **Model Inference**: 1000+ predictions/second (batched)

### Storage
- **Vector DB**: ~1MB per 1000 documents
- **Model Weights**: 5-20MB per model
- **Training Data**: Depends on dataset size

## Security Considerations

1. **API Keys**: Stored in environment variables
2. **Input Validation**: Pydantic models
3. **Rate Limiting**: Should be added for production
4. **CORS**: Configurable for security
5. **Data Privacy**: No PII in knowledge base

## Scalability

### Horizontal Scaling
- Multiple API instances behind load balancer
- Shared vector database
- Centralized model storage

### Vertical Scaling
- GPU for faster model training/inference
- Larger models for better accuracy
- More memory for bigger knowledge bases

## Future Enhancements

1. **Real-time Updates**: Stream processing for live data
2. **Multi-city Support**: Expand beyond Chicago
3. **Advanced LLM**: GPT-4 integration for explanations
4. **Visualization**: Interactive dashboards
5. **Mobile App**: Native iOS/Android apps
6. **Alerts**: Proactive notifications
7. **Ensemble Models**: Combine multiple approaches
8. **A/B Testing**: Compare model versions

## Monitoring & Observability

Recommended additions:
- Prometheus metrics
- Grafana dashboards
- Logging (structured JSON)
- Error tracking (Sentry)
- Performance monitoring
- Model drift detection
