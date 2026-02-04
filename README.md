# Urban Traffic Intelligence

ML system combining RAG (Retrieval-Augmented Generation) and predictive models for Chicago traffic forecasting.

## Overview

This system provides intelligent traffic forecasting by combining:
- **Machine Learning Models**: LSTM, GRU, and Attention-based models for time series prediction
- **RAG System**: Retrieval-Augmented Generation for context-aware predictions using traffic domain knowledge
- **REST API**: FastAPI-based service for easy integration

## Features

### 1. Predictive Models
- LSTM, GRU, and Attention LSTM architectures
- Multi-horizon traffic speed forecasting
- Feature engineering with temporal, lag, and rolling statistics
- Supports training on historical Chicago traffic data

### 2. RAG System
- Vector database (ChromaDB) for traffic knowledge storage
- Semantic search using sentence transformers
- Domain knowledge about Chicago traffic patterns, rush hours, weather impacts, and special events
- Context-aware predictions combining ML outputs with relevant insights

### 3. Data Collection & Processing
- Chicago Data Portal integration
- Synthetic data generation for testing
- Comprehensive preprocessing pipeline
- Feature engineering for time series forecasting

### 4. REST API
- Endpoints for traffic predictions
- Knowledge base queries
- Adding custom traffic knowledge
- Demo data generation

## Installation

```bash
# Clone the repository
git clone https://github.com/Cygnus2505/urban-traffic-intelligence.git
cd urban-traffic-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
CHICAGO_DATA_PORTAL_TOKEN=your_chicago_data_portal_token
```

## Quick Start

### Run the Example

```bash
python example.py
```

This will:
1. Generate synthetic traffic data
2. Initialize the RAG system with traffic knowledge
3. Demonstrate knowledge base queries
4. Train a simple prediction model
5. Make context-aware predictions

### Start the API Server

```bash
python -m src.api.main
```

Or with uvicorn:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the API documentation at: http://localhost:8000/docs

## Usage Examples

### Python API

```python
from src.forecaster import HybridTrafficForecaster
from src.data.collector import ChicagoTrafficCollector

# Initialize forecaster with RAG enabled
forecaster = HybridTrafficForecaster(use_rag=True)

# Generate sample data
collector = ChicagoTrafficCollector()
data = collector.generate_synthetic_data(num_days=30)

# Query knowledge base
answer = forecaster.query_knowledge(
    "What are typical rush hour patterns in Chicago?"
)
print(answer)

# Train model
forecaster.train_model(
    train_data=data,
    model_type='lstm',
    epochs=50
)

# Make predictions
predictions = forecaster.predict(
    input_data=recent_data,
    location="I-90 Kennedy Expressway",
    include_context=True
)
```

### REST API

**Query Knowledge Base:**
```bash
curl -X POST "http://localhost:8000/query-knowledge" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does weather affect traffic?"}'
```

**Generate Demo Data:**
```bash
curl -X GET "http://localhost:8000/generate-demo-data"
```

**Get Knowledge Stats:**
```bash
curl -X GET "http://localhost:8000/knowledge-stats"
```

## Architecture

```
urban-traffic-intelligence/
├── src/
│   ├── config.py              # Configuration management
│   ├── forecaster.py          # Hybrid forecasting system
│   ├── data/
│   │   ├── collector.py       # Data collection
│   │   └── preprocessor.py    # Data preprocessing
│   ├── models/
│   │   ├── forecasting_models.py  # LSTM, GRU, Attention models
│   │   └── trainer.py         # Model training
│   ├── rag/
│   │   ├── vector_db.py       # Vector database management
│   │   └── rag_system.py      # RAG system
│   └── api/
│       └── main.py            # FastAPI application
├── tests/                     # Test files
├── config/                    # Configuration files
├── example.py                 # Example usage script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Components

### Data Collection
- Fetches traffic data from Chicago Data Portal API
- Generates synthetic data for testing
- Supports multiple traffic locations

### Preprocessing
- Temporal feature engineering (hour, day, cyclical encoding)
- Lag features for historical patterns
- Rolling statistics (mean, std, min, max)
- Sequence creation for time series models

### ML Models
- **LSTM**: Long Short-Term Memory for sequential patterns
- **GRU**: Gated Recurrent Unit, lighter alternative
- **Attention LSTM**: Enhanced with attention mechanism

### RAG System
- **Vector Database**: ChromaDB for efficient semantic search
- **Embeddings**: Sentence transformers for document encoding
- **Knowledge Base**: Pre-loaded with Chicago traffic insights
- **Query Interface**: Natural language questions

### Hybrid Forecasting
- Combines ML predictions with RAG context
- Explains predictions using relevant knowledge
- Multi-location forecasting support

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Make traffic predictions |
| `/query-knowledge` | POST | Query knowledge base |
| `/add-knowledge` | POST | Add custom knowledge |
| `/generate-demo-data` | GET | Generate synthetic data |
| `/knowledge-stats` | GET | Knowledge base statistics |

## Development

### Running Tests

```bash
pytest tests/
```

### Adding Custom Knowledge

```python
forecaster.add_traffic_knowledge(
    documents=[
        "New traffic pattern or insight...",
    ],
    metadatas=[
        {"category": "patterns", "type": "custom"}
    ]
)
```

## Data Sources

- **Chicago Data Portal**: Real-time traffic data
  - API: https://data.cityofchicago.org/resource/8v9j-bter.json
- **Synthetic Data**: Generated for testing and demonstration

## Performance Considerations

- **Model Training**: GPU recommended for large datasets
- **Vector Database**: Persisted to disk for efficiency
- **Batch Processing**: Configurable batch sizes for predictions
- **Caching**: Model predictions can be cached

## Limitations

- Requires historical data for accurate predictions
- OpenAI API key needed for advanced RAG features (optional)
- Model performance depends on data quality and quantity

## Future Enhancements

- [ ] Integration with real-time traffic feeds
- [ ] Multi-modal predictions (traffic, weather, events)
- [ ] Advanced ensemble methods
- [ ] Mobile app integration
- [ ] Real-time alerts and notifications
- [ ] Interactive visualization dashboard

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

- Chicago Data Portal for traffic data
- OpenAI for embedding models
- PyTorch team for deep learning framework