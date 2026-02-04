# Quick Start Guide

This guide will help you get started with the Chicago Traffic Forecasting System quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Cygnus2505/urban-traffic-intelligence.git
cd urban-traffic-intelligence
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch (for deep learning models)
- ChromaDB (for vector database)
- Sentence Transformers (for embeddings)
- FastAPI (for REST API)
- And more...

### 4. Configure Environment Variables (Optional)

```bash
cp .env.example .env
# Edit .env and add your API keys if you have them
```

**Note:** The system works without API keys using default settings.

## Running the Example

The easiest way to see the system in action is to run the example script:

```bash
python example.py
```

This will:
1. Generate synthetic Chicago traffic data
2. Initialize the RAG knowledge base
3. Demonstrate knowledge queries
4. Train a simple prediction model
5. Make context-aware traffic predictions

## Starting the API Server

Start the FastAPI server:

```bash
python -m src.api.main
```

Or using uvicorn:

```bash
uvicorn src.api.main:app --reload
```

Access the interactive API documentation at: http://localhost:8000/docs

## API Usage Examples

### 1. Check System Health

```bash
curl http://localhost:8000/health
```

### 2. Generate Demo Data

```bash
curl http://localhost:8000/generate-demo-data
```

### 3. Query Knowledge Base

```bash
curl -X POST "http://localhost:8000/query-knowledge" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are typical rush hour patterns in Chicago?"}'
```

### 4. Get Knowledge Base Statistics

```bash
curl http://localhost:8000/knowledge-stats
```

## Python API Examples

### Basic Usage

```python
from src.forecaster import HybridTrafficForecaster
from src.data.collector import ChicagoTrafficCollector

# Initialize the forecasting system
forecaster = HybridTrafficForecaster(use_rag=True)

# Generate sample data
collector = ChicagoTrafficCollector()
data = collector.generate_synthetic_data(num_days=30, locations=10)

# Query the knowledge base
answer = forecaster.query_knowledge(
    "How does weather affect traffic in Chicago?"
)
print(answer)
```

### Training a Model

```python
# Train a prediction model
history = forecaster.train_model(
    train_data=data,
    model_type='lstm',  # or 'gru', 'attention_lstm'
    epochs=50
)
```

### Making Predictions

```python
# Make predictions with context
predictions = forecaster.predict(
    input_data=recent_data,
    location="I-90 Kennedy Expressway",
    include_context=True
)

print(f"Predictions for {predictions['location']}:")
for pred in predictions['predictions'][:5]:
    print(f"  {pred['timestamp']}: {pred['predicted_speed']:.1f} mph")
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with verbose output:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_rag_system.py -v
```

## Verifying Installation

Check that all Python files have valid syntax:

```bash
python verify_syntax.py
```

## Common Issues

### Module Not Found Errors

If you see `ModuleNotFoundError`, make sure you:
1. Activated your virtual environment
2. Installed all dependencies: `pip install -r requirements.txt`

### ChromaDB Issues

If you encounter ChromaDB errors, try:
```bash
pip install --upgrade chromadb
```

### PyTorch Installation

For CPU-only installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For GPU (CUDA) installation, visit: https://pytorch.org/get-started/locally/

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Read the Documentation**: Check `README.md` for detailed information
3. **Customize Knowledge**: Add your own traffic insights to the RAG system
4. **Train Models**: Use real Chicago traffic data for better predictions
5. **Build Applications**: Integrate the API into your traffic applications

## Getting Help

- Check the main `README.md` for detailed documentation
- Review example code in `example.py`
- Examine test files in `tests/` for usage patterns
- Open an issue on GitHub for bugs or questions

## License

MIT License - See LICENSE file for details
