# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API key authentication or OAuth2.

## Endpoints

### 1. Root Endpoint

Get API information and available endpoints.

**Request:**
```http
GET /
```

**Response:**
```json
{
  "message": "Chicago Traffic Forecasting API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "query": "/query-knowledge",
    "add_knowledge": "/add-knowledge",
    "generate_demo_data": "/generate-demo-data"
  }
}
```

---

### 2. Health Check

Check if the API is running and healthy.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "rag_enabled": true,
  "timestamp": "2024-02-04T12:30:00.000Z"
}
```

---

### 3. Make Traffic Prediction

Generate traffic speed predictions for a specific location.

**Request:**
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "location": "I-90 Kennedy Expressway",
  "include_context": true,
  "historical_data": [
    {
      "segment_id": 1,
      "timestamp": "2024-02-04T08:00:00",
      "speed": 45.5,
      "traffic_volume": 120,
      "street": "I-90",
      "direction": "NB"
    },
    ...
  ]
}
```

**Parameters:**
- `location` (string, required): Location identifier or name
- `include_context` (boolean, optional): Whether to include RAG context. Default: true
- `historical_data` (array, required): Array of historical traffic data points (minimum 200 points recommended)

**Response:**
```json
{
  "location": "I-90 Kennedy Expressway",
  "forecast_start": "2024-02-04T12:00:00",
  "predictions": [
    {
      "timestamp": "2024-02-04T13:00:00",
      "hour": 13,
      "predicted_speed": 42.3,
      "condition": "moderate traffic",
      "relevant_factors": ["rush_hour", "patterns"],
      "context": "Relevant traffic knowledge:\n1. Rush hour traffic..."
    },
    ...
  ],
  "explanation": "Normal traffic flow is expected with average speeds..."
}
```

**Error Responses:**

```json
{
  "detail": "Model not trained. Please train the model first or use demo mode."
}
```
Status: 503 Service Unavailable

```json
{
  "detail": "Error message"
}
```
Status: 500 Internal Server Error

---

### 4. Query Knowledge Base

Query the traffic knowledge base with a natural language question.

**Request:**
```http
POST /query-knowledge
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "What are typical rush hour patterns in Chicago?"
}
```

**Parameters:**
- `question` (string, required): Natural language question about traffic

**Response:**
```json
{
  "question": "What are typical rush hour patterns in Chicago?",
  "answer": "Based on traffic knowledge:\n• Rush hour traffic in Chicago typically occurs between 7-9 AM and 4-7 PM on weekdays...",
  "timestamp": "2024-02-04T12:30:00.000Z"
}
```

**Example Questions:**
- "How does weather affect traffic?"
- "What should I know about weekend traffic?"
- "When is the best time to travel on Lake Shore Drive?"
- "How do special events impact traffic?"

---

### 5. Add Traffic Knowledge

Add custom traffic knowledge to the RAG system.

**Request:**
```http
POST /add-knowledge
Content-Type: application/json
```

**Request Body:**
```json
{
  "documents": [
    "The Dan Ryan Expressway experiences heavy southbound traffic during Friday evening rush hours.",
    "Construction at the Jane Byrne Interchange has reduced capacity by 25%."
  ],
  "metadatas": [
    {
      "category": "patterns",
      "type": "dan_ryan",
      "source": "observation"
    },
    {
      "category": "factors",
      "type": "construction",
      "source": "official"
    }
  ]
}
```

**Parameters:**
- `documents` (array of strings, required): Traffic insights to add
- `metadatas` (array of objects, optional): Metadata for each document

**Metadata Fields:**
- `category`: Type of information (e.g., "patterns", "factors", "events")
- `type`: Specific sub-type
- `source`: Source of information
- `date`: Date of observation (if applicable)

**Response:**
```json
{
  "message": "Successfully added 2 documents",
  "timestamp": "2024-02-04T12:30:00.000Z"
}
```

---

### 6. Generate Demo Data

Generate synthetic traffic data for demonstration and testing.

**Request:**
```http
GET /generate-demo-data
```

**Response:**
```json
{
  "message": "Demo data generated successfully",
  "total_records": 5040,
  "sample": [
    {
      "segment_id": 1,
      "timestamp": "2024-01-28T00:00:00",
      "speed": 55.23,
      "traffic_volume": 45,
      "street": "Street_1",
      "direction": "SB",
      "hour": 0,
      "day_of_week": 6
    },
    ...
  ],
  "saved_to": "data/demo_traffic_data.csv"
}
```

**Query Parameters:**
- None

**Note:** This endpoint generates 7 days of hourly data for 5 locations (840 records per location).

---

### 7. Knowledge Base Statistics

Get statistics about the traffic knowledge base.

**Request:**
```http
GET /knowledge-stats
```

**Response:**
```json
{
  "total_documents": 15,
  "collection_name": "traffic_knowledge",
  "timestamp": "2024-02-04T12:30:00.000Z"
}
```

---

## Data Models

### TrafficDataPoint

```typescript
{
  segment_id: number;      // Traffic segment identifier
  timestamp: string;       // ISO 8601 datetime
  speed: number;          // Average speed in mph
  traffic_volume: number; // Number of vehicles
  street: string;         // Street name
  direction: string;      // Direction (NB, SB, EB, WB)
}
```

### Prediction

```typescript
{
  timestamp: string;         // ISO 8601 datetime
  hour: number;             // Hour of day (0-23)
  predicted_speed: number;  // Predicted speed in mph
  condition: string;        // Traffic condition description
  relevant_factors: string[]; // Relevant knowledge categories
  context?: string;         // RAG context (optional)
}
```

### ForecastResponse

```typescript
{
  location: string;           // Location identifier
  forecast_start: string;     // Start time (ISO 8601)
  predictions: Prediction[];  // Array of predictions
  explanation?: string;       // Overall explanation (optional)
}
```

---

## Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request parameters |
| 500 | Internal Server Error | Server error occurred |
| 503 | Service Unavailable | Service not ready (e.g., model not trained) |

---

## Rate Limiting

Currently, there is no rate limiting. For production use, consider implementing:
- Per-IP rate limiting
- API key-based quotas
- Request throttling

---

## CORS

The API currently allows all origins (`*`). For production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Code Examples

### Python

```python
import requests

# Query knowledge base
response = requests.post(
    "http://localhost:8000/query-knowledge",
    json={"question": "How does weather affect traffic?"}
)
print(response.json()["answer"])

# Generate demo data
response = requests.get("http://localhost:8000/generate-demo-data")
data = response.json()
print(f"Generated {data['total_records']} records")

# Add custom knowledge
response = requests.post(
    "http://localhost:8000/add-knowledge",
    json={
        "documents": ["New traffic insight..."],
        "metadatas": [{"category": "patterns"}]
    }
)
print(response.json()["message"])
```

### JavaScript

```javascript
// Query knowledge base
fetch('http://localhost:8000/query-knowledge', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    question: 'What are typical rush hour patterns?'
  })
})
.then(res => res.json())
.then(data => console.log(data.answer));

// Get knowledge stats
fetch('http://localhost:8000/knowledge-stats')
.then(res => res.json())
.then(data => console.log(data));
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Query knowledge
curl -X POST http://localhost:8000/query-knowledge \
  -H "Content-Type: application/json" \
  -d '{"question": "How does weather affect traffic?"}'

# Generate demo data
curl http://localhost:8000/generate-demo-data

# Get stats
curl http://localhost:8000/knowledge-stats
```

---

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Interactive testing
- Schema validation
- Example requests/responses
- Download OpenAPI spec

---

## Errors

### Common Error Patterns

**Model Not Trained:**
```json
{
  "detail": "Model not trained. Please train the model first or use demo mode."
}
```

**Invalid Input:**
```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**System Not Initialized:**
```json
{
  "detail": "System not initialized"
}
```

---

## Best Practices

1. **Batch Requests**: For multiple predictions, batch them when possible
2. **Cache Results**: Cache frequently requested knowledge queries
3. **Error Handling**: Always handle potential 500/503 errors
4. **Timeout**: Set reasonable timeouts for long-running predictions
5. **Data Quality**: Provide at least 200 historical data points for predictions

---

## Version History

### v1.0.0 (Current)
- Initial release
- Basic prediction endpoints
- RAG knowledge base
- Demo data generation

### Planned Features
- Model training via API
- Batch prediction endpoints
- WebSocket for real-time updates
- Authentication & authorization
- Advanced filtering options
