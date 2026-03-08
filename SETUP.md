# Setup Guide

## Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate models** (required, takes ~18 minutes)
   ```bash
   python scripts/generate_embeddings.py
   python scripts/train_clustering.py
   ```

3. **Verify setup**
   ```bash
   python scripts/preflight_check.py
   ```

4. **Start API**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

## Docker Setup

```bash
# Generate models first (must be done locally)
python scripts/generate_embeddings.py
python scripts/train_clustering.py

# Start with Docker
docker compose up -d

# Check status
curl http://localhost:8000/health
```

## Testing

```bash
# Test API
python tests/test_api.py

# Or use curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best graphics cards?"}'
```

## Configuration

Copy `.env.example` to `.env` and adjust settings:
- `SIMILARITY_THRESHOLD`: Cache hit threshold (default: 0.85)
- `N_CLUSTERS`: Number of clusters (default: 20)
- `CACHE_MAX_SIZE`: Maximum cache entries (default: 1000)
