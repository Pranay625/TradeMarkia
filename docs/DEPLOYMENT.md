# Deployment Guide

## Prerequisites

1. **Generate Models** (must be done before Docker build):
```bash
# Generate embeddings
python generate_embeddings.py

# Train clustering
python train_clustering.py
```

This creates:
- `models/embeddings.pkl` (29.33 MB)
- `models/clusters.pkl` (3.44 MB)

## Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build image
docker build -t semantic-cache-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/cache:/app/cache \
  -e SIMILARITY_THRESHOLD=0.85 \
  --name semantic-cache-api \
  semantic-cache-api

# View logs
docker logs -f semantic-cache-api

# Stop
docker stop semantic-cache-api
docker rm semantic-cache-api
```

## Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
# Paths
BASE_DIR=/app
MODELS_DIR=/app/models
CACHE_DIR=/app/cache

# Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
N_CLUSTERS=20

# Cache
SIMILARITY_THRESHOLD=0.85
CACHE_MAX_SIZE=1000
USE_CLUSTERING=true

# API
API_PORT=8000
LOG_LEVEL=INFO
```

## Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the best graphics cards?"}'

# Check stats
curl http://localhost:8000/cache/stats
```

## Troubleshooting

### Models not found
```
Error: Required file missing: /app/models/embeddings.pkl
```
**Solution**: Generate models before Docker build:
```bash
python generate_embeddings.py
python train_clustering.py
```

### Port already in use
```
Error: bind: address already in use
```
**Solution**: Change port in docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Out of memory
**Solution**: Reduce cache size in `.env`:
```
CACHE_MAX_SIZE=500
```

## Production Considerations

1. **Use reverse proxy** (nginx/traefik)
2. **Add authentication** (API keys)
3. **Enable HTTPS**
4. **Set up monitoring** (Prometheus/Grafana)
5. **Configure log aggregation**
6. **Set resource limits**:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

## Scaling

For multiple instances:
```yaml
services:
  semantic-cache-api:
    deploy:
      replicas: 3
```

Note: Cache is per-instance. For shared cache, implement Redis backend.
