# Docker Deployment Guide

## Prerequisites

**IMPORTANT**: Models must be generated BEFORE Docker deployment.

```bash
# 1. Install dependencies locally
pip install -r requirements.txt

# 2. Generate models (one-time, ~18 minutes)
python scripts/generate_embeddings.py
python scripts/train_clustering.py

# 3. Verify models exist
ls -lh models/
# Should show:
# embeddings.pkl (~29 MB)
# clusters.pkl (~3 MB)
```

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f semantic-cache-api

# Stop
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build image
docker build -t semantic-cache-api:latest .

# Run container
docker run -d \
  --name semantic-cache-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/cache:/app/cache \
  -e SIMILARITY_THRESHOLD=0.85 \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \
  semantic-cache-api:latest

# View logs
docker logs -f semantic-cache-api

# Stop and remove
docker stop semantic-cache-api
docker rm semantic-cache-api
```

## Verification

### 1. Check Container Status

```bash
# Docker Compose
docker-compose ps

# Docker CLI
docker ps | grep semantic-cache-api
```

### 2. Check Health

```bash
# Wait 60 seconds for startup, then:
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "query_engine": "operational",
    "semantic_cache": "operational",
    "embedding_model": "operational",
    "clustering": "operational"
  }
}
```

### 3. Test Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the best graphics cards for gaming?"}'

# Expected response:
{
  "query": "What are the best graphics cards for gaming?",
  "cache_hit": false,
  "similarity_score": 0.0,
  "result": "Generated response for: What are the best graphics cards for gaming?",
  "dominant_cluster": 2,
  "processing_time_ms": 67.85
}
```

### 4. Test Cache Hit

```bash
# Send similar query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Which GPU is good for gaming?"}'

# Expected response:
{
  "query": "Which GPU is good for gaming?",
  "cache_hit": true,
  "matched_query": "What are the best graphics cards for gaming?",
  "similarity_score": 0.8647,
  "result": "Generated response for: What are the best graphics cards for gaming?",
  "dominant_cluster": 2,
  "processing_time_ms": 25.66
}
```

### 5. Check Cache Stats

```bash
curl http://localhost:8000/cache/stats

# Expected response:
{
  "total_entries": 1,
  "total_queries": 2,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 50.0,
  "miss_rate": 50.0,
  "similarity_threshold": 0.85,
  "use_clustering": true
}
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs semantic-cache-api

# Common issues:
# 1. Models not found
#    Solution: Run scripts/generate_embeddings.py and scripts/train_clustering.py

# 2. Port 8000 already in use
#    Solution: Change port in docker-compose.yml:
#    ports:
#      - "8001:8000"

# 3. Permission denied on volumes
#    Solution: Check folder permissions:
#    chmod -R 755 models cache
```

### Health check failing

```bash
# Check if API is responding
docker exec semantic-cache-api curl http://localhost:8000/health

# Check startup logs
docker-compose logs semantic-cache-api | grep -i error
```

### Models not loading

```bash
# Verify volume mounts
docker inspect semantic-cache-api | grep -A 10 Mounts

# Check files inside container
docker exec semantic-cache-api ls -lh /app/models/

# Should show:
# embeddings.pkl
# clusters.pkl
```

## Environment Variables

Override in docker-compose.yml or pass via -e:

```yaml
environment:
  # Cache behavior
  - SIMILARITY_THRESHOLD=0.85    # 0.7-0.95 recommended
  - CACHE_MAX_SIZE=1000
  - USE_CLUSTERING=true
  
  # Model
  - EMBEDDING_MODEL=all-MiniLM-L6-v2
  - N_CLUSTERS=20
  
  # API
  - API_PORT=8000
  - LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
```

## Performance

### Resource Usage

```bash
# Check container stats
docker stats semantic-cache-api

# Typical usage:
# CPU: 5-10% (idle), 50-80% (processing)
# Memory: 1.5-2.5 GB
# Disk: ~1.2 GB (image) + 32 MB (models)
```

### Optimization

```yaml
# Add resource limits in docker-compose.yml:
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

## Production Deployment

### 1. Use Production Compose File

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  semantic-cache-api:
    image: semantic-cache-api:latest
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./cache:/app/cache
    environment:
      - LOG_LEVEL=WARNING
      - SIMILARITY_THRESHOLD=0.85
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### 2. Add Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Enable HTTPS

Use Let's Encrypt with certbot or configure SSL in nginx.

### 4. Set Up Monitoring

```bash
# Add Prometheus metrics endpoint
# Add health check monitoring
# Set up log aggregation (ELK stack)
```

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi semantic-cache-api:latest

# Remove volumes (WARNING: deletes cache)
docker-compose down -v

# Clean up Docker system
docker system prune -a
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t semantic-cache-api:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          docker tag semantic-cache-api:${{ github.sha }} registry.example.com/semantic-cache-api:latest
          docker push registry.example.com/semantic-cache-api:latest
```

## Support

For issues, check:
1. Container logs: `docker-compose logs`
2. Health endpoint: `curl http://localhost:8000/health`
3. API docs: `http://localhost:8000/docs`
