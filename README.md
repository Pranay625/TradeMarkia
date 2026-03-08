# Semantic Cache System for 20 Newsgroups

A semantic search system with fuzzy clustering and manual cache implementation for the 20 Newsgroups dataset.

## Overview

This project implements a semantic search system with three main components:
1. **Document embeddings** using sentence-transformers
2. **Fuzzy clustering** with Gaussian Mixture Models for probabilistic cluster assignments
3. **Semantic cache** built from scratch (no Redis) that recognizes similar queries

## Features

- Text preprocessing with deliberate cleaning choices
- Vector embeddings using `all-MiniLM-L6-v2` model
- Fuzzy clustering (GMM) - documents can belong to multiple clusters
- Manual semantic cache with cosine similarity matching
- Cluster-aware cache optimization for faster lookups
- FastAPI service with query, stats, and cache management endpoints
- Full Docker support

## Quick Start

### Prerequisites
- Python 3.11+
- 4GB RAM minimum

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Generate embeddings and train clustering (required first time, ~18 minutes)
python scripts/generate_embeddings.py
python scripts/train_clustering.py

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

### Docker Deployment

```bash
# Note: Generate models locally first (required)
python scripts/generate_embeddings.py
python scripts/train_clustering.py

# Start with Docker
docker compose up -d

# Check health
curl http://localhost:8000/health
```

## API Endpoints

### POST /query
Submit a natural language query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best graphics cards for gaming?"}'
```

Response:
```json
{
  "query": "What are the best graphics cards for gaming?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "result": "Generated response...",
  "dominant_cluster": 2,
  "processing_time_ms": 67.85
}
```

### GET /cache/stats
Get cache performance statistics:
```bash
curl http://localhost:8000/cache/stats
```

### DELETE /cache
Clear the cache:
```bash
curl -X DELETE http://localhost:8000/cache
```

## Design Decisions

### Part 1: Embeddings
- **Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Why**: Good balance between speed (~20 docs/sec) and semantic quality
- **Text Cleaning**: Removed email headers, quoted text, signatures, URLs - kept only semantic content

### Part 2: Fuzzy Clustering
- **Algorithm**: Gaussian Mixture Model (GMM)
- **Why**: Provides probabilistic membership - documents can belong to multiple clusters
- **Clusters**: 20 (matches dataset categories for interpretability)
- **Results**: 99.35% average confidence, only 0.01% truly uncertain documents

### Part 3: Semantic Cache
- **Implementation**: Manual in-memory cache using Python lists (no Redis/Memcached)
- **Similarity**: Cosine similarity between query embeddings
- **Threshold**: 0.85 (configurable) - tested from 0.70 to 0.95
- **Optimization**: Cluster-aware search reduces lookup time from O(n) to O(n/k)
- **Performance**: 2.6x faster on cache hits (~25ms vs ~70ms)

## Project Structure

```
semantic_cache_project/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   └── schemas.py         # Request/response models
├── src/                    # Core implementation
│   ├── data_loader.py     # Dataset loading
│   ├── text_cleaner.py    # Text preprocessing
│   ├── embedding_model.py # Embedding generation
│   ├── clustering.py      # Fuzzy clustering (GMM)
│   ├── semantic_cache.py  # Manual cache implementation
│   ├── query_engine.py    # Query processing pipeline
│   └── paths.py           # Path configuration
├── scripts/                # Setup and training scripts
│   ├── generate_embeddings.py
│   ├── train_clustering.py
│   └── preflight_check.py
├── tests/                  # Test suite
├── docs/                   # Additional documentation
├── models/                 # Generated models (not in git)
├── cache/                  # Runtime cache storage
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
└── requirements.txt       # Python dependencies
```

## Configuration

Edit `.env` or environment variables:
```bash
SIMILARITY_THRESHOLD=0.85    # Cache hit threshold
CACHE_MAX_SIZE=1000         # Max cached queries
N_CLUSTERS=20               # Number of clusters
LOG_LEVEL=INFO              # Logging level
```

## Testing

```bash
# Run API tests
python tests/test_api.py

# Run all tests
pytest tests/

# Verify setup
python scripts/preflight_check.py
```

## Performance Metrics

- Embedding generation: ~20 docs/sec
- Cache hit latency: ~25ms
- Cache miss latency: ~70ms
- Model size: 32MB total
- Cluster confidence: 99.35% average

## Technical Details

### Text Cleaning
The text cleaner removes:
- Email headers (From:, Subject:, etc.) - metadata, not content
- Quoted replies (lines starting with >) - duplicated content
- Signatures (-- delimiter) - boilerplate
- URLs and email addresses - not semantic content
- Excessive whitespace - formatting cleanup

### Fuzzy Clustering
Uses Gaussian Mixture Model to assign probabilistic cluster membership:
- Each document gets a probability distribution over all clusters
- Example: Document might be 60% cluster A, 30% cluster B, 10% cluster C
- Captures documents that span multiple topics (e.g., "computer graphics for medical imaging")

### Semantic Cache
Manual implementation without external libraries:
- Stores query embeddings and results in memory
- Computes cosine similarity for incoming queries
- Returns cached result if similarity >= threshold
- Uses cluster information to speed up search

## Additional Documentation

See `docs/` directory for detailed documentation:
- `CLUSTERING.md` - Clustering analysis and results
- `SEMANTIC_CACHE.md` - Cache implementation details
- `EMBEDDINGS.md` - Embedding model justification
- `DEPLOYMENT.md` - Deployment guide
- `PROJECT_STRUCTURE.md` - Architecture overview

## Requirements Met

✅ Part 1: Embedding & vector database with justified design choices  
✅ Part 2: Fuzzy clustering with probabilistic assignments and analysis  
✅ Part 3: Manual semantic cache with cluster optimization  
✅ Part 4: FastAPI service with all required endpoints  
✅ Bonus: Full Docker containerization  

## License

MIT
