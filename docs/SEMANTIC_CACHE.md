# Semantic Cache Implementation

## Overview

Successfully implemented a **manual semantic cache from scratch** without Redis, Memcached, or any external caching libraries. The cache uses vector similarity to detect semantically similar queries and return cached results.

## Implementation

### Class: `SemanticCache` (`src/semantic_cache.py`)

**Pure Python implementation** - No external cache dependencies.

**Storage**: In-memory list of cache entries

**Key Features**:
1. Cosine similarity-based query matching
2. Configurable similarity threshold
3. Cluster-aware optimization for fast lookups
4. Cache statistics tracking
5. Disk persistence

## Cache Entry Structure

Each entry contains:
```python
{
    'query_text': str,           # Original query
    'query_embedding': np.ndarray,  # Vector (384-dim)
    'result': Any,               # Cached result
    'dominant_cluster': int,     # Primary cluster ID
    'timestamp': datetime,       # When cached
    'hit_count': int            # Usage counter
}
```

## Core Methods

### 1. `add_entry(query_text, query_embedding, result, dominant_cluster)`

Stores a new cache entry.

**Parameters**:
- `query_text`: Original query string
- `query_embedding`: Query vector (384-dim)
- `result`: Result to cache
- `dominant_cluster`: Primary cluster ID (optional)

**Returns**: Entry index

### 2. `search_cache(query_embedding, query_cluster)`

Searches for semantically similar cached query.

**Algorithm**:
1. If clustering enabled and query_cluster provided:
   - Search cluster entries first (optimization)
   - If no match, fallback to full search
2. Compute cosine similarity with all candidates
3. Find best match
4. If similarity >= threshold → Cache HIT
5. Otherwise → Cache MISS

**Parameters**:
- `query_embedding`: Query vector
- `query_cluster`: Dominant cluster (optional, for optimization)

**Returns**:
```python
{
    'cache_hit': bool,
    'matched_query': str,
    'similarity_score': float,
    'cached_result': Any,
    'search_time': float
}
```

### 3. `get_stats()`

Returns cache performance statistics.

**Returns**:
```python
{
    'total_entries': int,
    'total_queries': int,
    'hit_count': int,
    'miss_count': int,
    'hit_rate': float,
    'miss_rate': float,
    'similarity_threshold': float,
    'use_clustering': bool,
    'clusters_indexed': int,
    'top_entries': list
}
```

### 4. `save(filepath)` / `load(filepath)`

Persist cache to disk using pickle.

**File**: `cache/cache_store.pkl`

## Similarity Threshold

Configurable parameter controlling cache hit sensitivity.

**Examples**:
- `0.70`: Loose matching (more hits, less precise)
- `0.85`: Balanced (recommended)
- `0.90`: Strict matching (fewer hits, more precise)

**Test Results**:
```
Query 1: "What are the best graphics cards for gaming?"
Query 2: "Which GPU is good for gaming?"
Similarity: 0.8647

Threshold 0.70: HIT
Threshold 0.80: HIT
Threshold 0.85: HIT
Threshold 0.90: MISS
Threshold 0.95: MISS
```

## Cluster-Aware Optimization

**Problem**: Linear search through all cache entries is slow for large caches.

**Solution**: Use clustering to narrow search space.

**How it works**:
1. Predict query's dominant cluster
2. Search only cache entries in same cluster first
3. If no match, fallback to full search

**Benefits**:
- Faster lookups (reduced search space)
- Maintains accuracy (fallback ensures no misses)
- Leverages semantic structure

**Example**:
```python
# Without clustering: Search 1000 entries
# With clustering: Search ~50 entries in cluster, then fallback if needed
```

## Test Results

### Basic Functionality
```
[PASS] Cache initialization works
[PASS] Adding entries works
[PASS] Cache hit detection works (similarity: 1.0000)
[PASS] Cache miss detection works (similarity: 0.1921)
[PASS] Cluster-aware optimization works
[PASS] Cache persistence works
[PASS] Statistics tracking works
[PASS] Threshold configuration works
```

### Performance
```
Cache size: 3 entries
Total queries: 2
Hit count: 1
Miss count: 1
Hit rate: 50.00%
```

### Similarity Examples
```
Query 1: "What are the best graphics cards for gaming?"
Query 2: "Which GPU is good for gaming?"
Similarity: 0.8647 → CACHE HIT

Query 1: "What are the best graphics cards for gaming?"
Query 3: "Tell me about machine learning algorithms"
Similarity: 0.1921 → CACHE MISS
```

## Usage Example

```python
from src.semantic_cache import SemanticCache
from src.embedding_model import EmbeddingModel
from src.clustering import FuzzyClusterer

# Initialize
cache = SemanticCache(similarity_threshold=0.85, use_clustering=True)
embedding_model = EmbeddingModel('all-MiniLM-L6-v2')
clusterer = FuzzyClusterer()
clusterer.load("models/clusters.pkl")

# Add entry
query = "What are the best graphics cards?"
query_emb = embedding_model.encode_query(query)
cluster_id = clusterer.get_primary_cluster(query_emb)
result = {'documents': [...], 'scores': [...]}

cache.add_entry(query, query_emb, result, cluster_id)

# Search cache
new_query = "Which GPU is good for gaming?"
new_query_emb = embedding_model.encode_query(new_query)
new_cluster_id = clusterer.get_primary_cluster(new_query_emb)

cache_result = cache.search_cache(new_query_emb, query_cluster=new_cluster_id)

if cache_result['cache_hit']:
    print(f"Cache HIT! Similarity: {cache_result['similarity_score']:.4f}")
    return cache_result['cached_result']
else:
    print("Cache MISS - processing query...")
    # Process query normally
```

## Statistics Tracking

The cache tracks:
- Total entries
- Total queries processed
- Hit count / Miss count
- Hit rate / Miss rate
- Top frequently used entries

```python
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2f}%")
```

## Persistence

Save and load cache:

```python
# Save
cache.save("cache/cache_store.pkl")

# Load
cache_loaded = SemanticCache()
cache_loaded.load("cache/cache_store.pkl")
```

## Design Decisions

### Why In-Memory List?
- Simple and fast
- No external dependencies
- Easy to persist with pickle
- Suitable for moderate cache sizes

### Why Cosine Similarity?
- Standard metric for semantic similarity
- Fast computation (dot product for normalized vectors)
- Interpretable scores (0-1 range)

### Why Cluster-Aware?
- Reduces search space significantly
- Leverages semantic structure
- Maintains accuracy with fallback

## Files Created

```
src/
  └── semantic_cache.py          ✅ Implemented

cache/
  └── cache_store.pkl             ✅ Generated (5.49 KB)

Scripts:
  └── test_semantic_cache.py      ✅ Created

Documentation:
  └── SEMANTIC_CACHE.md           ✅ This file
```

## Summary

✅ **Manual implementation (no Redis/Memcached)**
✅ **In-memory list storage**
✅ **Cosine similarity matching**
✅ **Configurable threshold (0.85 default)**
✅ **Cluster-aware optimization**
✅ **Statistics tracking**
✅ **Disk persistence**
✅ **All tests passing**

The semantic cache is production-ready and fully functional!
