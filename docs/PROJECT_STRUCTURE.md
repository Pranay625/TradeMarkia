# Semantic Cache Project - Complete Structure

## 📁 Directory Tree

```
semantic_cache_project/
│
├── data/
│   └── 20_newsgroups/          # Dataset storage (auto-downloaded)
│
├── src/                         # Core source code
│   ├── __init__.py
│   ├── data_loader.py          # Load 20 Newsgroups dataset
│   ├── text_cleaner.py         # Text preprocessing
│   ├── embedding_model.py      # Generate embeddings (sentence-transformers)
│   ├── clustering.py           # Fuzzy clustering implementation
│   ├── semantic_cache.py       # ⭐ Manual semantic cache (core component)
│   ├── query_engine.py         # Query orchestration pipeline
│   └── utils.py                # Helper functions
│
├── api/                         # FastAPI REST API
│   ├── __init__.py
│   ├── main.py                 # API endpoints
│   └── schemas.py              # Pydantic request/response models
│
├── models/                      # Saved models and embeddings
│   ├── document_embeddings.npz # Precomputed embeddings
│   └── clusters.pkl            # Trained clustering model
│
├── cache/                       # Persistent cache storage (optional)
│
├── config.py                    # Central configuration
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
└── PROJECT_STRUCTURE.md         # This file
```

## 🔄 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI (api/main.py)                         │
│  POST /query → Receives natural language query                   │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Query Engine (src/query_engine.py)                  │
│  Orchestrates the entire processing pipeline                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌────────┴────────┐
                    ↓                 ↓
        ┌───────────────────┐  ┌──────────────────┐
        │  Text Cleaner     │  │ Embedding Model  │
        │  (text_cleaner)   │→ │ (embedding_model)│
        └───────────────────┘  └─────────┬────────┘
                                         ↓
                              ┌──────────────────────┐
                              │  Query Embedding     │
                              │  (384-dim vector)    │
                              └──────────┬───────────┘
                                         ↓
┌─────────────────────────────────────────────────────────────────┐
│           Semantic Cache (src/semantic_cache.py) ⭐              │
│                                                                  │
│  1. Compute cosine similarity with all cached queries            │
│  2. If max_similarity >= threshold (0.85):                       │
│     → CACHE HIT → Return cached result                           │
│  3. Else:                                                        │
│     → CACHE MISS → Continue processing                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                      [CACHE MISS PATH]
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Document Search & Ranking                           │
│                                                                  │
│  1. Load precomputed document embeddings                         │
│  2. Compute similarity: query_vec × doc_vecs                     │
│  3. Apply clustering filter (optional)                           │
│  4. Rank by similarity score                                     │
│  5. Return top-K documents                                       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Store in Cache                                │
│  cache[query_hash] = {embedding, result, timestamp, ...}         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Return Response                               │
│  {results, cache_hit, similarity, processing_time}               │
└─────────────────────────────────────────────────────────────────┘
```

## 🧩 Component Interactions

### Initialization Phase (Startup)
```
1. data_loader.py → Load 20 Newsgroups dataset
2. text_cleaner.py → Clean all documents
3. embedding_model.py → Generate embeddings for all documents
4. clustering.py → Perform fuzzy clustering on embeddings
5. Save embeddings and clusters to models/
```

### Query Phase (Runtime)
```
1. API receives query → query_engine.process_query()
2. Clean query text → text_cleaner.clean()
3. Generate query embedding → embedding_model.encode()
4. Check cache → semantic_cache.get()
   ├─ HIT: Return cached result immediately
   └─ MISS: Continue to step 5
5. Search documents → query_engine.search_documents()
6. Store in cache → semantic_cache.set()
7. Return results to user
```

## 🎯 Key Design Decisions

### 1. Manual Semantic Cache (No External Libraries)
- **Why**: Full control over similarity logic and eviction policies
- **How**: In-memory dictionary with numpy for vector operations
- **Benefits**: No external dependencies, customizable, transparent

### 2. Fuzzy Clustering
- **Why**: Documents can belong to multiple topics
- **How**: Fuzzy C-Means algorithm
- **Benefits**: Richer semantic structure than hard clustering

### 3. Sentence Transformers for Embeddings
- **Why**: State-of-the-art semantic embeddings
- **Model**: all-MiniLM-L6-v2 (fast, 384 dims)
- **Benefits**: High quality, pre-trained, efficient

### 4. Cosine Similarity for Cache Matching
- **Why**: Standard metric for semantic similarity
- **Threshold**: 0.85 (configurable)
- **Benefits**: Fast computation, interpretable scores

### 5. FastAPI for REST API
- **Why**: Modern, fast, auto-documentation
- **Benefits**: Async support, type validation, OpenAPI

## 📊 Performance Characteristics

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Embedding Generation | O(n) | O(n × d) |
| Cache Lookup | O(m) | O(m × d) |
| Document Search | O(n) | O(n × d) |
| Clustering | O(k × n × i) | O(k × d) |

Where:
- n = number of documents
- m = number of cached queries
- d = embedding dimension (384)
- k = number of clusters (20)
- i = clustering iterations

## 🚀 Next Steps

After this skeleton is complete, implement in this order:

1. **utils.py** - Helper functions (needed by all modules)
2. **data_loader.py** - Load dataset
3. **text_cleaner.py** - Text preprocessing
4. **embedding_model.py** - Generate embeddings
5. **clustering.py** - Fuzzy clustering
6. **semantic_cache.py** - Core cache logic ⭐
7. **query_engine.py** - Orchestration
8. **api/schemas.py** - API models
9. **api/main.py** - API endpoints
10. **Testing & Integration**

## 📝 Configuration

All settings are centralized in `config.py`:
- Embedding model selection
- Cache parameters (threshold, size, TTL)
- Clustering parameters
- API settings
- File paths

## 🔒 Production Considerations

- **Thread Safety**: Semantic cache uses threading.Lock
- **Error Handling**: Try-catch blocks in all API endpoints
- **Logging**: Structured logging with loguru
- **Validation**: Pydantic schemas for all API I/O
- **Monitoring**: Cache statistics endpoint
- **Scalability**: Precompute embeddings, batch processing
