# Fuzzy Clustering Implementation

## Overview

Successfully implemented fuzzy clustering using Gaussian Mixture Model (GMM) from scikit-learn. Documents have probabilistic cluster membership rather than hard assignments.

## Why Fuzzy Clustering?

**Problem with Hard Clustering:**
- Documents often span multiple topics
- Example: "Computer graphics for medical imaging" belongs to both computer graphics AND medicine
- Hard clustering forces a single assignment, losing semantic richness

**Benefits of Fuzzy Clustering:**
1. **Probabilistic Membership**: Documents can belong to multiple clusters with different probabilities
2. **Semantic Ambiguity**: Captures documents that span multiple topics
3. **Uncertainty Detection**: Identifies documents with unclear categorization
4. **Richer Representation**: Provides nuanced similarity comparisons
5. **Better Search**: Enables cluster-based filtering with soft boundaries

## Implementation

### Class: `FuzzyClusterer` (`src/clustering.py`)

**Algorithm**: Gaussian Mixture Model (GMM)
- Assumes data comes from mixture of Gaussian distributions
- Each cluster is a Gaussian with mean and covariance
- Soft assignments via posterior probabilities

**Key Methods**:

1. **`fit(embeddings)`**
   - Trains GMM on document embeddings
   - Computes probability distributions for all documents
   - Determines dominant cluster per document

2. **`predict(embeddings)`**
   - Returns probability matrix (n_samples × n_clusters)
   - Each row sums to 1.0
   - Represents cluster membership probabilities

3. **`get_cluster_centers()`**
   - Returns cluster centroids (means)
   - Shape: (n_clusters, embedding_dim)

4. **`get_primary_cluster(embedding)`**
   - Returns dominant cluster (argmax of probabilities)

5. **`get_cluster_distribution(doc_idx)`**
   - Returns dict mapping cluster_id → probability

6. **`show_cluster_samples(cluster_id, documents, top_k)`**
   - Displays sample documents from a cluster
   - Shows top documents by probability
   - Useful for cluster interpretation

7. **`show_uncertain_documents(documents, threshold, top_k)`**
   - Finds documents with max_probability < threshold
   - These span multiple topics
   - Useful for quality analysis

8. **`save(filepath)` / `load(filepath)`**
   - Persist/load clustering results

## Results

### Training Statistics

- **Documents**: 19,997
- **Clusters**: 20
- **Training time**: 79.19 seconds
- **Convergence**: Yes (23 iterations)
- **Average confidence**: 0.9935 (99.35%)

### Cluster Sizes

Cluster distribution (by dominant cluster):

```
Cluster  0:  1,006 documents ( 5.03%)
Cluster  1:    902 documents ( 4.51%)
Cluster  2:  1,292 documents ( 6.46%)
Cluster  3:    764 documents ( 3.82%)
Cluster  4:    569 documents ( 2.85%)
Cluster  5:    973 documents ( 4.87%)
Cluster  6:    310 documents ( 1.55%)
Cluster  7:    941 documents ( 4.71%)
Cluster  8:    621 documents ( 3.11%)
Cluster  9:    862 documents ( 4.31%)
Cluster 10:    782 documents ( 3.91%)
Cluster 11:  1,358 documents ( 6.79%)
Cluster 12:    928 documents ( 4.64%)
Cluster 13:    961 documents ( 4.81%)
Cluster 14:  1,254 documents ( 6.27%)
Cluster 15:  1,993 documents ( 9.97%)
Cluster 16:    988 documents ( 4.94%)
Cluster 17:  1,336 documents ( 6.68%)
Cluster 18:  1,219 documents ( 6.10%)
Cluster 19:    938 documents ( 4.69%)
```

### Confidence Distribution

```
0.0 < confidence ≤ 0.3:      0 documents ( 0.00%)
0.3 < confidence ≤ 0.5:      2 documents ( 0.01%)  ← Uncertain
0.5 < confidence ≤ 0.7:    134 documents ( 0.67%)
0.7 < confidence ≤ 0.9:    276 documents ( 1.38%)
0.9 < confidence ≤ 1.0: 19,585 documents (97.94%)  ← High confidence
```

**Observations**:
- 97.94% of documents have high confidence (>0.9)
- Only 2 documents (0.01%) are truly uncertain (<0.5)
- GMM provides very confident cluster assignments

## Example: Fuzzy Membership

### High Confidence Document
```
Document: alt.atheism
Max probability: 1.0000

Cluster distribution:
  Cluster 18: 1.0000  ← Dominant
  Cluster  3: 0.0000
  Cluster 11: 0.0000
```
**Interpretation**: Clearly belongs to one cluster

### Low Confidence Document (Uncertain)
```
Document: alt.atheism
Max probability: 0.4530

Cluster distribution:
  Cluster 14: 0.4530  ← Slightly dominant
  Cluster  4: 0.4114  ← Almost equal
  Cluster 18: 0.1355
```
**Interpretation**: Spans multiple topics, ambiguous categorization

## Technical Details

### Gaussian Mixture Model

**Parameters**:
- `n_components`: 20 (number of clusters)
- `covariance_type`: 'diag' (diagonal covariance for efficiency)
- `max_iter`: 150
- `n_init`: 10 (number of initializations)
- `random_state`: 42 (reproducibility)

**Algorithm**:
1. Initialize cluster parameters (means, covariances, weights)
2. E-step: Compute posterior probabilities (cluster memberships)
3. M-step: Update parameters based on memberships
4. Repeat until convergence

**Probability Computation**:
```
P(cluster_k | document_i) = P(document_i | cluster_k) × P(cluster_k) / P(document_i)
```

### Storage Format

**File**: `models/clusters.pkl` (3.44 MB)

**Structure**:
```python
{
    'model': GaussianMixture,           # Trained GMM
    'cluster_probs': np.ndarray,        # Shape: (19997, 20)
    'dominant_clusters': np.ndarray,    # Shape: (19997,)
    'n_clusters': 20,
    'is_fitted': True
}
```

## Usage Examples

### Train Clustering
```bash
python train_clustering.py
```

### Load and Use
```python
from src.clustering import FuzzyClusterer

# Load clustering
clusterer = FuzzyClusterer()
clusterer.load("models/clusters.pkl")

# Get cluster distribution for document
dist = clusterer.get_cluster_distribution(doc_idx=100)
# Returns: {0: 0.05, 1: 0.02, ..., 18: 0.85, 19: 0.01}

# Get dominant cluster
primary = clusterer.get_primary_cluster(embedding)
# Returns: 18

# Predict on new embedding
probs = clusterer.predict(new_embedding)
# Returns: array([[0.05, 0.02, ..., 0.85, 0.01]])
```

### Inspect Clusters
```python
# Show sample documents from cluster
clusterer.show_cluster_samples(cluster_id=5, documents=docs, top_k=5)

# Show uncertain documents
clusterer.show_uncertain_documents(documents=docs, threshold=0.5, top_k=10)
```

## Integration with Semantic Cache

### Use Case 1: Cluster-Based Filtering
```python
# Query belongs to cluster 5 with 0.8 probability
query_cluster = clusterer.get_primary_cluster(query_embedding)

# Filter documents from same cluster
cluster_docs = np.where(clusterer.dominant_clusters == query_cluster)[0]

# Search only within cluster (faster)
similarities = compute_similarity(query_embedding, embeddings[cluster_docs])
```

### Use Case 2: Similarity Boosting
```python
# Boost similarity for documents in same cluster
base_similarity = cosine_similarity(query_emb, doc_emb)

# Get cluster overlap
query_probs = clusterer.predict(query_emb)
doc_probs = clusterer.cluster_probs[doc_idx]

# Compute cluster similarity (dot product of probability distributions)
cluster_similarity = np.dot(query_probs, doc_probs)

# Combined score
final_score = 0.7 * base_similarity + 0.3 * cluster_similarity
```

### Use Case 3: Cache Key Enhancement
```python
# Include cluster info in cache key
cache_key = {
    'embedding': query_embedding,
    'primary_cluster': primary_cluster,
    'cluster_probs': cluster_probs
}

# Match queries with similar cluster distributions
```

## Testing

### Test Script
```bash
python test_clustering.py
```

### Verification
- ✅ Probabilities sum to 1.0 (mean: 1.000000)
- ✅ Probabilities in valid range [0, 1]
- ✅ Cluster predictions work correctly
- ✅ Primary cluster identification works
- ✅ High average confidence (0.9935)

## Performance

- **Training time**: 79.19 seconds (~1.3 minutes)
- **Prediction time**: ~0.001 seconds per document
- **Memory**: 3.44 MB for clustering results
- **Scalability**: O(n × k × d) where n=docs, k=clusters, d=dimensions

## Next Steps

1. ✅ Data loading - DONE
2. ✅ Text cleaning - DONE
3. ✅ Embeddings - DONE
4. ✅ Fuzzy clustering - DONE
5. ⏭️ **Semantic cache** - NEXT
6. ⏭️ Query engine
7. ⏭️ API

## Files Created

- `src/clustering.py` - FuzzyClusterer class
- `train_clustering.py` - Training script
- `test_clustering.py` - Testing script
- `models/clusters.pkl` - Saved clustering (3.44 MB)
- `CLUSTERING.md` - This documentation

## Summary

✅ **Fuzzy clustering implemented with GMM**
✅ **Probabilistic cluster membership (not hard assignments)**
✅ **19,997 documents clustered into 20 clusters**
✅ **Average confidence: 99.35%**
✅ **Helper functions for cluster inspection**
✅ **Uncertain document detection**
✅ **Ready for semantic cache integration**

The fuzzy clustering system provides rich semantic structure for the cache!
