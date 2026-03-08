"""
Semantic Cache Module

This module implements a manual semantic cache without external caching libraries.

Responsibilities:
- Store query embeddings and their results in memory
- Detect semantically similar queries using cosine similarity
- Return cached results when similarity exceeds threshold
- Track cache performance metrics (hit rate, latency)
- Provide cluster-aware optimization for faster lookups

Key Functions:
- add_entry(): Store new query-result pair in cache
- search_cache(): Check cache for similar query and return result if found
- get_stats(): Return cache performance statistics
- save(): Persist cache to disk
- load(): Load cache from disk

Cache Structure:
Each entry contains:
- query_text: Original query string
- query_embedding: Vector representation
- result: Cached result
- dominant_cluster: Primary cluster ID
- timestamp: When cached

Similarity Detection:
- Compute cosine similarity between incoming query and all cached queries
- If max_similarity >= threshold (e.g., 0.85), return cached result
- Otherwise, cache miss - process query normally

Cluster-Aware Optimization:
- Predict query's dominant cluster
- Search cache entries in same cluster first
- Fallback to full search if no match in cluster
- Significantly reduces search time for large caches
"""

import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


class SemanticCache:
    """
    Manual semantic cache implementation using vector similarity.
    
    No external caching libraries (Redis, Memcached) are used.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, use_clustering: bool = True):
        """
        Initialize the semantic cache.
        
        Args:
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            use_clustering: Whether to use cluster-aware optimization
        """
        self.similarity_threshold = similarity_threshold
        self.use_clustering = use_clustering
        
        # Cache storage (in-memory list)
        self.cache_entries: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_queries = 0
        self.hit_count = 0
        self.miss_count = 0
        
        # Cluster index for fast lookup (cluster_id -> list of entry indices)
        self.cluster_index: Dict[int, List[int]] = {}
        
        print(f"Semantic Cache initialized")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Cluster-aware optimization: {self.use_clustering}")
    
    def add_entry(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: Any,
        dominant_cluster: Optional[int] = None
    ) -> int:
        """
        Store a new cache entry.
        """
        # Validate inputs
        if query_embedding is None:
            raise ValueError("Query embedding cannot be None")
        
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("Query embedding must be numpy array")
        
        if query_embedding.size == 0:
            raise ValueError("Query embedding cannot be empty")
        
        if len(query_embedding.shape) != 1:
            raise ValueError(f"Query embedding must be 1D array, got shape {query_embedding.shape}")
        
        entry = {
            'query_text': query_text,
            'query_embedding': query_embedding,
            'result': result,
            'dominant_cluster': dominant_cluster,
            'timestamp': datetime.now(),
            'hit_count': 0
        }
        
        entry_idx = len(self.cache_entries)
        self.cache_entries.append(entry)
        
        # Update cluster index
        if dominant_cluster is not None:
            if dominant_cluster not in self.cluster_index:
                self.cluster_index[dominant_cluster] = []
            self.cluster_index[dominant_cluster].append(entry_idx)
        
        return entry_idx
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Assumes embeddings are already normalized.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        return float(np.dot(embedding1, embedding2))
    
    def search_cache(
        self,
        query_embedding: np.ndarray,
        query_cluster: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search cache for semantically similar query.
        
        Steps:
        1. If clustering enabled and query_cluster provided, search cluster first
        2. Compute cosine similarity with cached embeddings
        3. Find most similar cached query
        4. If similarity >= threshold, return cache hit
        5. Otherwise, return cache miss
        
        Args:
            query_embedding: Query embedding vector
            query_cluster: Dominant cluster ID (optional, for optimization)
            
        Returns:
            Dictionary with:
                - cache_hit: Boolean
                - matched_query: Matched query text (if hit)
                - similarity_score: Similarity to matched query
                - cached_result: Cached result (if hit)
                - search_time: Time taken to search
        """
        self.total_queries += 1
        
        if len(self.cache_entries) == 0:
            self.miss_count += 1
            return {
                'cache_hit': False,
                'matched_query': None,
                'similarity_score': 0.0,
                'cached_result': None
            }
        
        start_time = datetime.now()
        
        # Determine search strategy
        if self.use_clustering and query_cluster is not None and query_cluster in self.cluster_index:
            # Cluster-aware search: search cluster entries first
            search_indices = self.cluster_index[query_cluster]
            
            # If no match in cluster, fallback to full search
            best_idx, best_similarity = self._search_entries(query_embedding, search_indices)
            
            if best_similarity < self.similarity_threshold:
                # Fallback to full search
                all_indices = list(range(len(self.cache_entries)))
                best_idx, best_similarity = self._search_entries(query_embedding, all_indices)
        else:
            # Full search
            all_indices = list(range(len(self.cache_entries)))
            best_idx, best_similarity = self._search_entries(query_embedding, all_indices)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold:
            # Cache hit!
            self.hit_count += 1
            matched_entry = self.cache_entries[best_idx]
            matched_entry['hit_count'] += 1
            
            return {
                'cache_hit': True,
                'matched_query': matched_entry['query_text'],
                'similarity_score': best_similarity,
                'cached_result': matched_entry['result'],
                'search_time': search_time,
                'entry_idx': best_idx
            }
        else:
            # Cache miss
            self.miss_count += 1
            return {
                'cache_hit': False,
                'matched_query': None,
                'similarity_score': best_similarity,
                'cached_result': None,
                'search_time': search_time
            }
    
    def _search_entries(
        self,
        query_embedding: np.ndarray,
        search_indices: List[int]
    ) -> Tuple[int, float]:
        """
        Search specific cache entries for best match.
        
        Args:
            query_embedding: Query embedding vector
            search_indices: List of entry indices to search
            
        Returns:
            Tuple of (best_entry_idx, best_similarity)
        """
        if not search_indices:
            return -1, 0.0
        
        best_idx = -1
        best_similarity = -1.0
        
        for idx in search_indices:
            entry = self.cache_entries[idx]
            similarity = self._compute_similarity(query_embedding, entry['query_embedding'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        return best_idx, best_similarity
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = (self.hit_count / self.total_queries * 100) if self.total_queries > 0 else 0.0
        miss_rate = (self.miss_count / self.total_queries * 100) if self.total_queries > 0 else 0.0
        
        # Most frequently used entries
        if self.cache_entries:
            sorted_entries = sorted(
                enumerate(self.cache_entries),
                key=lambda x: x[1]['hit_count'],
                reverse=True
            )
            top_entries = [
                {
                    'idx': idx,
                    'query': entry['query_text'][:50],
                    'hits': entry['hit_count']
                }
                for idx, entry in sorted_entries[:5]
            ]
        else:
            top_entries = []
        
        return {
            'total_entries': len(self.cache_entries),
            'total_queries': self.total_queries,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'similarity_threshold': self.similarity_threshold,
            'use_clustering': self.use_clustering,
            'clusters_indexed': len(self.cluster_index),
            'top_entries': top_entries
        }
    
    def clear(self):
        """Clear all cache entries and reset statistics."""
        self.cache_entries = []
        self.cluster_index = {}
        self.total_queries = 0
        self.hit_count = 0
        self.miss_count = 0
        print("Cache cleared")
    
    def save(self, filepath: str):
        """
        Save cache to disk.
        
        Args:
            filepath: Path to save file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'cache_entries': self.cache_entries,
            'cluster_index': self.cluster_index,
            'total_queries': self.total_queries,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'similarity_threshold': self.similarity_threshold,
            'use_clustering': self.use_clustering
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nCache saved to {filepath}")
        print(f"Entries: {len(self.cache_entries)}")
        print(f"File size: {Path(filepath).stat().st_size / 1024:.2f} KB")
    
    def load(self, filepath: str):
        """
        Load cache from disk.
        
        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.cache_entries = data['cache_entries']
        self.cluster_index = data['cluster_index']
        self.total_queries = data['total_queries']
        self.hit_count = data['hit_count']
        self.miss_count = data['miss_count']
        self.similarity_threshold = data['similarity_threshold']
        self.use_clustering = data['use_clustering']
        
        print(f"\nCache loaded from {filepath}")
        print(f"Entries: {len(self.cache_entries)}")
        print(f"Total queries: {self.total_queries}")
        print(f"Hit rate: {self.get_stats()['hit_rate']:.2f}%")
    
    def size(self) -> int:
        """Return current number of entries in cache."""
        return len(self.cache_entries)
