"""
Query Engine Module

This module orchestrates the entire query processing pipeline.

Responsibilities:
- Coordinate all components (cache, embeddings, clustering, search)
- Implement the main query workflow
- Check semantic cache before processing
- Generate embeddings for queries
- Search document embeddings for relevant results
- Update cache with new results
- Return formatted responses

Query Processing Flow:
1. Receive natural language query
2. Clean and preprocess query text
3. Generate query embedding
4. Check semantic cache for similar queries
5. If cache hit: return cached result
6. If cache miss:
   - Generate result (search documents or process query)
   - Store in cache
7. Return result to user

Key Functions:
- process_query(): Main entry point for query processing
- search_documents(): Find relevant documents using embeddings
- rank_results(): Score and order search results
"""

import time
from typing import Dict, Any, Optional, List
import numpy as np


class QueryEngine:
    """Orchestrates query processing with semantic caching."""
    
    def __init__(
        self,
        embedding_model,
        clusterer,
        semantic_cache,
        text_cleaner=None,
        embeddings=None,
        documents=None
    ):
        """
        Initialize query engine with all components.
        
        Args:
            embedding_model: EmbeddingModel instance
            clusterer: FuzzyClusterer instance
            semantic_cache: SemanticCache instance
            text_cleaner: TextCleaner instance (optional)
            embeddings: Precomputed document embeddings (optional)
            documents: Document list (optional)
        """
        self.embedding_model = embedding_model
        self.clusterer = clusterer
        self.semantic_cache = semantic_cache
        self.text_cleaner = text_cleaner
        self.embeddings = embeddings
        self.documents = documents
        
        print("Query Engine initialized")
        print(f"Embedding model: {embedding_model.model_name}")
        print(f"Cache threshold: {semantic_cache.similarity_threshold}")
        print(f"Cluster-aware: {semantic_cache.use_clustering}")
    
    def process_query(
        self,
        query_text: str,
        use_cache: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query_text: User's query string
            use_cache: Whether to use semantic cache
            top_k: Number of top results to return
            
        Returns:
            Dictionary containing:
                - query: Original query
                - cache_hit: Boolean indicating if result was cached
                - matched_query: Matched cached query (if hit)
                - similarity_score: Similarity to cached query (if hit)
                - result: Query result
                - dominant_cluster: Query's primary cluster
                - processing_time_ms: Time taken to process
        """
        start_time = time.time()
        
        # Step 1: Clean query text (if cleaner available)
        if self.text_cleaner:
            cleaned_query = self.text_cleaner.clean(query_text)
        else:
            cleaned_query = query_text
        
        # Step 2: Generate query embedding
        query_embedding = self.embedding_model.encode_query(cleaned_query)
        
        # Step 3: Predict dominant cluster
        dominant_cluster = self.clusterer.get_primary_cluster(query_embedding)
        
        # Step 4: Check semantic cache
        cache_result = None
        if use_cache:
            cache_result = self.semantic_cache.search_cache(
                query_embedding,
                query_cluster=dominant_cluster
            )
        
        # Step 5: Process based on cache result
        if cache_result and cache_result['cache_hit']:
            # Cache HIT - return cached result
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'query': query_text,
                'cache_hit': True,
                'matched_query': cache_result['matched_query'],
                'similarity_score': cache_result['similarity_score'],
                'result': cache_result['cached_result'],
                'dominant_cluster': dominant_cluster,
                'processing_time_ms': processing_time
            }
        else:
            # Cache MISS - generate result
            result = self._generate_result(query_text, query_embedding, top_k)
            
            # Store in cache
            if use_cache:
                self.semantic_cache.add_entry(
                    query_text,
                    query_embedding,
                    result,
                    dominant_cluster
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'query': query_text,
                'cache_hit': False,
                'matched_query': None,
                'similarity_score': cache_result['similarity_score'] if cache_result else 0.0,
                'result': result,
                'dominant_cluster': dominant_cluster,
                'processing_time_ms': processing_time
            }
    
    def _generate_result(self, query_text: str, query_embedding: np.ndarray, top_k: int) -> Any:
        """
        Generate result for query.
        
        For now, this is a simulated response.
        In production, this would search documents and return relevant results.
        
        Args:
            query_text: Query string
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Generated result
        """
        # Simulated result generation
        result = f"Generated response for: {query_text}"
        
        # If we have document embeddings, we could do real search:
        # if self.embeddings is not None:
        #     similarities = np.dot(self.embeddings, query_embedding)
        #     top_indices = np.argsort(similarities)[-top_k:][::-1]
        #     result = [self.documents[i] for i in top_indices]
        
        return result
    
    def search_documents(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        cluster_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            cluster_filter: Optional cluster to filter by
            
        Returns:
            List of document dictionaries with scores
        """
        if self.embeddings is None or self.documents is None:
            return []
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Apply cluster filter if specified
        if cluster_filter is not None:
            cluster_mask = self.clusterer.dominant_clusters == cluster_filter
            similarities = similarities * cluster_mask
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document_id': int(idx),
                'text': self.documents[idx]['text'][:200],
                'category': self.documents[idx]['category'],
                'similarity_score': float(similarities[idx]),
                'cluster_id': int(self.clusterer.dominant_clusters[idx])
            })
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        return self.semantic_cache.get_stats()
