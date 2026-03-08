"""
Utilities Module

This module provides helper functions and utilities used across the project.

Responsibilities:
- Logging configuration and setup
- Performance timing decorators
- File I/O helpers (save/load embeddings, models)
- Metrics calculation (precision, recall, etc.)
- Configuration management
- Data serialization utilities

Key Functions:
- setup_logging(): Configure logging for the application
- timer(): Decorator to measure function execution time
- save_embeddings(): Save embeddings to disk
- load_embeddings(): Load embeddings from disk
- compute_metrics(): Calculate performance metrics
"""

import time
import logging
import pickle
from functools import wraps
from typing import Any, Callable, Dict, Tuple
from pathlib import Path
import numpy as np


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def timer(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


def save_embeddings(embeddings: np.ndarray, filepath: str, metadata: Dict = None):
    """
    Save embeddings to disk with optional metadata.
    
    Args:
        embeddings: Numpy array of embeddings
        filepath: Path to save file
        metadata: Optional metadata dictionary
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'embeddings': embeddings,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Embeddings saved to {filepath}")
    print(f"Shape: {embeddings.shape}")
    print(f"Size: {Path(filepath).stat().st_size / (1024*1024):.2f} MB")


def load_embeddings(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load embeddings from disk.
    
    Args:
        filepath: Path to embeddings file
        
    Returns:
        Tuple of (embeddings, metadata)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    metadata = data.get('metadata', {})
    
    print(f"Embeddings loaded from {filepath}")
    print(f"Shape: {embeddings.shape}")
    
    return embeddings, metadata


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between -1 and 1
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def batch_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple documents.
    
    Args:
        query_vec: Query vector of shape (embedding_dim,)
        doc_vecs: Document vectors of shape (n_docs, embedding_dim)
        
    Returns:
        Array of similarity scores of shape (n_docs,)
    """
    # Normalize vectors
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    
    # Compute dot product
    similarities = np.dot(doc_norms, query_norm)
    return similarities


def hash_embedding(embedding: np.ndarray) -> str:
    """
    Generate a hash string for an embedding vector.
    
    Args:
        embedding: Embedding vector
        
    Returns:
        Hash string
    """
    import hashlib
    return hashlib.md5(embedding.tobytes()).hexdigest()


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end(self, operation: str):
        """End timing an operation and record duration."""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(elapsed)
            del self.start_times[operation]
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {}
        for op, times in self.timings.items():
            stats[op] = {
                'count': len(times),
                'total': sum(times),
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        return stats
