"""
Embedding Model Module

This module handles text-to-vector conversion using transformer models.

Responsibilities:
- Load pre-trained sentence transformer models
- Generate dense vector embeddings for text
- Support batch processing for efficiency
- Provide similarity computation utilities

Key Functions:
- encode(): Convert text to embedding vector
- encode_batch(): Process multiple texts efficiently
- compute_similarity(): Calculate cosine similarity between vectors

Recommended Models:
- all-MiniLM-L6-v2 (fast, 384 dims)
- all-mpnet-base-v2 (accurate, 768 dims)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from tqdm import tqdm


class EmbeddingModel:
    """Generates dense vector embeddings from text using sentence transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = None
    
    def load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            normalize: Normalize vector to unit length
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if self.model is None:
            self.load_model()
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embedding
    
    def encode_documents(self, documents: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (num_documents, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        print(f"\nGenerating embeddings for {len(documents)} documents...")
        print(f"Batch size: {batch_size}")
        
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"Embeddings generated. Shape: {embeddings.shape}")
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (alias for encode).
        
        Args:
            query: Query text string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.encode(query, normalize=True)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        # Cosine similarity (assuming normalized embeddings)
        return np.dot(embedding1, embedding2)
    
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        if self.model is None:
            self.load_model()
        return self.embedding_dim
