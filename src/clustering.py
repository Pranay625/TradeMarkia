"""
Clustering Module

This module implements fuzzy clustering for document embeddings.

Responsibilities:
- Perform fuzzy clustering using Gaussian Mixture Models
- Assign probabilistic membership scores (not hard assignments)
- Provide cluster analysis and inspection utilities
- Enable cluster-based filtering for queries

Key Functions:
- fit(): Train clustering model on embeddings
- predict(): Get cluster memberships for new embeddings
- get_cluster_centers(): Return cluster centroids

Why Fuzzy Clustering?
- Documents often belong to multiple topics (e.g., "computer graphics for medical imaging")
- Probabilistic membership captures semantic ambiguity
- Better represents real-world document distributions
- Enables nuanced similarity comparisons
- Helps identify uncertain/boundary documents
"""

import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class FuzzyClusterer:
    """Performs fuzzy clustering on document embeddings using Gaussian Mixture Models."""
    
    def __init__(self, n_clusters: int = 20, random_state: int = 42):
        """
        Initialize fuzzy clustering model.
        
        Args:
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',  # Diagonal covariance for efficiency
            random_state=random_state,
            max_iter=150,
            n_init=10
        )
        self.cluster_probs = None  # Probability distributions for each document
        self.dominant_clusters = None  # Dominant cluster for each document
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'FuzzyClusterer':
        """
        Train the clustering model on embeddings.
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            
        Returns:
            self
        """
        print(f"\nTraining Gaussian Mixture Model...")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of documents: {embeddings.shape[0]}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Fit GMM
        self.model.fit(embeddings)
        
        # Compute probability distributions for all documents
        self.cluster_probs = self.model.predict_proba(embeddings)
        
        # Determine dominant cluster for each document
        self.dominant_clusters = np.argmax(self.cluster_probs, axis=1)
        
        self.is_fitted = True
        
        print(f"Clustering complete!")
        print(f"Converged: {self.model.converged_}")
        print(f"Iterations: {self.model.n_iter_}")
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get cluster membership probabilities for embeddings.
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            
        Returns:
            Probability matrix of shape (n_samples, n_clusters)
            Each row sums to 1.0
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(embeddings)
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Return cluster centroids (means).
        
        Returns:
            numpy array of shape (n_clusters, embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.means_
    
    def get_primary_cluster(self, embedding: np.ndarray) -> int:
        """
        Get the primary (highest membership) cluster for an embedding.
        
        Args:
            embedding: Single embedding vector
            
        Returns:
            Cluster ID (int)
        """
        probs = self.predict(embedding.reshape(1, -1))
        return np.argmax(probs[0])
    
    def get_cluster_distribution(self, doc_idx: int) -> Dict[int, float]:
        """
        Get cluster probability distribution for a document.
        
        Args:
            doc_idx: Document index
            
        Returns:
            Dictionary mapping cluster_id -> probability
        """
        if self.cluster_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return {i: float(prob) for i, prob in enumerate(self.cluster_probs[doc_idx])}
    
    def get_cluster_stats(self) -> Dict:
        """
        Get statistics about cluster assignments.
        
        Returns:
            Dictionary with cluster statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Count documents per cluster (by dominant cluster)
        unique, counts = np.unique(self.dominant_clusters, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
        
        # Average max probability (confidence)
        max_probs = np.max(self.cluster_probs, axis=1)
        avg_confidence = np.mean(max_probs)
        
        return {
            'n_clusters': self.n_clusters,
            'cluster_sizes': cluster_sizes,
            'avg_confidence': float(avg_confidence),
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_
        }
    
    def show_cluster_samples(self, cluster_id: int, documents: List[Dict], top_k: int = 5):
        """
        Show sample documents from a specific cluster.
        
        Args:
            cluster_id: Cluster ID to inspect
            documents: List of document dictionaries
            top_k: Number of samples to show
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find documents where this is the dominant cluster
        cluster_docs = np.where(self.dominant_clusters == cluster_id)[0]
        
        # Get probabilities for these documents
        cluster_probs_for_docs = self.cluster_probs[cluster_docs, cluster_id]
        
        # Sort by probability (highest first)
        sorted_indices = np.argsort(cluster_probs_for_docs)[::-1][:top_k]
        top_doc_indices = cluster_docs[sorted_indices]
        
        print(f"\nCluster {cluster_id} - Top {top_k} Documents")
        print("="*70)
        print(f"Total documents in cluster: {len(cluster_docs)}")
        print("-"*70)
        
        for i, doc_idx in enumerate(top_doc_indices, 1):
            doc = documents[doc_idx]
            prob = self.cluster_probs[doc_idx, cluster_id]
            print(f"\n{i}. Document {doc_idx} (Probability: {prob:.4f})")
            print(f"   Category: {doc['category']}")
            print(f"   Text preview: {doc['text'][:150]}...")
            
            # Show top 3 cluster memberships
            top_clusters = np.argsort(self.cluster_probs[doc_idx])[::-1][:3]
            print(f"   Cluster distribution:")
            for c in top_clusters:
                print(f"     Cluster {c}: {self.cluster_probs[doc_idx, c]:.4f}")
    
    def show_uncertain_documents(self, documents: List[Dict], threshold: float = 0.5, top_k: int = 10):
        """
        Show documents with uncertain cluster membership.
        
        Uncertain documents are those where the highest cluster probability
        is below the threshold, indicating the document spans multiple topics.
        
        Args:
            documents: List of document dictionaries
            threshold: Maximum probability threshold for uncertainty
            top_k: Number of uncertain documents to show
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find documents with max probability below threshold
        max_probs = np.max(self.cluster_probs, axis=1)
        uncertain_mask = max_probs < threshold
        uncertain_indices = np.where(uncertain_mask)[0]
        
        # Sort by uncertainty (lowest max probability first)
        sorted_indices = np.argsort(max_probs[uncertain_indices])[:top_k]
        top_uncertain = uncertain_indices[sorted_indices]
        
        print(f"\nUncertain Documents (max probability < {threshold})")
        print("="*70)
        print(f"Total uncertain documents: {len(uncertain_indices)} ({len(uncertain_indices)/len(documents)*100:.2f}%)")
        print(f"Showing top {min(top_k, len(top_uncertain))} most uncertain")
        print("-"*70)
        
        for i, doc_idx in enumerate(top_uncertain, 1):
            doc = documents[doc_idx]
            max_prob = max_probs[doc_idx]
            dominant = self.dominant_clusters[doc_idx]
            
            print(f"\n{i}. Document {doc_idx} (Max probability: {max_prob:.4f})")
            print(f"   Category: {doc['category']}")
            print(f"   Dominant cluster: {dominant}")
            print(f"   Text preview: {doc['text'][:150]}...")
            
            # Show top 3 cluster memberships
            top_clusters = np.argsort(self.cluster_probs[doc_idx])[::-1][:3]
            print(f"   Cluster distribution:")
            for c in top_clusters:
                print(f"     Cluster {c}: {self.cluster_probs[doc_idx, c]:.4f}")
    
    def save(self, filepath: str):
        """
        Save clustering results to disk.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'cluster_probs': self.cluster_probs,
            'dominant_clusters': self.dominant_clusters,
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nClustering results saved to {filepath}")
        print(f"File size: {Path(filepath).stat().st_size / (1024*1024):.2f} MB")
    
    def load(self, filepath: str):
        """
        Load clustering results from disk.
        
        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.cluster_probs = data['cluster_probs']
        self.dominant_clusters = data['dominant_clusters']
        self.n_clusters = data['n_clusters']
        self.is_fitted = data['is_fitted']
        
        print(f"\nClustering results loaded from {filepath}")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of documents: {len(self.dominant_clusters)}")
