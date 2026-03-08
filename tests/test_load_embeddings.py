"""
Test Loading Embeddings

This script tests loading saved embeddings and performing similarity search.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.embedding_model import EmbeddingModel
from src.utils import load_embeddings, batch_cosine_similarity


def main():
    print("\n" + "="*70)
    print("TEST LOADING SAVED EMBEDDINGS")
    print("="*70)
    
    embeddings_path = project_root / "models" / "embeddings.pkl"
    
    # Check if embeddings exist
    if not embeddings_path.exists():
        print(f"\n[ERROR] Embeddings not found at {embeddings_path}")
        print("Please run: python generate_embeddings.py")
        return
    
    # Load embeddings
    print("\n[1] Loading saved embeddings...")
    print("-"*70)
    embeddings, metadata = load_embeddings(str(embeddings_path))
    
    print(f"\nMetadata:")
    print(f"  Model: {metadata.get('model_name', 'N/A')}")
    print(f"  Documents: {metadata.get('num_documents', 'N/A')}")
    print(f"  Embedding dim: {metadata.get('embedding_dim', 'N/A')}")
    print(f"  Generation time: {metadata.get('generation_time', 'N/A'):.2f}s")
    
    # Test similarity search
    print("\n[2] Testing similarity search...")
    print("-"*70)
    
    # Create a query
    model = EmbeddingModel(model_name='all-MiniLM-L6-v2', device='cpu')
    model.load_model()
    
    query = "What are the best graphics cards for gaming?"
    print(f"Query: {query}")
    
    query_embedding = model.encode_query(query)
    
    # Compute similarities
    print("\nComputing similarities with all documents...")
    similarities = batch_cosine_similarity(query_embedding, embeddings)
    
    # Get top 5 results
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    print(f"\nTop {top_k} most similar documents:")
    print("-"*70)
    
    categories = metadata.get('categories', [])
    for i, idx in enumerate(top_indices, 1):
        category = categories[idx] if idx < len(categories) else "Unknown"
        similarity = similarities[idx]
        print(f"{i}. Document {idx:5d} | Category: {category:30s} | Similarity: {similarity:.4f}")
    
    # Statistics
    print("\n[3] Similarity statistics...")
    print("-"*70)
    print(f"Mean similarity:   {np.mean(similarities):.4f}")
    print(f"Median similarity: {np.median(similarities):.4f}")
    print(f"Max similarity:    {np.max(similarities):.4f}")
    print(f"Min similarity:    {np.min(similarities):.4f}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Embeddings loaded and tested successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
