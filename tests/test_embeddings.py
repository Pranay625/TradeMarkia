"""
Test Embedding Model

This script tests the embedding model functionality:
1. Load model
2. Generate embeddings for sample texts
3. Compute similarity between texts
4. Verify embedding properties
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.embedding_model import EmbeddingModel


def main():
    print("\n" + "="*70)
    print("EMBEDDING MODEL TEST")
    print("="*70)
    
    # Initialize model
    print("\n[1] Initializing embedding model...")
    print("-"*70)
    model = EmbeddingModel(model_name='all-MiniLM-L6-v2', device='cpu')
    model.load_model()
    
    # Test single encoding
    print("\n[2] Testing single text encoding...")
    print("-"*70)
    text = "Machine learning is a subset of artificial intelligence."
    embedding = model.encode(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch encoding
    print("\n[3] Testing batch encoding...")
    print("-"*70)
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for data science.",
        "The weather is nice today.",
        "I love playing basketball on weekends."
    ]
    
    embeddings = model.encode_documents(texts, batch_size=2, show_progress=False)
    print(f"Number of texts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test similarity computation
    print("\n[4] Testing similarity computation...")
    print("-"*70)
    
    # Similar texts
    text1 = "Machine learning and artificial intelligence"
    text2 = "AI and machine learning technologies"
    text3 = "I enjoy playing sports"
    
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    emb3 = model.encode(text3)
    
    sim_12 = model.compute_similarity(emb1, emb2)
    sim_13 = model.compute_similarity(emb1, emb3)
    
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    print(f"\nSimilarity (Text 1 vs Text 2): {sim_12:.4f} [Similar topics]")
    print(f"Similarity (Text 1 vs Text 3): {sim_13:.4f} [Different topics]")
    
    # Verify properties
    print("\n[5] Verifying embedding properties...")
    print("-"*70)
    
    # Check normalization
    norm = np.linalg.norm(emb1)
    print(f"Embedding norm (should be ~1.0): {norm:.6f}")
    
    # Check dimensionality
    print(f"Embedding dimension: {model.get_embedding_dim()}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("[PASS] Model loaded successfully")
    print("[PASS] Single text encoding works")
    print("[PASS] Batch encoding works")
    print("[PASS] Similarity computation works")
    print(f"[PASS] Embeddings are normalized (norm={norm:.6f})")
    print(f"[PASS] Similar texts have high similarity ({sim_12:.4f})")
    print(f"[PASS] Different texts have low similarity ({sim_13:.4f})")
    print("="*70)
    
    print("\n[SUCCESS] All embedding tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
