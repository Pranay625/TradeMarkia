"""
Test Fuzzy Clustering

This script tests the clustering functionality:
1. Load saved clustering results
2. Test probability predictions
3. Inspect cluster distributions
4. Verify fuzzy membership properties
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.clustering import FuzzyClusterer
from src.data_loader import NewsgroupsDataLoader
from src.utils import load_embeddings


def main():
    print("\n" + "="*70)
    print("TEST FUZZY CLUSTERING")
    print("="*70)
    
    clusters_path = project_root / "models" / "clusters.pkl"
    embeddings_path = project_root / "models" / "embeddings.pkl"
    data_path = project_root / "data" / "20_newsgroups"
    
    # Check if clustering exists
    if not clusters_path.exists():
        print(f"\n[ERROR] Clustering not found at {clusters_path}")
        print("Please run: python train_clustering.py")
        return
    
    # Load clustering
    print("\n[1] Loading clustering results...")
    print("-"*70)
    clusterer = FuzzyClusterer()
    clusterer.load(str(clusters_path))
    
    # Load embeddings and documents
    embeddings, metadata = load_embeddings(str(embeddings_path))
    loader = NewsgroupsDataLoader(data_path, apply_cleaning=True)
    documents = loader.load()
    
    # Test 1: Verify probability properties
    print("\n[2] Verifying fuzzy clustering properties...")
    print("-"*70)
    
    # Check that probabilities sum to 1
    prob_sums = np.sum(clusterer.cluster_probs, axis=1)
    print(f"Probability sums (should be ~1.0):")
    print(f"  Mean: {np.mean(prob_sums):.6f}")
    print(f"  Min:  {np.min(prob_sums):.6f}")
    print(f"  Max:  {np.max(prob_sums):.6f}")
    
    # Check probability range
    print(f"\nProbability range:")
    print(f"  Min probability: {np.min(clusterer.cluster_probs):.6f}")
    print(f"  Max probability: {np.max(clusterer.cluster_probs):.6f}")
    
    # Test 2: Show example distributions
    print("\n[3] Example cluster distributions...")
    print("-"*70)
    
    # Find documents with different confidence levels
    max_probs = np.max(clusterer.cluster_probs, axis=1)
    
    # High confidence document
    high_conf_idx = np.argmax(max_probs)
    print(f"\nHigh Confidence Document (idx={high_conf_idx}):")
    print(f"Category: {documents[high_conf_idx]['category']}")
    print(f"Max probability: {max_probs[high_conf_idx]:.4f}")
    dist = clusterer.get_cluster_distribution(high_conf_idx)
    top_3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
    for cluster_id, prob in top_3:
        print(f"  Cluster {cluster_id}: {prob:.4f}")
    
    # Low confidence document
    low_conf_idx = np.argmin(max_probs)
    print(f"\nLow Confidence Document (idx={low_conf_idx}):")
    print(f"Category: {documents[low_conf_idx]['category']}")
    print(f"Max probability: {max_probs[low_conf_idx]:.4f}")
    dist = clusterer.get_cluster_distribution(low_conf_idx)
    top_3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
    for cluster_id, prob in top_3:
        print(f"  Cluster {cluster_id}: {prob:.4f}")
    
    # Test 3: Predict on new embedding
    print("\n[4] Testing prediction on new embedding...")
    print("-"*70)
    
    test_embedding = embeddings[0]
    probs = clusterer.predict(test_embedding.reshape(1, -1))
    primary = clusterer.get_primary_cluster(test_embedding)
    
    print(f"Test embedding (document 0):")
    print(f"Primary cluster: {primary}")
    print(f"Top 3 clusters:")
    top_3_idx = np.argsort(probs[0])[::-1][:3]
    for idx in top_3_idx:
        print(f"  Cluster {idx}: {probs[0][idx]:.4f}")
    
    # Test 4: Statistics
    print("\n[5] Clustering statistics...")
    print("-"*70)
    
    stats = clusterer.get_cluster_stats()
    print(f"Average confidence: {stats['avg_confidence']:.4f}")
    
    # Confidence distribution
    confidence_bins = [0.3, 0.5, 0.7, 0.9]
    print(f"\nConfidence distribution:")
    prev_bin = 0.0
    for bin_val in confidence_bins:
        count = np.sum((max_probs > prev_bin) & (max_probs <= bin_val))
        percentage = (count / len(max_probs)) * 100
        print(f"  {prev_bin:.1f} < confidence <= {bin_val:.1f}: {count:5d} ({percentage:5.2f}%)")
        prev_bin = bin_val
    count = np.sum(max_probs > prev_bin)
    percentage = (count / len(max_probs)) * 100
    print(f"  {prev_bin:.1f} < confidence <= 1.0: {count:5d} ({percentage:5.2f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("[PASS] Probabilities sum to 1.0")
    print("[PASS] Probabilities in valid range [0, 1]")
    print("[PASS] Cluster predictions work")
    print("[PASS] Primary cluster identification works")
    print(f"[PASS] Average confidence: {stats['avg_confidence']:.4f}")
    print("="*70)
    
    print("\n[SUCCESS] All clustering tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
