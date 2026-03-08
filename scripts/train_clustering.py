"""
Train Fuzzy Clustering

This script:
1. Loads document embeddings
2. Trains Gaussian Mixture Model for fuzzy clustering
3. Saves clustering results
4. Shows cluster statistics and samples
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.clustering import FuzzyClusterer
from src.data_loader import NewsgroupsDataLoader
from src.utils import load_embeddings
from src.paths import EMBEDDINGS_PATH, CLUSTERS_PATH, NEWSGROUPS_DATA_PATH
import config


def main():
    print("\n" + "="*70)
    print("FUZZY CLUSTERING TRAINING")
    print("="*70)
    
    # Configuration
    embeddings_path = EMBEDDINGS_PATH
    clusters_path = CLUSTERS_PATH
    data_path = NEWSGROUPS_DATA_PATH
    n_clusters = config.N_CLUSTERS
    
    # Step 1: Load embeddings
    print("\n[Step 1/4] Loading embeddings...")
    print("-"*70)
    embeddings, metadata = load_embeddings(str(embeddings_path))
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of documents: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Step 2: Train clustering
    print("\n[Step 2/4] Training fuzzy clustering...")
    print("-"*70)
    
    start_time = time.time()
    
    clusterer = FuzzyClusterer(n_clusters=n_clusters, random_state=42)
    clusterer.fit(embeddings)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nClustering training complete!")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Step 3: Show statistics
    print("\n[Step 3/4] Cluster statistics...")
    print("-"*70)
    
    stats = clusterer.get_cluster_stats()
    print(f"Number of clusters: {stats['n_clusters']}")
    print(f"Average confidence: {stats['avg_confidence']:.4f}")
    print(f"Converged: {stats['converged']}")
    print(f"Iterations: {stats['n_iter']}")
    
    print(f"\nCluster sizes (by dominant cluster):")
    for cluster_id in sorted(stats['cluster_sizes'].keys()):
        count = stats['cluster_sizes'][cluster_id]
        percentage = (count / embeddings.shape[0]) * 100
        print(f"  Cluster {cluster_id:2d}: {count:5d} documents ({percentage:5.2f}%)")
    
    # Step 4: Save clustering results
    print("\n[Step 4/4] Saving clustering results...")
    print("-"*70)
    
    clusterer.save(str(clusters_path))
    
    # Show sample clusters
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    # Load documents for inspection
    loader = NewsgroupsDataLoader(data_path, apply_cleaning=True)
    documents = loader.load()
    
    # Show samples from first 3 clusters
    for cluster_id in range(min(3, n_clusters)):
        clusterer.show_cluster_samples(cluster_id, documents, top_k=3)
    
    # Show uncertain documents
    clusterer.show_uncertain_documents(documents, threshold=0.5, top_k=5)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Documents clustered:    {embeddings.shape[0]}")
    print(f"Number of clusters:     {n_clusters}")
    print(f"Average confidence:     {stats['avg_confidence']:.4f}")
    print(f"Training time:          {elapsed_time:.2f} seconds")
    print(f"Saved to:               {clusters_path}")
    print("="*70)
    
    print("\n[SUCCESS] Fuzzy clustering complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
