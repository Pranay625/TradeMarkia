"""
Generate Document Embeddings

This script:
1. Loads the 20 Newsgroups dataset
2. Cleans the text
3. Generates embeddings using sentence-transformers
4. Saves embeddings to disk

Usage:
    python generate_embeddings.py
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import NewsgroupsDataLoader
from src.embedding_model import EmbeddingModel
from src.utils import save_embeddings
from src.paths import NEWSGROUPS_DATA_PATH, EMBEDDINGS_PATH


def main():
    print("\n" + "="*70)
    print("DOCUMENT EMBEDDINGS GENERATION")
    print("="*70)
    
    # Configuration
    embeddings_path = EMBEDDINGS_PATH
    model_name = "all-MiniLM-L6-v2"
    batch_size = 32
    
    # Step 1: Load dataset
    print("\n[Step 1/3] Loading dataset...")
    print("-"*70)
    loader = NewsgroupsDataLoader(NEWSGROUPS_DATA_PATH, apply_cleaning=True)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents")
    print(f"Categories: {len(loader.get_categories())}")
    
    # Extract text from documents
    texts = [doc['text'] for doc in documents]
    categories = [doc['category'] for doc in documents]
    
    # Step 2: Generate embeddings
    print("\n[Step 2/3] Generating embeddings...")
    print("-"*70)
    print(f"Model: {model_name}")
    
    start_time = time.time()
    
    embedding_model = EmbeddingModel(model_name=model_name, device='cpu')
    embeddings = embedding_model.encode_documents(
        texts,
        batch_size=batch_size,
        show_progress=True
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nEmbedding generation complete!")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Documents per second: {len(documents) / elapsed_time:.2f}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Step 3: Save embeddings
    print("\n[Step 3/3] Saving embeddings...")
    print("-"*70)
    
    metadata = {
        'model_name': model_name,
        'num_documents': len(documents),
        'embedding_dim': embeddings.shape[1],
        'categories': categories,
        'generation_time': elapsed_time
    }
    
    save_embeddings(embeddings, str(embeddings_path), metadata)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Documents processed:    {len(documents)}")
    print(f"Embedding dimension:    {embeddings.shape[1]}")
    print(f"Total embeddings:       {embeddings.shape[0]}")
    print(f"Model used:             {model_name}")
    print(f"Saved to:               {embeddings_path}")
    print(f"Processing time:        {elapsed_time:.2f} seconds")
    print("="*70)
    
    print("\n[SUCCESS] Embeddings generated and saved successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
