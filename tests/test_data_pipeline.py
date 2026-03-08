"""
Test script for data loading and text cleaning.

This script demonstrates:
1. Loading the 20 Newsgroups dataset from disk
2. Applying text cleaning
3. Displaying dataset statistics
4. Showing before/after cleaning examples
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import NewsgroupsDataLoader
from src.text_cleaner import TextCleaner


def main():
    print("\n" + "="*70)
    print("20 NEWSGROUPS DATASET INGESTION PIPELINE TEST")
    print("="*70)
    
    # Path to dataset
    data_path = project_root / "data" / "20_newsgroups"
    
    if not data_path.exists():
        print(f"\n[ERROR] Dataset not found at {data_path}")
        print("Please ensure the 20_newsgroups dataset is in the data/ directory")
        return
    
    # Test 1: Load dataset with cleaning
    print("\n[1] Loading dataset with text cleaning enabled...")
    loader = NewsgroupsDataLoader(data_path, apply_cleaning=True)
    documents = loader.load()
    
    # Display statistics
    loader.print_stats()
    
    # Test 2: Show cleaning example
    if documents:
        print("\n[2] Text Cleaning Example")
        print("="*70)
        
        # Find a document with substantial content
        sample_doc = None
        for doc in documents:
            if len(doc['text']) > 200:
                sample_doc = doc
                break
        
        if sample_doc:
            print(f"\nCategory: {sample_doc['category']}")
            print(f"\nCleaned Text (first 500 chars):")
            print("-"*70)
            print(sample_doc['text'][:500])
            print("...")
            print("-"*70)
    
    # Test 3: Compare with/without cleaning
    print("\n[3] Cleaning Impact Comparison")
    print("="*70)
    
    # Load without cleaning
    loader_raw = NewsgroupsDataLoader(data_path, apply_cleaning=False)
    documents_raw = loader_raw.load()
    
    if documents and documents_raw:
        # Calculate average document length
        avg_len_cleaned = sum(len(d['text']) for d in documents) / len(documents)
        avg_len_raw = sum(len(d['text']) for d in documents_raw) / len(documents_raw)
        
        print(f"Documents loaded (raw):     {len(documents_raw)}")
        print(f"Documents loaded (cleaned): {len(documents)}")
        print(f"Average length (raw):       {avg_len_raw:.0f} characters")
        print(f"Average length (cleaned):   {avg_len_cleaned:.0f} characters")
        print(f"Size reduction:             {((avg_len_raw - avg_len_cleaned) / avg_len_raw * 100):.1f}%")
    
    print("\n" + "="*70)
    print("[SUCCESS] Dataset ingestion pipeline test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
