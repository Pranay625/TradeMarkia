"""
Data Loader Module

This module handles loading and managing the 20 Newsgroups dataset.

Responsibilities:
- Load 20 Newsgroups dataset from disk
- Extract category names from folder structure
- Read documents with proper encoding handling
- Apply text cleaning
- Return structured document list

Key Functions:
- load(): Load all documents from data directory
- get_documents(): Return list of documents with metadata
- print_stats(): Display dataset statistics
"""

import os
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional
from .text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class NewsgroupsDataLoader:
    """Loads and manages the 20 Newsgroups dataset from disk."""
    
    def __init__(self, data_path: str, apply_cleaning: bool = True):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to 20_newsgroups directory
            apply_cleaning: Whether to apply text cleaning
        """
        self.data_path = Path(data_path)
        self.apply_cleaning = apply_cleaning
        self.text_cleaner = TextCleaner() if apply_cleaning else None
        self.documents = []
        self.categories = []
    
    def load(self) -> List[Dict[str, str]]:
        """
        Load the dataset from disk.
        
        Returns:
            List of dictionaries with 'category' and 'text' keys
        """
        documents = []
        categories = []
        
        # Iterate through category folders
        for category_dir in sorted(self.data_path.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            categories.append(category_name)
            
            # Read each document in the category
            for doc_file in category_dir.iterdir():
                if doc_file.is_file():
                    try:
                        # Read with encoding error handling
                        text = doc_file.read_text(encoding='utf-8', errors='ignore')
                        
                        # Apply cleaning if enabled
                        if self.apply_cleaning and text.strip():
                            text = self.text_cleaner.clean(text)
                        
                        # Only add non-empty documents
                        if text.strip():
                            documents.append({
                                'category': category_name,
                                'text': text
                            })
                    except Exception as e:
                        # Log error but continue processing
                        logger.warning(f"Failed to load document {doc_file}: {e}")
                        continue
        
        self.documents = documents
        self.categories = sorted(set(categories))
        return documents
    
    def get_documents(self) -> List[Dict[str, str]]:
        """Return list of loaded documents."""
        return self.documents
    
    def get_categories(self) -> List[str]:
        """Return list of unique categories."""
        return self.categories
    
    def print_stats(self):
        """Print dataset statistics."""
        if not self.documents:
            print("No documents loaded. Call load() first.")
            return
        
        print(f"\n{'='*60}")
        print("20 NEWSGROUPS DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total documents: {len(self.documents)}")
        print(f"Total categories: {len(self.categories)}")
        print(f"\nCategory Distribution:")
        print(f"{'-'*60}")
        
        # Count documents per category
        category_counts = Counter(doc['category'] for doc in self.documents)
        
        for category, count in sorted(category_counts.items()):
            percentage = (count / len(self.documents)) * 100
            print(f"  {category:30s} {count:5d} ({percentage:5.2f}%)")
        
        print(f"{'='*60}\n")
