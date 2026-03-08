"""
Path Configuration Module

Centralized path management for the application.
All paths are configurable via environment variables.
"""

import os
from pathlib import Path

# Base directory - defaults to /app for Docker, can be overridden
BASE_DIR = Path(os.getenv('BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Data directories
DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', BASE_DIR / 'models'))
CACHE_DIR = Path(os.getenv('CACHE_DIR', BASE_DIR / 'cache'))

# Specific paths
NEWSGROUPS_DATA_PATH = DATA_DIR / '20_newsgroups'
EMBEDDINGS_PATH = MODELS_DIR / 'embeddings.pkl'
CLUSTERS_PATH = MODELS_DIR / 'clusters.pkl'
CACHE_STORE_PATH = CACHE_DIR / 'cache_store.pkl'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
