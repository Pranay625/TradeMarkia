"""
Configuration Module

Central configuration for the semantic cache system.
All values can be overridden via environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
EMBEDDING_DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')

# Clustering Configuration
N_CLUSTERS = int(os.getenv('N_CLUSTERS', '20'))
FUZZINESS_PARAMETER = 2.0
MAX_CLUSTERING_ITERATIONS = 150

# Semantic Cache Configuration
CACHE_SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))
CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '1000'))
CACHE_TTL_SECONDS = 3600
CACHE_EVICTION_POLICY = 'lru'
USE_CLUSTERING = os.getenv('USE_CLUSTERING', 'true').lower() == 'true'

# Query Engine Configuration
TOP_K_RESULTS = 5
BATCH_SIZE = 32

# Data Configuration
NEWSGROUPS_SUBSET = 'train'
NEWSGROUPS_CATEGORIES = None
REMOVE_PARTS = ('headers', 'footers', 'quotes')

# Text Cleaning Configuration
LOWERCASE = True
REMOVE_STOPWORDS = False
REMOVE_NUMBERS = False

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_RELOAD = False

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = None
