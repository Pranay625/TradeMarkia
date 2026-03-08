"""
Pre-flight Check Script

Validates that all required files and configurations are present
before starting the API server.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.paths import EMBEDDINGS_PATH, CLUSTERS_PATH, MODELS_DIR, CACHE_DIR
import config


def check_directories():
    """Check required directories exist."""
    print("Checking directories...")
    
    dirs = {
        'Models': MODELS_DIR,
        'Cache': CACHE_DIR
    }
    
    for name, path in dirs.items():
        if path.exists():
            print(f"  [OK] {name} directory: {path}")
        else:
            print(f"  [CREATING] {name} directory: {path}")
            path.mkdir(parents=True, exist_ok=True)


def check_models():
    """Check required model files exist."""
    print("\nChecking model files...")
    
    files = {
        'Embeddings': EMBEDDINGS_PATH,
        'Clusters': CLUSTERS_PATH
    }
    
    missing = []
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"  [MISSING] {name}: {path}")
            missing.append((name, path))
    
    return missing


def check_config():
    """Check configuration values."""
    print("\nChecking configuration...")
    
    configs = {
        'Embedding Model': config.EMBEDDING_MODEL_NAME,
        'Similarity Threshold': config.CACHE_SIMILARITY_THRESHOLD,
        'Number of Clusters': config.N_CLUSTERS,
        'Cache Max Size': config.CACHE_MAX_SIZE,
        'API Port': config.API_PORT
    }
    
    for name, value in configs.items():
        print(f"  [OK] {name}: {value}")


def main():
    print("="*70)
    print("PRE-FLIGHT CHECK")
    print("="*70)
    
    # Check directories
    check_directories()
    
    # Check models
    missing = check_models()
    
    # Check config
    check_config()
    
    # Summary
    print("\n" + "="*70)
    if missing:
        print("FAILED - Missing required files:")
        for name, path in missing:
            print(f"  - {name}: {path}")
        print("\nPlease run:")
        print("  python generate_embeddings.py")
        print("  python train_clustering.py")
        print("="*70)
        sys.exit(1)
    else:
        print("SUCCESS - All checks passed!")
        print("Ready to start API server:")
        print("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
        print("="*70)
        sys.exit(0)


if __name__ == "__main__":
    main()
