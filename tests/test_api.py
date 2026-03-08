"""
Test FastAPI Service

This script tests the API endpoints using HTTP requests.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n[1] Testing GET /health...")
    print("-" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("[PASS] Health check successful")


def test_query_cache_miss():
    """Test query endpoint (cache miss)."""
    print("\n[2] Testing POST /query (cache miss)...")
    print("-" * 70)
    
    payload = {
        "query": "What are the best graphics cards for gaming?",
        "use_cache": True,
        "top_k": 5
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    
    data = response.json()
    print(f"Query: {data['query']}")
    print(f"Cache hit: {data['cache_hit']}")
    print(f"Dominant cluster: {data['dominant_cluster']}")
    print(f"Result: {data['result']}")
    print(f"Processing time: {data['processing_time_ms']:.2f} ms")
    
    assert response.status_code == 200
    assert data['cache_hit'] == False
    print("[PASS] Query processed (cache miss)")
    
    return data


def test_query_cache_hit():
    """Test query endpoint (cache hit)."""
    print("\n[3] Testing POST /query (cache hit)...")
    print("-" * 70)
    
    payload = {
        "query": "Which GPU is good for gaming?",
        "use_cache": True,
        "top_k": 5
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    
    data = response.json()
    print(f"Query: {data['query']}")
    print(f"Cache hit: {data['cache_hit']}")
    print(f"Matched query: {data['matched_query']}")
    print(f"Similarity: {data['similarity_score']:.4f}")
    print(f"Result: {data['result']}")
    print(f"Processing time: {data['processing_time_ms']:.2f} ms")
    
    assert response.status_code == 200
    assert data['cache_hit'] == True
    print("[PASS] Query processed (cache hit)")
    
    return data


def test_cache_stats():
    """Test cache stats endpoint."""
    print("\n[4] Testing GET /cache/stats...")
    print("-" * 70)
    
    response = requests.get(f"{BASE_URL}/cache/stats")
    print(f"Status: {response.status_code}")
    
    data = response.json()
    print(f"Total entries: {data['total_entries']}")
    print(f"Total queries: {data['total_queries']}")
    print(f"Hit count: {data['hit_count']}")
    print(f"Miss count: {data['miss_count']}")
    print(f"Hit rate: {data['hit_rate']:.2f}%")
    print(f"Similarity threshold: {data['similarity_threshold']}")
    
    assert response.status_code == 200
    print("[PASS] Cache stats retrieved")
    
    return data


def test_clear_cache():
    """Test clear cache endpoint."""
    print("\n[5] Testing DELETE /cache...")
    print("-" * 70)
    
    response = requests.delete(f"{BASE_URL}/cache")
    print(f"Status: {response.status_code}")
    
    data = response.json()
    print(f"Message: {data['message']}")
    print(f"Success: {data['success']}")
    
    assert response.status_code == 200
    assert data['success'] == True
    print("[PASS] Cache cleared")


def test_multiple_queries():
    """Test multiple queries."""
    print("\n[6] Testing multiple queries...")
    print("-" * 70)
    
    queries = [
        "What are the best graphics cards for gaming?",
        "How do I fix my car engine?",
        "What is the best treatment for diabetes?",
        "Which GPU is good for gaming?",  # Similar to first
        "Tell me about machine learning algorithms"
    ]
    
    for i, query_text in enumerate(queries, 1):
        payload = {"query": query_text, "use_cache": True}
        response = requests.post(f"{BASE_URL}/query", json=payload)
        data = response.json()
        
        status = "HIT" if data['cache_hit'] else "MISS"
        print(f"{i}. [{status}] '{query_text[:40]}...' (cluster {data['dominant_cluster']})")
    
    print("[PASS] Multiple queries processed")


def main():
    print("\n" + "=" * 70)
    print("FASTAPI SERVICE TEST")
    print("=" * 70)
    print("\nMake sure the API server is running:")
    print("  uvicorn api.main:app --reload")
    print("\nWaiting for server to be ready...")
    
    # Wait for server
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("Server is ready!\n")
                break
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(1)
            else:
                print("\n[ERROR] Server not responding. Please start the server first.")
                return
    
    try:
        # Run tests
        test_health()
        test_query_cache_miss()
        test_query_cache_hit()
        test_cache_stats()
        test_multiple_queries()
        test_cache_stats()  # Check stats after multiple queries
        test_clear_cache()
        test_cache_stats()  # Check stats after clear
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("[PASS] Health check works")
        print("[PASS] Query endpoint works (cache miss)")
        print("[PASS] Query endpoint works (cache hit)")
        print("[PASS] Cache stats endpoint works")
        print("[PASS] Clear cache endpoint works")
        print("[PASS] Multiple queries work")
        print("=" * 70)
        
        print("\n[SUCCESS] All API tests passed!")
        print("=" * 70 + "\n")
    
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed: {e}")


if __name__ == "__main__":
    main()
