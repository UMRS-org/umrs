import requests
import json

def test_health(base_url):
    resp = requests.get(f"{base_url}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data

def test_status(base_url):
    resp = requests.get(f"{base_url}/status")
    assert resp.status_code in (200, 503)  # 503 if not initialized

def test_list_buckets(base_url):
    resp = requests.get(f"{base_url}/buckets")
    assert resp.status_code in (200, 503)

def test_root(base_url):
    resp = requests.get(f"{base_url}/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data

if __name__ == "__main__":
    BASE_URL = "http://localhost:8000"
    test_health(BASE_URL)
    test_status(BASE_URL)
    test_list_buckets(BASE_URL)
    test_root(BASE_URL)
    print("API baseline tests passed.")
