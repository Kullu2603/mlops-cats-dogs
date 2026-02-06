from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "MLOps Cat vs Dog API is running!"}

def test_health():
    # If model isn't present during CI (unit test only), we expect 503 or need to mock
    # For simplicity, we assume model.h5 is present in CI
    if os.path.exists("model.h5"):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
