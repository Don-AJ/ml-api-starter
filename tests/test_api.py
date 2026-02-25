from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid():
    payload = {
        "features": [14.2, 20.1, 92.0, 650.0, 0.10, 0.12, 0.10, 0.05, 0.18, 0.06,
                    0.30, 1.00, 2.00, 25.0, 0.006, 0.02, 0.02, 0.01, 0.02, 0.003,
                    16.0, 25.0, 105.0, 850.0, 0.14, 0.30, 0.25, 0.10, 0.30, 0.08]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probability_class_1" in data

def test_predict_invalid_length():
    payload = {"features": [1.0, 2.0]}  # wrong length
    r = client.post("/predict", json=payload)
    assert r.status_code == 422