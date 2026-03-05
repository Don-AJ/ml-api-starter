import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid(client):
    payload = {
        "features": [
            14.2,
            20.1,
            92.0,
            650.0,
            0.10,
            0.12,
            0.10,
            0.05,
            0.18,
            0.06,
            0.30,
            1.00,
            2.00,
            25.0,
            0.006,
            0.02,
            0.02,
            0.01,
            0.02,
            0.003,
            16.0,
            25.0,
            105.0,
            850.0,
            0.14,
            0.30,
            0.25,
            0.10,
            0.30,
            0.08,
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert "prediction" in data
    assert "probability_class_1" in data
    assert "model_version" in data
    assert "environment" in data


def test_predict_invalid_feature_count(client):
    r = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert r.status_code in [400, 422]
