import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True

def test_predict():
    payload = {
        "MedInc":3.5,"HouseAge":20,"AveRooms":5.4,"AveBedrms":1.1,
        "Population":800,"AveOccup":3.0,"Latitude":34.19,"Longitude":-118.53
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
