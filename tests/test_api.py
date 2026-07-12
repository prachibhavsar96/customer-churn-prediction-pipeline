import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.api import CustomerInput, app

EXAMPLE_PAYLOAD = CustomerInput.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=EXAMPLE_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["churn_prediction"] in (0, 1)
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_prediction_matches_probability_threshold(client):
    response = client.post("/predict", json=EXAMPLE_PAYLOAD)
    body = response.json()

    prediction = body["churn_prediction"]
    probability = body["churn_probability"]

    if prediction == 1:
        assert probability >= 0.5
    else:
        assert probability < 0.5
