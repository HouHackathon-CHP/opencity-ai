# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpenCity AI Microservice."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "opencity-ai"
    assert "ai_model" in data


def test_analyze_civic_classification(client):
    payload = {
        "text": "Đèn tín hiệu giao thông ngã tư Chùa Bộc bị hỏng gây ùn tắc nghiêm trọng",
        "labels": ["transport", "infrastructure", "environment", "intake"],
        "examples": [
            {"label": "transport", "text": "Tắc đường kẹt xe"},
            {"label": "infrastructure", "text": "Hỏng cột điện vỡ cống"},
            {"label": "environment", "text": "Rác thải ô nhiễm cây gãy"},
            {"label": "intake", "text": "Thủ tục hành chính"},
        ],
        "candidates": [],
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "ranked_labels" in data
    assert len(data["ranked_labels"]) == 4
    # The top ranked label should be transport
    top_label = data["ranked_labels"][0]["label"]
    assert top_label in ["transport", "infrastructure"]


def test_analyze_spatiotemporal_deduplication(client):
    payload = {
        "text": "Cây xà cừ cổ thụ bật gốc chắn ngang đường Chùa Bộc",
        "labels": ["transport", "infrastructure", "environment", "intake"],
        "examples": [
            {"label": "environment", "text": "Cây đổ gãy cành"},
            {"label": "transport", "text": "Tắc nghẽn giao thông"},
        ],
        "candidates": [
            {
                "id": "rep_nearby_same_time",
                "text": "Cây to đổ chắn lối đi phố Chùa Bộc",
                "latitude": 21.0080,
                "longitude": 105.8285,
                "created_at": "2026-11-14T08:00:00Z",
            },
            {
                "id": "rep_far_away",
                "text": "Ngập nước tại quận Cầu Giấy",
                "latitude": 21.0360,
                "longitude": 105.7900,
                "created_at": "2026-11-14T08:00:00Z",
            },
        ],
        "latitude": 21.0082,
        "longitude": 105.8286,
        "created_at": "2026-11-14T08:15:00Z",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "similar" in data
    assert len(data["similar"]) > 0
    # Nearby incident should have high similarity score
    assert data["similar"][0]["id"] == "rep_nearby_same_time"
    assert data["similar"][0]["similarity"] >= 0.5
