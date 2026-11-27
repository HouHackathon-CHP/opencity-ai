# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpenCity AI Microservice."""

import sys
from pathlib import Path

# Ensure app module is in python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_analyze_requires_internal_secret_when_configured(client, monkeypatch):
    from app.core import security

    monkeypatch.setattr(security, "INTERNAL_SECRET", "integration-secret")
    payload = {
        "text": "Cột điện hỏng cần sửa ngay",
        "labels": ["infrastructure", "environment"],
        "examples": [
            {"label": "infrastructure", "text": "Cột điện hỏng"},
            {"label": "environment", "text": "Rác thải tồn đọng"},
        ],
    }
    assert client.post("/analyze", json=payload).status_code == 403
    response = client.post("/analyze", json=payload,
                           headers={"X-Internal-Secret": "integration-secret",
                                    "X-DXOS-Tenant": "demo"})
    assert response.status_code == 200


def test_backend_triage_contract_accepts_location_and_urgency(client, monkeypatch):
    from app.core import security

    monkeypatch.setattr(security, "INTERNAL_SECRET", "integration-secret")
    headers = {"X-Internal-Secret": "integration-secret", "X-DXOS-Tenant": "demo"}
    unit_response = client.post("/analyze", headers=headers, json={
        "text": "Cây đổ chắn đường trước trường học",
        "labels": ["environment", "infrastructure", "intake", "transport"],
        "examples": [
            {"label": "environment", "text": "Cây xanh gãy đổ nguy hiểm"},
            {"label": "transport", "text": "Xe buýt chậm tuyến đường"},
        ],
        "candidates": [{
            "id": "report-2", "text": "Cây đổ trên phố gần trường học",
            "latitude": 21.028, "longitude": 105.854,
            "created_at": "2026-11-25T08:00:00+07:00",
        }],
        "latitude": 21.0281, "longitude": 105.8541,
        "created_at": "2026-11-25T08:15:00+07:00",
    })
    assert unit_response.status_code == 200
    result = unit_response.json()
    assert result["ranked_labels"][0]["label"] in {"environment", "infrastructure", "intake", "transport"}
    assert "model_version" in result and "similar" in result and "signals" in result

    urgency_response = client.post("/analyze", headers=headers, json={
        "text": "Cây đổ chắn đường trước trường học",
        "labels": ["high", "normal", "low"],
        "examples": [
            {"label": "high", "text": "Cây đổ chắn đường nguy hiểm"},
            {"label": "normal", "text": "Đèn đường hỏng cần sửa"},
            {"label": "low", "text": "Đề xuất trồng thêm cây xanh"},
        ],
    })
    assert urgency_response.status_code == 200
    assert urgency_response.json()["ranked_labels"][0]["label"] in {"high", "normal", "low"}


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


def test_voice_transcription(client):
    import io
    import wave
    import numpy as np

    # Generate 1-second in-memory WAV audio for testing
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 10000).astype(np.int16)
        wf.writeframes(samples.tobytes())
    buf.seek(0)

    files = {"audio": ("test.wav", buf.read(), "audio/wav")}
    response = client.post("/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "suggested_title" in data
    assert "suggested_category" in data
    assert "duration_seconds" in data
    assert data["duration_seconds"] > 0

