# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for AI analysis, classification, and deduplication."""

from pydantic import BaseModel, Field


class Example(BaseModel):
    """Reference classification example for dynamic few-shot fallback."""
    label: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=3, max_length=2000)


class Candidate(BaseModel):
    """Candidate report for spatiotemporal deduplication."""
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=3, max_length=2000)
    latitude: float | None = None
    longitude: float | None = None
    created_at: str | None = None


class AnalysisRequest(BaseModel):
    """Incoming request for civic unit triage and incident deduplication."""
    text: str = Field(min_length=5, max_length=4000)
    labels: list[str] = Field(min_length=2, max_length=20)
    examples: list[Example] = Field(min_length=2, max_length=120)
    candidates: list[Candidate] = Field(default_factory=list, max_length=150)
    latitude: float | None = None
    longitude: float | None = None
    created_at: str | None = None


class RankedLabel(BaseModel):
    label: str
    score: float


class SimilarMatch(BaseModel):
    id: str
    similarity: float


class AnalysisResponse(BaseModel):
    model_version: str
    ranked_labels: list[RankedLabel]
    signals: list[str]
    similar: list[SimilarMatch]


class HealthResponse(BaseModel):
    status: str
    service: str
    ai_model: str
    device: str
    model_version: str
