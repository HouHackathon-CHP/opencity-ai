# SPDX-License-Identifier: Apache-2.0
"""Text analysis, civic unit routing, and deduplication API router."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import MODEL_VERSION
from app.core.security import internal_caller
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.civic_classifier import classify_text
from app.services.deduplication import find_similar_reports
from app.services.model_loader import get_ai_engine

router = APIRouter(tags=["AI Inference"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(body: AnalysisRequest, tenant: str = Depends(internal_caller)) -> AnalysisResponse:
    """Analyze civic incident text, predict municipal department and urgency, and detect duplicates."""
    labels = list(dict.fromkeys(body.labels))
    if len(labels) != len(body.labels) or any(example.label not in labels for example in body.examples):
        raise HTTPException(status_code=422, detail="Examples must use unique declared labels")

    ranked, signals = classify_text(body.text, labels, body.examples)
    model, _, _, _ = get_ai_engine()

    similar_matches = find_similar_reports(
        body.text,
        body.candidates,
        query_lat=body.latitude,
        query_lon=body.longitude,
        query_time=body.created_at,
    )

    return AnalysisResponse(
        model_version=MODEL_VERSION if model is not None else "multinomial-nb-bow-v1",
        ranked_labels=ranked,
        signals=signals,
        similar=similar_matches,
    )
