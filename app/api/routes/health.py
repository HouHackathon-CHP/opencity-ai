# SPDX-License-Identifier: Apache-2.0
"""Health check API router."""

from fastapi import APIRouter

from app.core.config import DEVICE, MODEL_VERSION
from app.schemas.analysis import HealthResponse
from app.services.model_loader import get_ai_engine

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return health status, active AI model backbone, and compute device."""
    model, _, _, is_multitask = get_ai_engine()
    if model is not None:
        ai_status = "phobert-base-v2-multitask" if is_multitask else "phobert-base-v2-12labels"
    else:
        ai_status = "fallback-nb"

    return HealthResponse(
        status="ok",
        service="opencity-ai",
        ai_model=ai_status,
        device=DEVICE,
        model_version=MODEL_VERSION if model is not None else "multinomial-nb-bow-v1",
    )
