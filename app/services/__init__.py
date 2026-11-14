# SPDX-License-Identifier: Apache-2.0
"""AI Services module."""

from app.services.civic_classifier import classify_text
from app.services.deduplication import find_similar_reports
from app.services.model_loader import get_ai_engine

__all__ = ["classify_text", "find_similar_reports", "get_ai_engine"]
