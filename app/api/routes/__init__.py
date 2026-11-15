# SPDX-License-Identifier: Apache-2.0
"""API routers."""

from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router
from app.api.routes.voice import router as voice_router

__all__ = ["analyze_router", "health_router", "voice_router"]
