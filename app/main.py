# SPDX-License-Identifier: Apache-2.0
"""OpenCity AI Microservice Entrypoint.

Modular multi-task Vietnamese civic NLP architecture:
- 1 Shared PhoBERT-base-v2 backbone
- Task 1: Civic Category & Municipal Unit Routing (12 categories -> 4 departments)
- Task 2: Urgency Level Estimation (high / normal / low)
- Task 3: Spatiotemporal Deduplication (L2 normalized embeddings + Haversine + time decay)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analyze_router, health_router, voice_router
from app.services.model_loader import load_model
from app.services.voice_transcriber import load_whisper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("opencity.ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for model pre-warming and graceful shutdown."""
    logger.info("Initializing OpenCity AI Microservice...")
    load_model()
    load_whisper()
    yield
    logger.info("Shutting down OpenCity AI Microservice...")


app = FastAPI(
    title="OpenCity AI Microservice",
    description="Dedicated microservice for Vietnamese civic NLP, triage routing, urgency estimation and deduplication.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular routes
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(voice_router)
