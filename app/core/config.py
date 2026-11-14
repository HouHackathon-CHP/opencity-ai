# SPDX-License-Identifier: Apache-2.0
"""Configuration settings for OpenCity AI Microservice."""

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MULTITASK_DIR = MODELS_DIR / "phobert_multitask"

INTERNAL_SECRET = os.getenv("INTERNAL_GATEWAY_SECRET", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "phobert-base-v2-multitask-civic")
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

# Device configuration (auto-detect CUDA if available, fallback to CPU)
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

DEVICE = "cuda" if CUDA_AVAILABLE and os.getenv("FORCE_CPU", "0") != "1" else "cpu"
