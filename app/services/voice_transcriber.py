# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text Voice Transcription Service using Wav2Vec2 Vietnamese (CTC Architecture)."""

import io
import logging
import os
import re
from typing import Any

import numpy as np

from app.core.config import DEVICE, MODELS_DIR
from app.services.civic_classifier import classify_text
from app.services.model_loader import get_ai_engine

logger = logging.getLogger("opencity.ai.voice")

_voice_model = None
_voice_processor = None
WAV2VEC2_LOCAL_DIR = MODELS_DIR / "wav2vec2_vi"
DEFAULT_MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
VOICE_MODEL_NAME = os.getenv(
    "VOICE_MODEL_NAME",
    os.getenv(
        "PHOWHISPER_MODEL_NAME",
        str(WAV2VEC2_LOCAL_DIR) if (WAV2VEC2_LOCAL_DIR / "model.safetensors").exists() else DEFAULT_MODEL_NAME,
    ),
)


def load_voice_model():
    """Load open-source Wav2Vec2 Vietnamese ASR model trained on 13,000+ hours of speech."""
    global _voice_model, _voice_processor
    if _voice_model is not None:
        return _voice_model, _voice_processor

    try:
        from transformers import AutoModelForCTC, AutoProcessor

        logger.info("Loading Vietnamese ASR model (Wav2Vec2): %s onto %s...", VOICE_MODEL_NAME, DEVICE)
        _voice_processor = AutoProcessor.from_pretrained(VOICE_MODEL_NAME)
        model = AutoModelForCTC.from_pretrained(VOICE_MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        _voice_model = model
        logger.info("Wav2Vec2 Vietnamese ASR model loaded successfully.")
    except Exception as exc:
        logger.warning("Could not load Wav2Vec2 model (%s). Voice service in fallback mode.", exc)
        _voice_model = None
        _voice_processor = None

    return _voice_model, _voice_processor


# Backward-compatible aliases
load_phowhisper = load_voice_model
load_whisper = load_voice_model


def decode_audio_to_16k_mono(audio_bytes: bytes) -> tuple[np.ndarray, float]:
    """Decode audio bytes (WAV, WebM, MP3, OGG, M4A) to 16kHz float32 mono numpy array with volume normalization."""
    import av

    container = av.open(io.BytesIO(audio_bytes))
    if not container.streams.audio:
        raise ValueError("No audio stream found in the uploaded file.")

    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    audio_frames = []

    for frame in container.decode(stream):
        resampled = resampler.resample(frame)
        if resampled:
            for r in resampled:
                audio_frames.append(r.to_ndarray())

    # Flush all remaining frames in the resampler buffer to prevent dropping initial/final words
    flushed = resampler.resample(None)
    if flushed:
        for r in flushed:
            audio_frames.append(r.to_ndarray())

    if not audio_frames:
        return np.zeros(16000, dtype=np.float32), 1.0

    audio_data = np.concatenate(audio_frames, axis=1)[0].astype(np.float32)

    # Normalize audio gain so soft microphone recording doesn't cause consonant/vowel distortion
    max_amp = float(np.abs(audio_data).max())
    if max_amp > 1e-4:
        audio_data = (audio_data / max_amp) * 0.95

    duration_seconds = round(len(audio_data) / 16000.0, 2)
    return audio_data, duration_seconds


def generate_suggested_title(text: str) -> str:
    """Generate a concise title from transcribed civic report text."""
    clean = text.strip()
    clean = re.sub(r"^(tôi muốn phản ánh là|tôi xin phản ánh|phản ánh là|báo cáo là)\s+", "", clean, flags=re.I)
    # Take first sentence or first 12 words
    sentences = re.split(r"[.?!;\n]", clean)
    first_sentence = sentences[0].strip() if sentences else clean
    words = first_sentence.split()
    if len(words) > 10:
        return " ".join(words[:10]) + "…"
    return first_sentence[:80] if first_sentence else "Phản ánh bằng giọng nói"


UNIT_TO_VIETNAMESE = {
    "transport": "Giao thông",
    "infrastructure": "Hạ tầng & chiếu sáng",
    "environment": "Môi trường & cây xanh",
    "intake": "Tiếp nhận chung",
}


def transcribe_audio(audio_bytes: bytes) -> dict[str, Any]:
    """Transcribe audio into text using Wav2Vec2 Vietnamese CTC and infer civic routing."""
    model, processor = load_voice_model()
    audio_data, duration = decode_audio_to_16k_mono(audio_bytes)

    if model is None or processor is None:
        return {
            "text": "Không thể xử lý âm thanh (mô hình chưa sẵn sàng).",
            "suggested_title": "Phản ánh bằng giọng nói",
            "suggested_category": "Tiếp nhận chung",
            "suggested_unit": "intake",
            "urgency": "normal",
            "duration_seconds": duration,
        }

    max_amp = float(np.abs(audio_data).max()) if len(audio_data) > 0 else 0.0
    if max_amp < 0.005 or duration < 0.3:
        return {
            "text": "Âm thanh không rõ hoặc có nhiều tiếng ồn.",
            "suggested_title": "Phản ánh bằng giọng nói",
            "suggested_category": "Tiếp nhận chung",
            "suggested_unit": "intake",
            "urgency": "normal",
            "duration_seconds": duration,
        }

    import torch

    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        raw_text = processor.batch_decode(pred_ids)[0].strip()

    # Format transcribed Vietnamese text with proper capitalization and punctuation
    if raw_text:
        transcription = raw_text[0].upper() + raw_text[1:]
        if not transcription.endswith((".", "!", "?")):
            transcription += "."
    else:
        transcription = "Âm thanh không rõ hoặc có nhiều tiếng ồn."

    # Run PhoBERT Multi-Task on the transcribed text for instant civic routing
    ranked_units, _ = classify_text(
        transcription,
        ["transport", "infrastructure", "environment", "intake"],
        [],
    )
    ranked_urgency, _ = classify_text(
        transcription,
        ["high", "normal", "low"],
        [],
    )

    top_unit = ranked_units[0]["label"] if ranked_units else "intake"
    urgency = ranked_urgency[0]["label"] if ranked_urgency else "normal"
    suggested_category = UNIT_TO_VIETNAMESE.get(top_unit, "Tiếp nhận chung")
    suggested_title = generate_suggested_title(transcription)

    return {
        "text": transcription,
        "suggested_title": suggested_title,
        "suggested_category": suggested_category,
        "suggested_unit": top_unit,
        "urgency": urgency,
        "duration_seconds": duration,
    }
