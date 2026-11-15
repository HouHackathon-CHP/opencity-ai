# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text Voice Transcription Service using open-source Whisper."""

import io
import logging
import re
from typing import Any

import numpy as np

from app.core.config import DEVICE
from app.services.civic_classifier import classify_text
from app.services.model_loader import get_ai_engine

logger = logging.getLogger("opencity.ai.voice")

_whisper_model = None
_whisper_processor = None
WHISPER_MODEL_NAME = "openai/whisper-tiny"


def load_whisper():
    """Load open-source Whisper model for Vietnamese speech recognition."""
    global _whisper_model, _whisper_processor
    if _whisper_model is not None:
        return _whisper_model, _whisper_processor

    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        logger.info("Loading Whisper model: %s onto %s...", WHISPER_MODEL_NAME, DEVICE)
        _whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_NAME)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        _whisper_model = model
        logger.info("Whisper model loaded successfully.")
    except Exception as exc:
        logger.warning("Could not load Whisper model (%s). Voice service in fallback mode.", exc)
        _whisper_model = None
        _whisper_processor = None

    return _whisper_model, _whisper_processor


def decode_audio_to_16k_mono(audio_bytes: bytes) -> tuple[np.ndarray, float]:
    """Decode audio bytes (WAV, WebM, MP3, OGG, M4A) to 16kHz float32 mono numpy array."""
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

    if not audio_frames:
        return np.zeros(16000, dtype=np.float32), 1.0

    audio_data = np.concatenate(audio_frames, axis=1)[0].astype(np.float32)
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
    """Transcribe audio into text and automatically infer civic routing and urgency."""
    model, processor = load_whisper()
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

    import torch

    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="vi", task="transcribe")
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    # Post-process transcription
    if not transcription:
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
