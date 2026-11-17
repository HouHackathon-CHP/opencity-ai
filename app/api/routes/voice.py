# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text Voice Transcription API Route."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.security import internal_caller
from app.schemas.analysis import VoiceTranscriptionResponse
from app.services.voice_transcriber import transcribe_audio

router = APIRouter(tags=["Voice AI"])

MAX_AUDIO_BYTES = 25 * 1024 * 104  # 25 MiB


@router.post("/transcribe", response_model=VoiceTranscriptionResponse)
@router.post("/api/opencity/reports/voice-transcribe", response_model=VoiceTranscriptionResponse)
async def transcribe_speech(
    audio: UploadFile = File(..., description="Audio recording file (WebM, WAV, MP3, OGG, M4A)"),
    tenant: str = Depends(internal_caller),
) -> VoiceTranscriptionResponse:
    """Transcribe spoken citizen report, generate title, and predict civic category & urgency."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 25 MiB limit.")

    try:
        result = transcribe_audio(audio_bytes)
        return VoiceTranscriptionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {exc}")
