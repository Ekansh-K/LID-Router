"""
Mode A: Single-Language Decode.

The fastest path — when LID is highly confident about one language,
just pick the best backend and decode once.

Average decode calls: 1.0
"""
import numpy as np
from typing import Optional

from src.utils import get_logger, TranscriptResult
from src.asr.backend_selector import select_backend
from src.asr.whisper_backend import WhisperBackend
from src.asr.mms_backend import MMSBackend

log = get_logger("decoding.single")


def decode_single(audio: np.ndarray,
                  language: str,
                  whisper_backend: WhisperBackend,
                  mms_backend: MMSBackend,
                  lid_confidence: float = None) -> TranscriptResult:
    """Run single-language ASR on the top-1 detected language.
    
    Args:
        audio: preprocessed audio segment (float32, 16kHz)
        language: canonical ISO 639-3 code (the LID top-1 pick)
        whisper_backend: loaded WhisperBackend instance
        mms_backend: loaded MMSBackend instance
        lid_confidence: fused LID confidence for backend selection
    
    Returns:
        TranscriptResult from the chosen backend.
    """
    backend_name = select_backend(language, mode="A",
                                  lid_confidence=lid_confidence)
    log.info(f"Mode A: decoding '{language}' with {backend_name}")

    if backend_name == "whisper":
        return whisper_backend.transcribe(audio, language)
    else:
        return mms_backend.transcribe(audio, language)
