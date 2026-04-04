"""
Whisper ASR backend.

Shares the Whisper model with DecoderLID to avoid double-loading.
Used for high-resource languages where Whisper is strong (99 langs).

GPU required: ~3 GB VRAM (shared with DecoderLID).
"""
import numpy as np
from typing import Optional

from src.utils import get_logger, get_language_map, TranscriptResult, timed

log = get_logger("asr.whisper")


class WhisperBackend:
    """Whisper ASR via the openai-whisper library.
    
    Design decisions:
    - Accepts a pre-loaded whisper model to share with DecoderLID.
    - Returns TranscriptResult with avg log-prob as confidence.
    - Temperature=0 for deterministic greedy decoding.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda",
                 whisper_model=None, beam_size: int = 5):
        self.model_size = model_size
        self.device = device
        self._model = whisper_model
        self.beam_size = beam_size
        self._lang_map = get_language_map()

    def load(self):
        if self._model is not None:
            return
        import whisper
        log.info(f"Loading WhisperBackend (whisper-{self.model_size}) on {self.device}...")
        self._model = whisper.load_model(self.model_size, device=self.device)
        log.info("WhisperBackend loaded")

    def set_model(self, model):
        """Accept a pre-loaded model (shared with DecoderLID)."""
        self._model = model

    def unload(self):
        import torch
        del self._model
        self._model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    @timed
    def transcribe(self, audio: np.ndarray, language: str,
                   sr: int = 16000) -> TranscriptResult:
        """Transcribe audio with a specified language.
        
        Args:
            audio: float32 waveform
            language: canonical ISO 639-3 code (converted to whisper code internally)
            sr: sample rate
        
        Returns:
            TranscriptResult with text, confidence, backend name.
        """
        if not self.is_loaded():
            self.load()

        # Convert canonical code to whisper code
        whisper_code = self._lang_map.to_whisper(language)
        if not whisper_code:
            log.warning(f"Language {language} not in Whisper's set — "
                       "this should not happen if backend selection is correct")
            whisper_code = language  # try anyway

        audio_f32 = audio.astype(np.float32)

        result = self._model.transcribe(
            audio_f32,
            language=whisper_code,
            beam_size=self.beam_size,
            temperature=0.0,
            without_timestamps=True,
        )

        text = result["text"].strip()
        # avg_logprob is per-segment; aggregate across segments
        avg_logprob = np.mean([seg["avg_logprob"] for seg in result["segments"]]) \
            if result.get("segments") else -1.0
        # Convert log-prob to a 0-1 confidence score (sigmoid-ish)
        confidence = min(1.0, np.exp(avg_logprob))

        return TranscriptResult(
            text=text,
            language=language,
            confidence=confidence,
            backend="whisper",
            log_probs=avg_logprob,
        )

    @timed
    def transcribe_auto(self, audio: np.ndarray) -> TranscriptResult:
        """Transcribe with Whisper's own language auto-detection (Mode C fallback)."""
        if not self.is_loaded():
            self.load()

        audio_f32 = audio.astype(np.float32)

        result = self._model.transcribe(
            audio_f32,
            language=None,  # auto-detect
            beam_size=self.beam_size,
            temperature=0.0,
            without_timestamps=True,
        )

        text = result["text"].strip()
        detected_whisper = result.get("language", "unknown")
        detected_canon = self._lang_map.from_whisper(detected_whisper) or detected_whisper

        avg_logprob = np.mean([seg["avg_logprob"] for seg in result["segments"]]) \
            if result.get("segments") else -1.0
        confidence = min(1.0, np.exp(avg_logprob))

        return TranscriptResult(
            text=text,
            language=detected_canon,
            confidence=confidence,
            backend="whisper_auto",
            log_probs=avg_logprob,
        )
