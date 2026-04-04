"""
Decoder-based LID using Whisper's built-in language detection.

This is the SECONDARY LID signal — complementary to the acoustic (encoder-based)
MMS-LID. Whisper's detect_language runs the encoder then scores language tokens
through the decoder, making it a genuinely different approach.

Covers 99 languages. For languages outside Whisper's set, this signal is absent
and fusion falls back to acoustic-only.

GPU required: ~3 GB VRAM in FP16 (shares the model with WhisperBackend ASR).
"""
import numpy as np
from typing import Dict, Optional

from src.lid.acoustic_lid import BaseLID
from src.utils import get_logger, timed

log = get_logger("lid.decoder")


class DecoderLID(BaseLID):
    """Whisper language detection wrapper.
    
    Key design decisions:
    - Uses the openai-whisper library (not HF transformers) because
      its detect_language() API directly returns clean probabilities.
    - Can share the same loaded Whisper model with WhisperBackend ASR
      to avoid double-loading (~3 GB VRAM saved).
    - Returns Whisper's native codes (e.g., "en", "zh"). The fusion 
      module normalizes them to canonical ISO 639-3 via LanguageMap.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda",
                 whisper_model=None):
        """
        Args:
            model_size: Whisper model variant.
            device: "cuda" or "cpu".
            whisper_model: Optional pre-loaded whisper model instance.
                          Pass this to share the model with WhisperBackend.
        """
        self.model_size = model_size
        self.device = device
        self._model = whisper_model

    def load(self):
        """Load Whisper model. Skip if model was passed via constructor."""
        if self._model is not None:
            log.info("DecoderLID using pre-loaded Whisper model")
            return
        import whisper
        log.info(f"Loading DecoderLID (whisper-{self.model_size}) on {self.device}...")
        self._model = whisper.load_model(self.model_size, device=self.device)
        log.info("DecoderLID loaded")

    def unload(self):
        import torch
        del self._model
        self._model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        log.info("DecoderLID unloaded")

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model(self):
        """Expose the underlying Whisper model for sharing with ASR backend."""
        if not self.is_loaded():
            self.load()
        return self._model

    @timed
    def predict(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Detect language using Whisper's decoder-based approach.
        
        Returns {whisper_lang_code: probability} for all 99 supported languages.
        """
        if not self.is_loaded():
            self.load()

        import whisper
        import torch

        # Whisper expects float32, exactly 30s padded/trimmed
        audio_f32 = audio.astype(np.float32)
        audio_padded = whisper.pad_or_trim(audio_f32)

        mel = whisper.log_mel_spectrogram(
            audio_padded, n_mels=self._model.dims.n_mels
        ).to(self.device)

        with torch.no_grad():
            _, probs = self._model.detect_language(mel)

        # probs is already a dict {lang_code: float}
        return dict(probs)
