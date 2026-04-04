"""
MMS-1b-all ASR backend with adapter switching.

Covers 1162 languages via lightweight adapters (~1MB each).
Adapter switching is fast (~50ms) — ideal for multi-hypothesis decoding.

GPU required: ~2 GB VRAM in FP16.
"""
import numpy as np
from typing import Optional

from src.utils import get_logger, get_language_map, TranscriptResult, timed

log = get_logger("asr.mms")


class MMSBackend:
    """MMS-1b-all ASR via HuggingFace Transformers.
    
    Design decisions:
    - Adapter switching tracked via _current_lang to avoid redundant loads.
    - Logit-based confidence: mean of per-frame max softmax probabilities.
    - FP16 support for memory efficiency.
    """

    def __init__(self, model_id: str = "facebook/mms-1b-all",
                 device: str = "cuda",
                 precision: str = "fp16"):
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self._model = None
        self._processor = None
        self._current_lang: Optional[str] = None
        self._lang_map = get_language_map()

    def load(self):
        import torch
        from transformers import Wav2Vec2ForCTC, AutoProcessor

        log.info(f"Loading MMSBackend ({self.model_id}) on {self.device}...")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

        if self.precision == "fp16" and self.device == "cuda":
            self._model = self._model.half()

        self._model = self._model.to(self.device)
        self._model.eval()
        log.info("MMSBackend loaded")

    def unload(self):
        import torch
        del self._model, self._processor
        self._model = None
        self._processor = None
        self._current_lang = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    def _switch_adapter(self, lang: str):
        """Load the language-specific adapter. Fast operation (~50ms)."""
        # Convert canonical code to MMS-ASR code
        mms_code = self._lang_map.to_mms_asr(lang) or lang

        if mms_code == self._current_lang:
            return  # already loaded

        try:
            self._processor.tokenizer.set_target_lang(mms_code)
            self._model.load_adapter(mms_code)
            self._current_lang = mms_code
            log.debug(f"Switched MMS adapter to '{mms_code}'")
        except Exception as e:
            log.error(f"Failed to load adapter for '{mms_code}': {e}")
            raise

    @timed
    def transcribe(self, audio: np.ndarray, language: str,
                   sr: int = 16000) -> TranscriptResult:
        """Transcribe audio with a specific language adapter.
        
        Args:
            audio: float32 waveform
            language: canonical ISO 639-3 code
        """
        if not self.is_loaded():
            self.load()

        import torch

        self._switch_adapter(language)

        inputs = self._processor(
            audio.astype(np.float32),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.precision == "fp16" and self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # Greedy decoding
        predicted_ids = torch.argmax(logits.float(), dim=-1)[0]
        transcription = self._processor.decode(predicted_ids)

        # Confidence: mean max-softmax across frames
        frame_probs = torch.softmax(logits.float(), dim=-1)[0]
        max_probs = frame_probs.max(dim=-1).values
        confidence = max_probs.mean().item()

        return TranscriptResult(
            text=transcription.strip(),
            language=language,
            confidence=confidence,
            backend="mms",
        )
