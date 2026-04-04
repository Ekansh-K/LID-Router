"""
Acoustic LID using Meta MMS-LID-4017 (wav2vec2-based, 4017 languages).

This is the PRIMARY LID signal. It takes raw waveform and produces
a probability distribution over 4017 language classes.

GPU required: ~2 GB VRAM in FP16.
"""
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod

from src.utils import get_logger, get_language_map, timed

log = get_logger("lid.acoustic")


class BaseLID(ABC):
    """Interface all LID implementations must follow."""
    @abstractmethod
    def predict(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Return {canonical_lang_code: probability} for detected languages."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass


class AcousticLID(BaseLID):
    """MMS-LID-4017 wrapper.
    
    Key design decisions vs. the plan:
    - Returns canonical ISO 639-3 codes (normalized via LanguageMap)
    - Prunes low-probability outputs (< min_prob) to keep dict manageable
    - Supports FP16 to halve VRAM usage
    - Model can be loaded/unloaded for memory management
    """

    def __init__(self, model_id: str = "facebook/mms-lid-4017",
                 device: str = "cuda",
                 precision: str = "fp16",
                 min_prob: float = 0.001):
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.min_prob = min_prob
        self._model = None
        self._processor = None
        self._id2label = None

    def load(self):
        """Load model into memory. Call explicitly to control timing."""
        import torch
        from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

        log.info(f"Loading AcousticLID ({self.model_id}) on {self.device}...")
        self._processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_id)

        if self.precision == "fp16" and self.device == "cuda":
            self._model = self._model.half()

        self._model = self._model.to(self.device)
        self._model.eval()
        self._id2label = self._model.config.id2label
        log.info(f"AcousticLID loaded — {len(self._id2label)} language classes")

    def unload(self):
        """Free GPU memory."""
        import torch
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        log.info("AcousticLID unloaded")

    def is_loaded(self) -> bool:
        return self._model is not None

    @timed
    def predict(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """Run LID inference. Returns {lang_code: probability}.
        
        The returned codes are the raw MMS-LID codes (ISO 639-3).
        Normalization to canonical codes happens in the fusion module.
        """
        if not self.is_loaded():
            self.load()

        import torch

        inputs = self._processor(
            audio, sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.precision == "fp16" and self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits.float(), dim=-1)[0]  # back to fp32 for softmax

        # Build output dict, pruning low-prob entries
        result = {}
        for idx, prob_val in enumerate(probs):
            p = prob_val.item()
            if p >= self.min_prob:
                lang_code = self._id2label[idx]
                result[lang_code] = p

        return result
