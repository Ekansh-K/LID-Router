"""
Baseline LID using SpeechBrain ECAPA-TDNN (107 languages, ~20M params).

This is NOT used in the main pipeline — only for ablation experiments (A3)
to measure whether the much larger MMS-LID-4017 actually helps.

GPU required: ~0.1 GB — trivially small.
"""
import numpy as np
from typing import Dict

from src.lid.acoustic_lid import BaseLID
from src.utils import get_logger, timed

log = get_logger("lid.baseline")


class BaselineLID(BaseLID):
    """SpeechBrain ECAPA-TDNN language identification.
    
    Returns VoxLingua107 language codes which need normalization.
    """

    def __init__(self, model_id: str = "speechbrain/lang-id-voxlingua107-ecapa",
                 device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._classifier = None

    def load(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        log.info(f"Loading BaselineLID ({self.model_id}) on {self.device}...")
        self._classifier = EncoderClassifier.from_hparams(
            source=self.model_id,
            run_opts={"device": self.device},
        )
        log.info("BaselineLID loaded — 107 language classes")

    def unload(self):
        import torch
        del self._classifier
        self._classifier = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        log.info("BaselineLID unloaded")

    def is_loaded(self) -> bool:
        return self._classifier is not None

    @timed
    def predict(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        if not self.is_loaded():
            self.load()

        import torch

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._classifier.classify_batch(audio_tensor)
            # output is (posterior, score, index, label)
            posteriors = output[0].squeeze(0)  # shape: (107,)
            # SpeechBrain returns log-posteriors; convert to probabilities
            probs = torch.softmax(posteriors, dim=-1)

        # Get label list
        label_encoder = self._classifier.hparams.label_encoder
        result = {}
        for idx, p in enumerate(probs):
            p_val = p.item()
            if p_val >= 0.001:
                lang_label = label_encoder.decode_ndim(idx)
                # SpeechBrain uses labels like "en: English" — extract code
                lang_code = lang_label.split(":")[0].strip() if ":" in lang_label else lang_label
                result[lang_code] = p_val

        return result
