"""
Evaluation metrics: CER, WER, LID accuracy, and routing statistics.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from src.utils import get_logger

log = get_logger("metrics")


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate via jiwer."""
    from jiwer import cer
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via jiwer."""
    from jiwer import wer
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return wer(reference, hypothesis)


@dataclass
class LIDResult:
    """Stores a single LID evaluation result."""
    true_lang: str
    predicted_lang: str
    top1_prob: float
    correct: bool
    in_top3: bool
    routing_mode: str


@dataclass
class ASRResult:
    """Stores a single ASR evaluation result."""
    true_lang: str
    predicted_lang: str
    reference_text: str
    hypothesis_text: str
    cer: float
    wer: float
    routing_mode: str
    backend: str
    candidates_considered: int
    uncertainty_vec: Optional[List[float]] = None  # [top1_prob, gap, entropy, top3_conc, in_confusion, single_lid_only]


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""
    lid_results: List[LIDResult] = field(default_factory=list)
    asr_results: List[ASRResult] = field(default_factory=list)

    def lid_accuracy(self, lang: Optional[str] = None) -> float:
        """Overall or per-language LID accuracy."""
        items = self.lid_results
        if lang:
            items = [r for r in items if r.true_lang == lang]
        if not items:
            return 0.0
        return sum(r.correct for r in items) / len(items)

    def lid_top3_accuracy(self, lang: Optional[str] = None) -> float:
        items = self.lid_results
        if lang:
            items = [r for r in items if r.true_lang == lang]
        if not items:
            return 0.0
        return sum(r.in_top3 for r in items) / len(items)

    def mean_cer(self, lang: Optional[str] = None) -> float:
        items = self.asr_results
        if lang:
            items = [r for r in items if r.true_lang == lang]
        if not items:
            return 0.0
        return np.mean([r.cer for r in items])

    def mean_wer(self, lang: Optional[str] = None) -> float:
        items = self.asr_results
        if lang:
            items = [r for r in items if r.true_lang == lang]
        if not items:
            return 0.0
        return np.mean([r.wer for r in items])

    def routing_distribution(self) -> Dict[str, int]:
        """Count of utterances per routing mode."""
        dist = defaultdict(int)
        for r in self.lid_results:
            dist[r.routing_mode] += 1
        return dict(dist)

    def avg_decode_calls(self) -> float:
        """Average number of ASR calls per utterance."""
        if not self.asr_results:
            return 0.0
        return np.mean([r.candidates_considered for r in self.asr_results])

    def routing_accuracy(self) -> float:
        """Fraction of Mode A decisions where LID was correct."""
        mode_a = [r for r in self.lid_results if r.routing_mode == "A"]
        if not mode_a:
            return 0.0
        return sum(r.correct for r in mode_a) / len(mode_a)

    def recovery_rate(self) -> float:
        """Fraction of Mode B/C utterances where correct lang was found."""
        bc = [r for r in self.lid_results if r.routing_mode in ("B", "C")]
        if not bc:
            return 0.0
        return sum(r.in_top3 for r in bc) / len(bc)

    def per_language_summary(self) -> Dict[str, dict]:
        """Per-language breakdown of key metrics."""
        langs = set(r.true_lang for r in self.lid_results)
        summary = {}
        for lang in sorted(langs):
            summary[lang] = {
                "lid_accuracy": self.lid_accuracy(lang),
                "lid_top3_accuracy": self.lid_top3_accuracy(lang),
                "mean_cer": self.mean_cer(lang),
                "mean_wer": self.mean_wer(lang),
                "n_samples": len([r for r in self.lid_results if r.true_lang == lang]),
            }
        return summary

    def confusion_matrix_data(self) -> Tuple[List[str], np.ndarray]:
        """Build confusion matrix from LID results.
        
        Returns (lang_list, matrix) where matrix[i][j] = count of
        true=lang_list[i] predicted as lang_list[j].
        """
        langs = sorted(set(r.true_lang for r in self.lid_results))
        lang_to_idx = {l: i for i, l in enumerate(langs)}
        n = len(langs)
        matrix = np.zeros((n, n), dtype=int)

        for r in self.lid_results:
            i = lang_to_idx.get(r.true_lang)
            j = lang_to_idx.get(r.predicted_lang)
            if i is not None and j is not None:
                matrix[i][j] += 1

        return langs, matrix

    def to_dict(self) -> dict:
        """Export summary as a flat dict (for JSON/CSV serialization)."""
        return {
            "overall_lid_accuracy": self.lid_accuracy(),
            "overall_lid_top3_accuracy": self.lid_top3_accuracy(),
            "overall_mean_cer": self.mean_cer(),
            "overall_mean_wer": self.mean_wer(),
            "routing_distribution": self.routing_distribution(),
            "avg_decode_calls": self.avg_decode_calls(),
            "routing_accuracy_mode_a": self.routing_accuracy(),
            "recovery_rate_mode_bc": self.recovery_rate(),
            "n_samples": len(self.lid_results),
            "per_language": self.per_language_summary(),
        }
