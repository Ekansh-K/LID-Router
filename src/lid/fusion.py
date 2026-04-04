"""
Confidence Fusion: combines acoustic (MMS-LID) and decoder (Whisper) LID signals.

This module handles:
1. Language code normalization (MMS uses ISO 639-3, Whisper uses ISO 639-1)
2. Weighted score interpolation
3. Uncertainty signal computation (top1_prob, gap, entropy, etc.)

Design decisions vs. the plan:
- Languages covered by BOTH models get full fusion.
- Languages covered ONLY by MMS-LID get a penalty factor (no confirmation signal).
- Uncertainty signals are computed on the fused distribution.
- The confusion_cluster flag is set here (looked up from the cluster config).
"""
import math
import numpy as np
from typing import Dict, Optional, Tuple

from src.utils import (
    get_logger, get_language_map, load_config, timed,
    UncertaintySignals, LanguageMap,
)

log = get_logger("lid.fusion")


class LIDFusion:
    """Fuses two LID probability distributions and computes uncertainty."""

    def __init__(self, alpha: float = 0.6,
                 single_signal_penalty: float = 0.85,
                 lang_map: Optional[LanguageMap] = None,
                 confusion_clusters: Optional[dict] = None):
        """
        Args:
            alpha: weight for acoustic LID. (1-alpha) for decoder LID.
            single_signal_penalty: multiplier for langs only in one model.
            lang_map: LanguageMap instance. Loaded from config if None.
            confusion_clusters: parsed confusion cluster config.
        """
        self.alpha = alpha
        self.single_signal_penalty = single_signal_penalty
        self.lang_map = lang_map or get_language_map()
        self._confusion_lookup = self._build_confusion_lookup(confusion_clusters)

    def _build_confusion_lookup(self, clusters: Optional[dict]) -> Dict[str, list]:
        """Build lang_code → [partner_codes] lookup from cluster config."""
        lookup: Dict[str, list] = {}
        if clusters is None:
            try:
                from src.utils import load_yaml, _PROJECT_ROOT
                clusters = load_yaml(_PROJECT_ROOT / "config" / "confusion_clusters.yaml")
            except Exception:
                return lookup

        for cluster in clusters.get("clusters", []):
            langs = cluster.get("languages", [])
            for lang in langs:
                partners = [l for l in langs if l != lang]
                if lang in lookup:
                    lookup[lang] = list(set(lookup[lang] + partners))
                else:
                    lookup[lang] = partners
        return lookup

    def _normalize_acoustic_probs(self, raw_probs: Dict[str, float]) -> Dict[str, float]:
        """Convert MMS-LID codes to canonical ISO 639-3 codes.
        
        MMS-LID-4017 already uses ISO 639-3. Codes not in our language_map
        are passed through as-is (they're valid ISO 639-3, just outside our
        30-lang evaluation subset, and may still be valid MMS adapter names).
        """
        normalized = {}
        for code, prob in raw_probs.items():
            # Try to find canonical code via reverse lookup
            canon = self.lang_map.from_mms_lid(code)
            key = canon if canon else code
            normalized[key] = prob
        return normalized

    def _normalize_decoder_probs(self, raw_probs: Dict[str, float]) -> Dict[str, float]:
        """Convert Whisper codes (en, zh, etc.) to canonical ISO 639-3.
        
        Codes that don't map to any entry in language_map are dropped —
        they can't be reliably routed by the ASR backends.
        """
        normalized = {}
        for code, prob in raw_probs.items():
            canon = self.lang_map.from_whisper(code)
            if canon is None:
                # Skip unmapped Whisper codes — don't leak raw ISO 639-1 codes
                # into fused_probs where ASR backends can't handle them.
                continue
            normalized[canon] = normalized.get(canon, 0.0) + prob
        return normalized

    @timed
    def fuse(self, acoustic_probs: Dict[str, float],
             decoder_probs: Dict[str, float]) -> Dict[str, float]:
        """Weighted interpolation of two LID distributions.
        
        Returns fused scores in canonical language codes.
        """
        # Normalize both to canonical codes
        a_probs = self._normalize_acoustic_probs(acoustic_probs)
        d_probs = self._normalize_decoder_probs(decoder_probs)

        all_langs = set(a_probs.keys()) | set(d_probs.keys())
        fused = {}

        for lang in all_langs:
            a_score = a_probs.get(lang, 0.0)
            d_score = d_probs.get(lang, 0.0)

            in_both = (a_score > 0) and (d_score > 0)
            in_acoustic_only = (a_score > 0) and (d_score == 0)
            in_decoder_only = (a_score == 0) and (d_score > 0)

            if in_both:
                score = self.alpha * a_score + (1 - self.alpha) * d_score
            elif in_acoustic_only:
                # Language not in Whisper's 99 → use acoustic only with penalty
                score = a_score * self.single_signal_penalty
            elif in_decoder_only:
                # Rare: language only in Whisper but not in MMS → use decoder only with penalty
                score = d_score * self.single_signal_penalty
            else:
                score = 0.0

            if score > 0.0001:
                fused[lang] = score

        # Re-normalize to ensure it's a valid distribution
        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}

        return fused

    def compute_uncertainty(self, fused_probs: Dict[str, float]) -> UncertaintySignals:
        """Compute uncertainty signals from the fused distribution."""
        if not fused_probs:
            return UncertaintySignals()

        sorted_probs = sorted(fused_probs.values(), reverse=True)
        sorted_langs = sorted(fused_probs, key=fused_probs.get, reverse=True)

        top1_prob = sorted_probs[0]
        top1_lang = sorted_langs[0]
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        # Shannon entropy (higher = more uncertain)
        entropy = 0.0
        for p in sorted_probs:
            if p > 0:
                entropy -= p * math.log2(p)

        # Top-3 concentration (what fraction of probability mass is in top 3)
        top3_conc = sum(sorted_probs[:3])

        # Check confusion cluster membership
        in_confusion = top1_lang in self._confusion_lookup

        return UncertaintySignals(
            top1_prob=top1_prob,
            gap=gap,
            entropy=entropy,
            top3_concentration=top3_conc,
            in_confusion_cluster=in_confusion,
            single_lid_only=False,  # updated by pipeline if applicable
        )

    def fuse_and_analyze(self, acoustic_probs: Dict[str, float],
                         decoder_probs: Dict[str, float]
                         ) -> Tuple[Dict[str, float], UncertaintySignals]:
        """Convenience: fuse + compute uncertainty in one call."""
        fused = self.fuse(acoustic_probs, decoder_probs)
        uncertainty = self.compute_uncertainty(fused)
        return fused, uncertainty

    def get_confusion_partners(self, lang: str) -> list:
        """Return known confusion partners for a language."""
        return self._confusion_lookup.get(lang, [])
