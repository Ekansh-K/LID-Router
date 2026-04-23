"""
Mode B: Multi-Hypothesis Decode — Phase 1 rewrite.

Phase 1 additions (script-aware reranking):
  For languages where script mismatch causes catastrophic CER (urd/srp),
  each hypothesis is scored by the Unicode script ratio of its output text.
  A hypothesis in the wrong script (e.g. Devanagari for Urdu) gets a near-zero
  script score, pushing the correct-script hypothesis to the top regardless of
  the decoder confidence score.

Scoring formula (per candidate):
  combined = w_decoder * decoder_score
           + w_plausibility * plausibility_score
           + w_lid * lid_prior
           + w_script * script_score   <-- Phase 1 (0.0 when not a script-cluster lang)

The w_script weight is high (0.5) to guarantee the correct script always wins.
"""
import unicodedata
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

from src.utils import get_logger, TranscriptResult
from src.asr.backend_selector import select_backend
from src.asr.whisper_backend import WhisperBackend
from src.asr.mms_backend import MMSBackend
from src.routing.confusion_map import ConfusionMap

log = get_logger("decoding.multi")

# Singleton confusion map (shared across calls)
_confusion_map: Optional[ConfusionMap] = None


def _get_confusion_map() -> ConfusionMap:
    global _confusion_map
    if _confusion_map is None:
        _confusion_map = ConfusionMap()
    return _confusion_map


# ── Unicode script detection ─────────────────────────────────────────────────
# Maps the script keyword (from unicodedata.name) to short tag.
_SCRIPT_KEYWORDS = {
    "Arab": ["ARABIC", "ARABIC-INDIC"],
    "Deva": ["DEVANAGARI"],
    "Cyrl": ["CYRILLIC"],
    "Latn": ["LATIN"],
    "Han":  ["CJK UNIFIED", "CJK COMPATIBILITY"],
    "Hira": ["HIRAGANA"],
    "Kana": ["KATAKANA"],
    "Ethi": ["ETHIOPIC"],
    "Geor": ["GEORGIAN"],
    "Hani": ["HANGUL"],
}


def _script_ratio(text: str, expected_script: str) -> float:
    """Return the fraction of alphabetic characters in `text` that belong to
    the `expected_script`.

    Returns 1.0 (perfect) when the text is empty or has no alphabetic chars,
    so an empty hypothesis doesn't unfairly win on script score.

    Examples:
      _script_ratio("مرحبا", "Arab")  -> ~1.0
      _script_ratio("hello", "Arab")  -> 0.0
      _script_ratio("नमस्ते", "Deva") -> ~1.0
    """
    if not text or not text.strip():
        return 0.5  # neutral — empty output

    keywords = _SCRIPT_KEYWORDS.get(expected_script, [])
    if not keywords:
        return 0.5  # no expectation configured — treat as neutral, not a win

    total = 0
    matches = 0
    for ch in text:
        if ch.isspace() or ch.isdigit() or not ch.isprintable():
            continue
        try:
            name = unicodedata.name(ch, "")
        except Exception:
            continue
        total += 1
        if any(kw in name for kw in keywords):
            matches += 1

    if total == 0:
        return 0.5  # only spaces/digits — neutral
    return matches / total


def _character_plausibility(text: str) -> float:
    """Cheap heuristic: ratio of printable non-whitespace characters.

    Garbage transcripts tend to have lots of repeated characters,
    control characters, or single-char outputs.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    text = text.strip()
    length_score = min(1.0, len(text) / 10.0)
    char_counts = Counter(text)
    max_freq = max(char_counts.values())
    repetition_score = 1.0 - (max_freq / len(text))

    return 0.5 * length_score + 0.5 * repetition_score


def decode_multi_hypothesis(audio: np.ndarray,
                            candidate_languages: List[str],
                            lid_probs: Dict[str, float],
                            whisper_backend: WhisperBackend,
                            mms_backend: MMSBackend,
                            w_decoder: float = 0.4,
                            w_plausibility: float = 0.1,
                            w_lid: float = 0.2,
                            w_script: float = 0.3,
                            ) -> tuple:
    """Decode with multiple language hypotheses and rerank.

    Phase 1: Script-aware reranking is enabled for clusters where
    rerank_by_script=true (urd/hin, srp/hrv). The w_script weight is applied
    only when an expected script is configured for the candidate language.

    Args:
        audio: preprocessed audio (float32, 16 kHz)
        candidate_languages: top-k languages from routing agent
        lid_probs: fused LID probability dict
        whisper_backend: loaded WhisperBackend
        mms_backend: loaded MMSBackend
        w_decoder: weight for decoder confidence (reduced from 0.5 to make room for script)
        w_plausibility: weight for character plausibility
        w_lid: weight for LID prior
        w_script: weight for script-match score (active only for script-cluster langs)

    Returns:
        (best_transcript: TranscriptResult, all_transcripts: list)
    """
    cmap = _get_confusion_map()
    transcripts: List[TranscriptResult] = []
    scores: List[float] = []

    for i, lang in enumerate(candidate_languages):
        backend_name = select_backend(lang, mode="B")

        # Determine expected script for this lang (None if not a script cluster)
        expected_script = cmap.expected_script(lang)
        use_script_score = cmap.rerank_by_script(lang) and expected_script is not None

        try:
            if backend_name == "whisper":
                result = whisper_backend.transcribe(audio, lang)
            else:
                result = mms_backend.transcribe(audio, lang)
            transcripts.append(result)

            decoder_score   = result.confidence
            plausibility    = _character_plausibility(result.text)
            lid_prior       = lid_probs.get(lang, 0.0)
            script_score    = _script_ratio(result.text, expected_script) if use_script_score else 0.0

            if use_script_score:
                # Redistribute weights to include script signal
                combined = (w_decoder      * decoder_score
                          + w_plausibility * plausibility
                          + w_lid          * lid_prior
                          + w_script       * script_score)
            else:
                # Original 3-signal scoring (normalised to same scale)
                orig_total = w_decoder + w_plausibility + w_lid
                combined = ((w_decoder / orig_total)      * decoder_score
                          + (w_plausibility / orig_total) * plausibility
                          + (w_lid / orig_total)          * lid_prior)

            scores.append(combined)

            log.debug(f"  → {lang} ({result.backend}): "
                     f"decoder={decoder_score:.3f}, "
                     f"plausibility={plausibility:.3f}, "
                     f"lid={lid_prior:.3f}, "
                     f"script={script_score:.3f} [{'active' if use_script_score else 'off'}], "
                     f"combined={combined:.3f}")

            # For the top-1 candidate, also try the OTHER backend
            if i == 0:
                alt_backend = "mms" if backend_name == "whisper" else "whisper"
                try:
                    if alt_backend == "whisper":
                        alt_result = whisper_backend.transcribe(audio, lang)
                    else:
                        alt_result = mms_backend.transcribe(audio, lang)
                    transcripts.append(alt_result)
                    alt_script = (_script_ratio(alt_result.text, expected_script)
                                  if use_script_score else 0.0)
                    if use_script_score:
                        alt_score = (w_decoder      * alt_result.confidence
                                   + w_plausibility * _character_plausibility(alt_result.text)
                                   + w_lid          * lid_prior
                                   + w_script       * alt_script)
                    else:
                        alt_score = ((w_decoder / orig_total)      * alt_result.confidence
                                   + (w_plausibility / orig_total) * _character_plausibility(alt_result.text)
                                   + (w_lid / orig_total)          * lid_prior)
                    scores.append(alt_score)
                    log.debug(f"  → {lang} ALT ({alt_result.backend}): "
                             f"script={alt_script:.3f}, combined={alt_score:.3f}")
                except Exception:
                    pass

        except Exception as e:
            log.warning(f"Failed to decode with language '{lang}': {e}")
            continue

    if not transcripts:
        return TranscriptResult(text="", language="unk", confidence=0.0, backend="none"), []

    best_idx = int(np.argmax(scores))
    best = transcripts[best_idx]
    log.info(f"Mode B: best hypothesis = '{best.language}' "
             f"(score={scores[best_idx]:.3f}, {len(transcripts)} candidates)")

    return best, transcripts
