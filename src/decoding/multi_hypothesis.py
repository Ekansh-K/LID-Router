"""
Mode B: Multi-Hypothesis Decode — Phase 1 (fixed).

Phase 1 script-aware reranking — CLUSTER-LEVEL scoring:
  The target script is determined once from the TOP-1 routing candidate and
  applied to ALL hypotheses uniformly.

  WHY: Per-candidate script scoring (previous approach) is broken because:
    - urd candidate: scored on Arab → 1.0 if Perso-Arabic output
    - hin candidate: scored on Deva → 1.0 if Devanagari output
    Both candidates score 1.0 on their own script, so lid_prior decides the
    winner — which always picks the wrong-language top-1 (LID error case).

  FIX: Use the TOP-1 candidate's expected script as the single target:
    - urd is top-1 → target_script = 'Arab'
    - hin hypothesis → Devanagari → Arab score = 0.0  (correctly loses)
    - urd hypothesis → Perso-Arabic → Arab score = 1.0 (correctly wins)

  Same logic for srp/hrv:
    - srp is top-1 → target_script = 'Cyrl'
    - hrv/Latin hypothesis → Cyrl score = 0.0  (correctly loses)
    - srp/Cyrillic hypothesis → Cyrl score = 1.0 (correctly wins)

Scoring formula (all languages scored with SAME target_script):
  combined = w_decoder * decoder_conf
           + w_plausibility * char_plausibility
           + w_lid * lid_prior
           + w_script * script_ratio(text, TOP1_script)
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
    """Fraction of alphabetic characters in `text` that belong to `expected_script`.

    Returns:
        0.5  — when text is empty/digits-only (neutral, not a win)
        0.5  — when expected_script has no configured keywords (unknown)
        0.0–1.0 — actual script match ratio otherwise

    Examples:
      _script_ratio("مرحبا", "Arab")  -> ~1.0
      _script_ratio("hello", "Arab")  -> 0.0
      _script_ratio("नमस्ते", "Deva") -> ~1.0
      _script_ratio("",     "Arab")   -> 0.5 (neutral)
    """
    if not text or not text.strip():
        return 0.5

    keywords = _SCRIPT_KEYWORDS.get(expected_script, [])
    if not keywords:
        return 0.5  # no expectation configured — neutral (not a win)

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
    """Cheap heuristic: how plausible is this transcript text?

    Garbage transcripts tend to have lots of repeated characters or very
    short outputs. Returns 0.0–1.0.
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
    """Decode with multiple language hypotheses and script-aware reranking.

    Args:
        audio: preprocessed audio (float32, 16 kHz)
        candidate_languages: ordered list from routing agent (top-1 first)
        lid_probs: fused LID probability dict
        whisper_backend / mms_backend: loaded backends
        w_*: scoring weights

    Returns:
        (best_transcript: TranscriptResult, all_transcripts: list)
    """
    cmap = _get_confusion_map()
    transcripts: List[TranscriptResult] = []
    scores: List[float] = []

    # ── Cluster-level target script: determined from TOP-1 candidate only ───────
    # ALL hypotheses are scored against this single script.
    # This is what makes script reranking discriminative:
    #   urd top-1 → target='Arab' → hin/Devanagari scores 0.0, urd/Perso-Arabic scores 1.0
    #   srp top-1 → target='Cyrl' → hrv/Latin scores 0.0, srp/Cyrillic scores 1.0
    top1_lang = candidate_languages[0] if candidate_languages else None
    target_script = cmap.expected_script(top1_lang) if top1_lang else None
    use_script_scoring = (target_script is not None
                          and top1_lang is not None
                          and cmap.rerank_by_script(top1_lang))

    orig_total = w_decoder + w_plausibility + w_lid  # normalisation for non-script path

    for i, lang in enumerate(candidate_languages):
        backend_name = select_backend(lang, mode="B")

        try:
            if backend_name == "whisper":
                result = whisper_backend.transcribe(audio, lang)
            else:
                result = mms_backend.transcribe(audio, lang)
            transcripts.append(result)

            decoder_score = result.confidence
            plausibility  = _character_plausibility(result.text)
            lid_prior     = lid_probs.get(lang, 0.0)

            if use_script_scoring:
                # Score against TOP-1's script (same for every candidate)
                script_score = _script_ratio(result.text, target_script)
                combined = (w_decoder      * decoder_score
                          + w_plausibility * plausibility
                          + w_lid          * lid_prior
                          + w_script       * script_score)
            else:
                script_score = 0.0
                combined = ((w_decoder      / orig_total) * decoder_score
                          + (w_plausibility / orig_total) * plausibility
                          + (w_lid          / orig_total) * lid_prior)

            scores.append(combined)

            log.debug(f"  -> {lang} ({result.backend}): "
                     f"decoder={decoder_score:.3f}, plaus={plausibility:.3f}, "
                     f"lid={lid_prior:.3f}, "
                     f"script({target_script})={script_score:.3f} "
                     f"[{'active' if use_script_scoring else 'off'}], "
                     f"combined={combined:.3f}")

            # For top-1 candidate: also try the OTHER backend (picks best model)
            if i == 0:
                alt_backend = "mms" if backend_name == "whisper" else "whisper"
                try:
                    if alt_backend == "whisper":
                        alt_result = whisper_backend.transcribe(audio, lang)
                    else:
                        alt_result = mms_backend.transcribe(audio, lang)
                    transcripts.append(alt_result)

                    if use_script_scoring:
                        alt_script = _script_ratio(alt_result.text, target_script)
                        alt_score  = (w_decoder      * alt_result.confidence
                                    + w_plausibility * _character_plausibility(alt_result.text)
                                    + w_lid          * lid_prior
                                    + w_script       * alt_script)
                    else:
                        alt_script = 0.0
                        alt_score  = ((w_decoder      / orig_total) * alt_result.confidence
                                    + (w_plausibility / orig_total) * _character_plausibility(alt_result.text)
                                    + (w_lid          / orig_total) * lid_prior)

                    scores.append(alt_score)
                    log.debug(f"  -> {lang} ALT ({alt_result.backend}): "
                             f"script({target_script})={alt_script:.3f}, combined={alt_score:.3f}")
                except Exception:
                    pass

        except Exception as e:
            log.warning(f"Failed to decode with language '{lang}': {e}")
            continue

    if not transcripts:
        return TranscriptResult(text="", language="unk", confidence=0.0, backend="none"), []

    best_idx = int(np.argmax(scores))
    best = transcripts[best_idx]
    log.info(f"Mode B: best='{best.language}' "
             f"(score={scores[best_idx]:.3f}, {len(transcripts)} candidates, "
             f"script_target={target_script})")

    return best, transcripts
