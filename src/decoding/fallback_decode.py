"""
Mode C: Fallback Decode — tiered fallback for low-confidence LID.

Tier 1: Restrict to language family of top-1 guess, run multi-hypothesis within it.
Tier 2: Broad MMS decode with top-5 most probable adapters.
Tier 3: Whisper auto-detect (completely independent pipeline).

The best result across tiers is returned, flagged as low-confidence.

Average decode calls: 3-6.
"""
import numpy as np
from typing import Dict, List, Optional

from src.utils import (
    get_logger, get_language_map, TranscriptResult
)
from src.asr.whisper_backend import WhisperBackend
from src.asr.mms_backend import MMSBackend
from src.decoding.multi_hypothesis import decode_multi_hypothesis, _character_plausibility

log = get_logger("decoding.fallback")


def decode_fallback(audio: np.ndarray,
                    candidate_languages: List[str],
                    lid_probs: Dict[str, float],
                    whisper_backend: WhisperBackend,
                    mms_backend: MMSBackend,
                    ) -> tuple:
    """Tiered fallback decoding for uncertain LID.
    
    Returns:
        (best_transcript: TranscriptResult, all_transcripts: list)
    """
    lang_map = get_language_map()
    all_results: List[TranscriptResult] = []

    # ── Tier 1: Family-aware restriction ──
    top1 = candidate_languages[0] if candidate_languages else None
    if top1:
        family = lang_map.get_family(top1)
        if family:
            family_langs = lang_map.langs_in_family(family)
            # Intersect with languages that have non-zero LID probability
            family_candidates = [l for l in family_langs if l in lid_probs]
            if family_candidates:
                log.info(f"Mode C Tier 1: family '{family}' → {family_candidates}")
                best_t1, tier1_results = decode_multi_hypothesis(
                    audio, family_candidates[:5], lid_probs,
                    whisper_backend, mms_backend
                )
                all_results.extend(tier1_results)

    # ── Tier 2: Broad MMS decode with top-5 languages ──
    mms_candidates = candidate_languages[:5]
    log.info(f"Mode C Tier 2: MMS broad decode → {mms_candidates}")
    for lang in mms_candidates:
        try:
            result = mms_backend.transcribe(audio, lang)
            all_results.append(result)
        except Exception as e:
            log.warning(f"Tier 2: MMS decode failed for '{lang}': {e}")

    # ── Tier 3: Whisper auto-detect ──
    log.info("Mode C Tier 3: Whisper auto-detect")
    try:
        auto_result = whisper_backend.transcribe_auto(audio)
        all_results.append(auto_result)
    except Exception as e:
        log.warning(f"Tier 3: Whisper auto failed: {e}")

    if not all_results:
        return TranscriptResult(text="", language="unk", confidence=0.0, backend="none"), []

    # ── Pick best across all tiers ──
    # Score by decoder confidence + character plausibility
    scores = []
    for r in all_results:
        plaus = _character_plausibility(r.text)
        lid_prior = lid_probs.get(r.language, 0.05)
        score = 0.5 * r.confidence + 0.3 * plaus + 0.2 * lid_prior
        scores.append(score)

    best_idx = int(np.argmax(scores))
    best = all_results[best_idx]
    log.info(f"Mode C: best result = '{best.language}' from {best.backend} "
             f"(score={scores[best_idx]:.3f}, {len(all_results)} total candidates)")

    return best, all_results
