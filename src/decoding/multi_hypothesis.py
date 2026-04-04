"""
Mode B: Multi-Hypothesis Decode.

Run ASR with top-k candidate languages, then rerank the transcripts
to pick the best one. This recovers from LID errors when the correct
language is among the top candidates but not necessarily #1.

Average decode calls: 2-3 (up to top_k + confusion partners).

Reranking signals:
  1. ASR decoder confidence (avg log-prob / max-softmax)
  2. Character plausibility (ratio of valid characters for the language)
  3. LID prior (original LID probability for that language)
"""
import numpy as np
from typing import List, Dict
from collections import Counter

from src.utils import get_logger, TranscriptResult
from src.asr.backend_selector import select_backend
from src.asr.whisper_backend import WhisperBackend
from src.asr.mms_backend import MMSBackend

log = get_logger("decoding.multi")


def _character_plausibility(text: str) -> float:
    """Cheap heuristic: ratio of printable non-whitespace characters.
    
    Garbage transcripts tend to have lots of repeated characters,
    control characters, or single-char outputs. This catches the
    worst failures without needing an external LM.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    text = text.strip()
    # Penalize very short outputs
    length_score = min(1.0, len(text) / 10.0)

    # Penalize high character repetition (e.g., "aaaaaaa")
    char_counts = Counter(text)
    max_freq = max(char_counts.values())
    repetition_score = 1.0 - (max_freq / len(text))

    return 0.5 * length_score + 0.5 * repetition_score


def decode_multi_hypothesis(audio: np.ndarray,
                            candidate_languages: List[str],
                            lid_probs: Dict[str, float],
                            whisper_backend: WhisperBackend,
                            mms_backend: MMSBackend,
                            w_decoder: float = 0.5,
                            w_plausibility: float = 0.2,
                            w_lid: float = 0.3,
                            ) -> tuple:
    """Decode with multiple language hypotheses and rerank.
    
    Args:
        audio: preprocessed audio
        candidate_languages: top-k languages from routing agent
        lid_probs: fused LID probability dict
        whisper_backend: loaded WhisperBackend
        mms_backend: loaded MMSBackend
        w_decoder: weight for decoder confidence in reranking
        w_plausibility: weight for character plausibility
        w_lid: weight for LID prior
    
    Returns:
        (best_transcript: TranscriptResult, all_transcripts: list)
    """
    transcripts: List[TranscriptResult] = []
    scores: List[float] = []

    for lang in candidate_languages:
        backend_name = select_backend(lang, mode="B")

        try:
            if backend_name == "whisper":
                result = whisper_backend.transcribe(audio, lang)
            else:
                result = mms_backend.transcribe(audio, lang)
            transcripts.append(result)

            # Compute reranking score
            decoder_score = result.confidence
            plausibility = _character_plausibility(result.text)
            lid_prior = lid_probs.get(lang, 0.0)

            combined = (w_decoder * decoder_score +
                        w_plausibility * plausibility +
                        w_lid * lid_prior)
            scores.append(combined)

            log.debug(f"  → {lang} ({result.backend}): "
                     f"decoder={decoder_score:.3f}, "
                     f"plausibility={plausibility:.3f}, "
                     f"lid={lid_prior:.3f}, "
                     f"combined={combined:.3f}")
        except Exception as e:
            log.warning(f"Failed to decode with language '{lang}': {e}")
            continue

    if not transcripts:
        # All decoding attempts failed — return empty result
        return TranscriptResult(text="", language="unk", confidence=0.0, backend="none"), []

    # Pick the best
    best_idx = int(np.argmax(scores))
    best = transcripts[best_idx]
    log.info(f"Mode B: best hypothesis = '{best.language}' "
             f"(score={scores[best_idx]:.3f}, {len(transcripts)} candidates)")

    return best, transcripts
