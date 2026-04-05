"""
Backend selector: decides whether to use Whisper or MMS for a given language.

Track 3 — Confidence-Weighted MMS Fallback:
  Uses per-language backend quality data (measured CER on FLEURS test set)
  to pick the best backend instead of always defaulting to Whisper.

  Key insight: MMS is better on 22/28 languages.
  Only 6 languages (arb, cmn, hin, jpn, srp, urd) genuinely prefer Whisper.
"""
from typing import Optional
from pathlib import Path
from src.utils import get_logger, load_yaml

log = get_logger("asr.selector")

# ── Singleton: per-language backend quality table ──
_backend_prefs = None
_default_backend = "mms"


def _load_backend_prefs() -> dict:
    """Load config/backend_quality.yaml once and cache."""
    global _backend_prefs, _default_backend
    if _backend_prefs is None:
        config_path = (Path(__file__).resolve().parent.parent.parent
                       / "config" / "backend_quality.yaml")
        try:
            data = load_yaml(config_path)
            _backend_prefs = data.get("backend_preferences", {})
            _default_backend = data.get("default_backend", "mms")
        except FileNotFoundError:
            log.warning("backend_quality.yaml not found — defaulting to MMS")
            _backend_prefs = {}
    return _backend_prefs


def select_backend(language: str, mode: str = "A",
                   lid_confidence: Optional[float] = None) -> str:
    """Choose 'whisper' or 'mms' for a given language and routing mode.

    Track 3 logic:
      1. Per-language measured CER picks the empirically better backend.
      2. When LID confidence is low (< 0.5), always prefer MMS (safer default
         because MMS adapters degrade more gracefully on wrong-language input).
      3. Mode B uses preferred backend per language (not always MMS).

    Args:
        language: canonical ISO 639-3 code
        mode: routing mode ("A", "B", "C")
        lid_confidence: optional fused LID confidence (0-1)

    Returns:
        "whisper" or "mms"
    """
    prefs = _load_backend_prefs()
    lang_info = prefs.get(language)

    # When LID confidence is low, default to MMS regardless of mode.
    # MMS adapters produce less catastrophic output on wrong-language input.
    if lid_confidence is not None and lid_confidence < 0.5:
        log.debug(f"Low LID conf ({lid_confidence:.2f}) for '{language}' → MMS")
        return "mms"

    # Mode C fallback is handled entirely in fallback_decode.py
    if mode == "C":
        return _default_backend

    # Mode A (single decode) and Mode B (multi-hypothesis):
    # Use per-language preference table
    if lang_info:
        preferred = lang_info.get("preferred", _default_backend)
        log.debug(f"Backend for '{language}' mode {mode}: {preferred}")
        return preferred

    # Unknown language → MMS (broader 1162-language coverage)
    log.debug(f"'{language}' not in quality table → {_default_backend}")
    return _default_backend
