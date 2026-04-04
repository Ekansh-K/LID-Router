"""
Backend selector: decides whether to use Whisper or MMS for a given language.

Logic:
- High-resource language in Whisper's 99 → Whisper (better quality)
- Language NOT in Whisper's set → MMS (broader coverage)
- Multi-hypothesis → prefer MMS (adapter switching is faster than Whisper re-decode)
"""
from src.utils import get_logger, get_language_map

log = get_logger("asr.selector")


def select_backend(language: str, mode: str = "A") -> str:
    """Choose 'whisper' or 'mms' for a given language and routing mode.
    
    Args:
        language: canonical ISO 639-3 code
        mode: routing mode ("A", "B", "C")
    
    Returns:
        "whisper" or "mms"
    """
    lang_map = get_language_map()

    in_whisper = lang_map.whisper_supported(language)

    # Mode B (multi-hypothesis): prefer MMS for speed
    if mode == "B":
        if in_whisper:
            # Still use MMS for multi-hypo since adapter switching is
            # much faster than running Whisper multiple times
            return "mms"
        return "mms"

    # Mode A (single decode): use Whisper if available for higher quality
    if mode == "A" and in_whisper:
        return "whisper"

    # Mode C fallback handled separately in fallback_decode.py
    # Default: MMS
    return "mms"
