"""
Tests for decoding modes — uses mocked ASR backends.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.utils import TranscriptResult


def _mock_backends():
    """Create mock WhisperBackend and MMSBackend for testing."""
    whisper = MagicMock()
    whisper.transcribe.return_value = TranscriptResult(
        text="hello world", language="eng", confidence=0.9, backend="whisper"
    )
    whisper.transcribe_auto.return_value = TranscriptResult(
        text="hello world auto", language="eng", confidence=0.85, backend="whisper_auto"
    )

    mms = MagicMock()
    mms.transcribe.return_value = TranscriptResult(
        text="hello world mms", language="eng", confidence=0.8, backend="mms"
    )

    return whisper, mms


class TestSingleDecode:

    def test_decode_single(self):
        from src.decoding.single_decode import decode_single
        whisper, mms = _mock_backends()
        audio = np.zeros(16000, dtype=np.float32)

        result = decode_single(audio, "eng", whisper, mms)
        assert result.text != ""
        assert result.language == "eng"


class TestMultiHypothesis:

    def test_reranking(self):
        from src.decoding.multi_hypothesis import decode_multi_hypothesis
        whisper, mms = _mock_backends()

        # Make MMS return different quality for different languages
        def mms_side_effect(audio, language, sr=16000):
            if language == "hin":
                return TranscriptResult(text="hindi text here", language="hin",
                                       confidence=0.85, backend="mms")
            elif language == "urd":
                return TranscriptResult(text="u", language="urd",
                                       confidence=0.3, backend="mms")
            return TranscriptResult(text="default", language=language,
                                   confidence=0.5, backend="mms")
        mms.transcribe.side_effect = mms_side_effect

        audio = np.zeros(16000, dtype=np.float32)
        lid_probs = {"hin": 0.55, "urd": 0.35, "eng": 0.10}

        best, all_results = decode_multi_hypothesis(
            audio, ["hin", "urd"], lid_probs, whisper, mms
        )

        assert best.language == "hin"  # should win due to better confidence + plausibility
        assert len(all_results) == 2


class TestFallbackDecode:

    def test_tiered_fallback(self):
        from src.decoding.fallback_decode import decode_fallback
        whisper, mms = _mock_backends()

        audio = np.zeros(16000, dtype=np.float32)
        lid_probs = {"yor": 0.2, "hau": 0.18, "swh": 0.15}

        best, all_results = decode_fallback(
            audio, ["yor", "hau", "swh"], lid_probs, whisper, mms
        )

        assert best.text != ""
        assert len(all_results) > 0  # should have results from multiple tiers
