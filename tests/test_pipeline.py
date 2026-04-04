"""
Tests for the end-to-end pipeline with mocked LID and ASR models.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.utils import TranscriptResult, PipelineOutput


class TestPipelineIntegration:

    def test_pipeline_structure(self):
        """Verify pipeline can be instantiated without GPU."""
        from src.pipeline import Pipeline
        # Don't load models — just test construction
        pipe = Pipeline(config={
            "lid": {
                "acoustic": {"model_id": "test", "device": "cpu", "precision": "fp32"},
                "decoder": {"model_size": "tiny", "device": "cpu"},
            },
            "fusion": {"default_alpha": 0.6, "single_signal_penalty": 0.85},
            "routing": {},
            "asr": {
                "whisper": {"model_size": "tiny", "device": "cpu"},
                "mms": {"model_id": "test", "device": "cpu"},
            },
        })
        assert pipe.routing_agent is not None
        assert pipe.fusion is not None

    def test_output_structure(self):
        """Verify PipelineOutput has all required fields."""
        output = PipelineOutput(
            transcript="test",
            detected_language="eng",
            confidence=0.9,
            routing_mode="A",
            candidates_considered=1,
            backend_used="whisper",
        )
        assert output.transcript == "test"
        assert output.routing_mode == "A"
        assert output.lid_distribution == {}
        assert output.all_transcripts == []
