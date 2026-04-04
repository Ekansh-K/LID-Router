"""
Tests for LID modules — runs on CPU with mocked models.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.lid.fusion import LIDFusion
from src.utils import UncertaintySignals


class TestLIDFusion:
    """Test the fusion module with synthetic probability distributions."""

    def setup_method(self):
        # Create fusion with known confusion clusters
        clusters = {
            "clusters": [
                {"id": "test_cluster", "languages": ["hin", "urd"]},
            ]
        }
        self.fusion = LIDFusion(alpha=0.6, confusion_clusters=clusters)

    def test_fusion_both_signals(self):
        """When both models agree, fused score should be high."""
        acoustic = {"hin": 0.9, "urd": 0.05, "eng": 0.05}
        decoder = {"hin": 0.85, "urd": 0.1, "eng": 0.05}

        fused = self.fusion.fuse(acoustic, decoder)
        assert max(fused, key=fused.get) == "hin"
        assert fused["hin"] > 0.8

    def test_fusion_disagreement(self):
        """When models disagree, uncertainty should increase."""
        acoustic = {"hin": 0.6, "urd": 0.3, "eng": 0.1}
        decoder = {"hin": 0.2, "urd": 0.7, "eng": 0.1}

        fused, uncertainty = self.fusion.fuse_and_analyze(acoustic, decoder)
        # Neither model is very confident
        assert uncertainty.gap < 0.5
        assert uncertainty.entropy > 0.5

    def test_fusion_single_signal(self):
        """Language only in acoustic model gets penalty."""
        acoustic = {"yor": 0.9, "hau": 0.1}
        decoder = {"eng": 0.5, "fra": 0.5}  # yor not in whisper

        fused = self.fusion.fuse(acoustic, decoder)
        # yor should still be top but with penalty
        assert "yor" in fused

    def test_uncertainty_confident(self):
        probs = {"eng": 0.95, "fra": 0.03, "deu": 0.02}
        u = self.fusion.compute_uncertainty(probs)
        assert u.top1_prob == 0.95
        assert u.gap == pytest.approx(0.92, abs=0.01)
        assert u.entropy < 0.5

    def test_uncertainty_confused(self):
        probs = {"hin": 0.45, "urd": 0.40, "eng": 0.15}
        u = self.fusion.compute_uncertainty(probs)
        assert u.in_confusion_cluster is True  # hin is in cluster
        assert u.gap < 0.1

    def test_confusion_partners(self):
        partners = self.fusion.get_confusion_partners("hin")
        assert "urd" in partners
        assert self.fusion.get_confusion_partners("eng") == []

    def test_empty_probs(self):
        u = self.fusion.compute_uncertainty({})
        assert u.top1_prob == 0.0

    def test_renormalization(self):
        """Fused distribution should sum to ~1."""
        acoustic = {"eng": 0.5, "fra": 0.3, "deu": 0.2}
        decoder = {"eng": 0.6, "fra": 0.25, "deu": 0.15}
        fused = self.fusion.fuse(acoustic, decoder)
        assert abs(sum(fused.values()) - 1.0) < 0.01
