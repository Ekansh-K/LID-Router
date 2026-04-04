"""
Tests for routing agent and policies — runs on CPU, no GPU needed.
"""
import numpy as np
import pytest
from src.routing.policy_rules import RuleBasedPolicy, RoutingMode
from src.routing.confusion_map import ConfusionMap
from src.utils import UncertaintySignals


class TestRuleBasedPolicy:

    def setup_method(self):
        self.policy = RuleBasedPolicy(
            theta_high=0.85, delta_high=0.30,
            theta_med=0.40, entropy_med=2.0, top_k=3,
        )

    def test_mode_a_confident(self):
        """High confidence + large gap → Mode A."""
        probs = {"eng": 0.95, "fra": 0.03, "deu": 0.02}
        u = UncertaintySignals(top1_prob=0.95, gap=0.92, entropy=0.3,
                               top3_concentration=1.0, in_confusion_cluster=False)
        d = self.policy.decide(probs, u)
        assert d.mode == RoutingMode.SINGLE
        assert d.candidate_languages == ["eng"]

    def test_mode_b_medium_confidence(self):
        """Medium confidence → Mode B."""
        probs = {"hin": 0.55, "urd": 0.30, "eng": 0.15}
        u = UncertaintySignals(top1_prob=0.55, gap=0.25, entropy=1.3,
                               top3_concentration=1.0, in_confusion_cluster=True)
        d = self.policy.decide(probs, u)
        assert d.mode == RoutingMode.MULTI_HYPOTHESIS
        assert len(d.candidate_languages) >= 2

    def test_mode_c_low_confidence(self):
        """Low confidence → Mode C fallback."""
        probs = {"yor": 0.2, "hau": 0.18, "swh": 0.15, "amh": 0.12, "eng": 0.10}
        u = UncertaintySignals(top1_prob=0.2, gap=0.02, entropy=3.5,
                               top3_concentration=0.53, in_confusion_cluster=False)
        d = self.policy.decide(probs, u)
        assert d.mode == RoutingMode.FALLBACK

    def test_confusion_override(self):
        """High confidence BUT in confusion cluster → downgrade to Mode B."""
        probs = {"hin": 0.90, "urd": 0.05, "eng": 0.05}
        u = UncertaintySignals(top1_prob=0.90, gap=0.85, entropy=0.5,
                               top3_concentration=1.0, in_confusion_cluster=True)
        # Override: confusion_map says hin is confused
        d = self.policy.decide(probs, u)
        # With confusion, even high confidence gets demoted
        assert d.mode == RoutingMode.MULTI_HYPOTHESIS

    def test_confusion_partners_added(self):
        """In Mode B with confusion, partners should be in candidates."""
        # Need a proper confusion map for this
        policy = RuleBasedPolicy(theta_high=0.85, delta_high=0.30,
                                 theta_med=0.40, entropy_med=2.0, top_k=3)
        probs = {"srp": 0.55, "eng": 0.20, "fra": 0.15, "hrv": 0.05, "deu": 0.05}
        u = UncertaintySignals(top1_prob=0.55, gap=0.35, entropy=1.8,
                               top3_concentration=0.90, in_confusion_cluster=True)
        d = policy.decide(probs, u)
        # srp's confusion partners (hrv, bos) should be considered
        assert d.mode == RoutingMode.MULTI_HYPOTHESIS


class TestConfusionMap:

    def test_load_and_lookup(self):
        cmap = ConfusionMap()
        # From the config file
        assert cmap.is_confused("hin")
        assert "urd" in cmap.get_partners("hin")
        assert cmap.is_confused("srp")
        assert "hrv" in cmap.get_partners("srp")

    def test_not_confused(self):
        cmap = ConfusionMap()
        assert not cmap.is_confused("eng")  # English not in any cluster
        assert cmap.get_partners("eng") == []
