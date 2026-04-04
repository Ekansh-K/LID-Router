"""
Rule-based routing policy — the baseline routing strategy.

Thresholds are tuned on the FLEURS dev set (Phase 4).
The rules are interpretable and serve as the comparison point
for the learned MLP policy.
"""
from typing import Dict

from src.utils import UncertaintySignals, get_logger
from src.routing.confusion_map import ConfusionMap

log = get_logger("routing.policy_rules")


class RoutingMode:
    SINGLE = "A"
    MULTI_HYPOTHESIS = "B"
    FALLBACK = "C"


class RoutingDecision:
    __slots__ = ("mode", "candidate_languages", "confidence", "reason")

    def __init__(self, mode: str, candidate_languages: list,
                 confidence: float, reason: str):
        self.mode = mode
        self.candidate_languages = candidate_languages
        self.confidence = confidence
        self.reason = reason

    def __repr__(self):
        return (f"RoutingDecision(mode={self.mode}, "
                f"langs={self.candidate_languages}, "
                f"conf={self.confidence:.3f}, "
                f"reason='{self.reason}')")


class RuleBasedPolicy:
    """Threshold-based routing.
    
    Mode A (Single):  high confidence + large gap + not in confusion cluster
    Mode B (Multi):   medium confidence OR low entropy
    Mode C (Fallback): everything else
    """

    def __init__(self,
                 theta_high: float = 0.85,
                 delta_high: float = 0.30,
                 theta_med: float = 0.40,
                 entropy_med: float = 2.0,
                 top_k: int = 3,
                 confusion_map: ConfusionMap = None):
        self.theta_high = theta_high
        self.delta_high = delta_high
        self.theta_med = theta_med
        self.entropy_med = entropy_med
        self.top_k = top_k
        self.confusion_map = confusion_map or ConfusionMap()

    def decide(self, fused_probs: Dict[str, float],
               uncertainty: UncertaintySignals) -> RoutingDecision:

        if not fused_probs:
            return RoutingDecision(
                mode=RoutingMode.FALLBACK,
                candidate_languages=[],
                confidence=0.0,
                reason="No fused probabilities available"
            )

        sorted_langs = sorted(fused_probs, key=fused_probs.get, reverse=True)
        top1 = sorted_langs[0]
        in_confusion = self.confusion_map.is_confused(top1)

        # Mode A: confident and clear winner, not in a tricky cluster
        if (uncertainty.top1_prob > self.theta_high
                and uncertainty.gap > self.delta_high
                and not in_confusion):
            return RoutingDecision(
                mode=RoutingMode.SINGLE,
                candidate_languages=[top1],
                confidence=uncertainty.top1_prob,
                reason="High confidence, large gap, no confusion cluster"
            )

        # Mode B: moderate confidence or low entropy
        if uncertainty.top1_prob > self.theta_med or uncertainty.entropy < self.entropy_med:
            candidates = sorted_langs[:self.top_k]
            # Inject confusion partners even if they're not in top-k
            if in_confusion:
                partners = self.confusion_map.get_partners(top1)
                candidate_set = set(candidates)
                for p in partners:
                    if p not in candidate_set and p in fused_probs:
                        candidates.append(p)
                # Cap at top_k + 2 to keep it bounded
                candidates = candidates[:self.top_k + 2]
            return RoutingDecision(
                mode=RoutingMode.MULTI_HYPOTHESIS,
                candidate_languages=candidates,
                confidence=uncertainty.top1_prob,
                reason=("Medium confidence" +
                        (", confusion partners added" if in_confusion else ""))
            )

        # Mode C: low confidence fallback
        return RoutingDecision(
            mode=RoutingMode.FALLBACK,
            candidate_languages=sorted_langs[:5],
            confidence=uncertainty.top1_prob,
            reason="Low confidence, fallback triggered"
        )
