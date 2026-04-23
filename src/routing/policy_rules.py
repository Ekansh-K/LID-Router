"""
Rule-based routing policy — the baseline routing strategy.

Phase 1+2 additions:
  - Temperature scaling: applies per-cluster tau to fused_probs before
    computing gap/entropy, so confused clusters appear less certain.
  - force_mode_b_threshold: compares RAW (pre-temperature) top1_prob.
    For languages with proven low LID accuracy (urd=67%, srp=83%) we
    force Mode B regardless of the distribution shape.
  - Flat-distribution guard: if post-temperature gap < flat_gap_override,
    force Mode B for any in-cluster language.
"""
from typing import Dict

from src.utils import UncertaintySignals, get_logger, get_language_map
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


def _apply_temperature(probs: Dict[str, float], tau: float) -> Dict[str, float]:
    """Apply temperature scaling to a probability distribution.

    Uses p^(1/tau) re-normalisation (entropy-scaling, not logit-scaling).
    tau > 1  → soften (probabilities converge toward uniform).
    tau == 1 → no change.
    tau < 1  → sharpen (probabilities move toward one-hot).

    NOTE: This is applied ONLY for gap/entropy computation inside the routing
    policy. The force_mode_b_threshold check intentionally uses the RAW prob
    (before temperature) to correctly gate on the original LID confidence.
    """
    if abs(tau - 1.0) < 1e-6:
        return probs
    # p^(1/tau) then re-normalise; skip zero-prob entries to avoid 0^inf
    scaled = {k: v ** (1.0 / tau) for k, v in probs.items() if v > 0}
    total = sum(scaled.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in scaled.items()}


class RuleBasedPolicy:
    """Threshold-based routing with Phase 1+2 extensions.

    Mode A (Single):   high confidence + large tempered-gap + not in confusion
                       cluster AND raw top1_prob >= force_mode_b_threshold.
    Mode B (Multi):    medium confidence OR low entropy, OR forced by cluster
                       threshold / flat-gap guard.
    Mode C (Fallback): everything else (very low confidence).
    """

    def __init__(self,
                 theta_high: float = 0.85,
                 delta_high: float = 0.30,
                 theta_med: float = 0.40,
                 entropy_med: float = 2.0,
                 flat_gap_override: float = 0.10,
                 top_k: int = 3,
                 confusion_map: ConfusionMap = None):
        self.theta_high = theta_high
        self.delta_high = delta_high
        self.theta_med = theta_med
        self.entropy_med = entropy_med
        self.flat_gap_override = flat_gap_override  # Phase 2: flat-dist guard
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

        lang_map = get_language_map()
        sorted_langs = [
            l for l in sorted(fused_probs, key=fused_probs.get, reverse=True)
            if lang_map.asr_capable(l)
        ]
        if not sorted_langs:
            return RoutingDecision(
                mode=RoutingMode.FALLBACK,
                candidate_languages=[],
                confidence=0.0,
                reason="No ASR-capable languages in LID output"
            )

        top1 = sorted_langs[0]
        in_confusion = self.confusion_map.is_confused(top1)

        # RAW probability for the top-1 ASR-capable language.
        # Used for force_mode_b_threshold — we want the original LID confidence,
        # NOT the temperature-scaled value (temperature scaling only affects gap/entropy).
        raw_top1_prob = fused_probs[top1]

        # ── Phase 1: force_mode_b_threshold — checked against RAW prob ─────────
        # This must come BEFORE temperature scaling so we gate on true LID confidence.
        force_b_threshold = self.confusion_map.force_mode_b_threshold(top1)
        if force_b_threshold > 0 and raw_top1_prob < force_b_threshold:
            candidates = sorted_langs[:self.top_k]
            _inject_partners(candidates, top1, self.confusion_map, fused_probs, lang_map, self.top_k)
            log.debug(f"Forced Mode B for '{top1}': raw_prob={raw_top1_prob:.3f} < threshold={force_b_threshold:.3f}")
            return RoutingDecision(
                mode=RoutingMode.MULTI_HYPOTHESIS,
                candidate_languages=candidates,
                confidence=raw_top1_prob,
                reason=(f"Forced Mode B: raw LID {raw_top1_prob:.3f} < "
                        f"cluster threshold {force_b_threshold:.3f}")
            )

        # ── Phase 2: Apply per-cluster temperature for gap/entropy computation ──
        tau = self.confusion_map.temperature(top1)
        tempered_probs = _apply_temperature(fused_probs, tau)
        t_sorted_langs = [
            l for l in sorted(tempered_probs, key=tempered_probs.get, reverse=True)
            if lang_map.asr_capable(l)
        ]
        t_top1 = t_sorted_langs[0] if t_sorted_langs else top1
        top1_prob = tempered_probs.get(t_top1, raw_top1_prob)
        top2_prob = (tempered_probs.get(t_sorted_langs[1], 0.0)
                     if len(t_sorted_langs) > 1 else 0.0)
        tempered_gap = top1_prob - top2_prob

        # ── Phase 2: Flat-distribution guard (uses tempered gap) ──────────────
        if tempered_gap < self.flat_gap_override and in_confusion:
            candidates = t_sorted_langs[:self.top_k]
            _inject_partners(candidates, top1, self.confusion_map, fused_probs, lang_map, self.top_k)
            log.debug(f"Flat-dist guard triggered for '{top1}': gap={tempered_gap:.3f}")
            return RoutingDecision(
                mode=RoutingMode.MULTI_HYPOTHESIS,
                candidate_languages=candidates,
                confidence=raw_top1_prob,
                reason=(f"Flat distribution guard: tempered gap={tempered_gap:.3f} "
                        f"< {self.flat_gap_override} in confusion cluster (tau={tau:.1f})")
            )

        # ── Standard Mode A ───────────────────────────────────────────────────
        if (raw_top1_prob > self.theta_high
                and tempered_gap > self.delta_high
                and not in_confusion):
            return RoutingDecision(
                mode=RoutingMode.SINGLE,
                candidate_languages=[t_top1],
                confidence=raw_top1_prob,
                reason="High confidence, large gap, no confusion cluster"
            )

        # ── Standard Mode B ───────────────────────────────────────────────────
        if raw_top1_prob > self.theta_med or uncertainty.entropy < self.entropy_med:
            candidates = t_sorted_langs[:self.top_k]
            if in_confusion:
                _inject_partners(candidates, top1, self.confusion_map, fused_probs, lang_map, self.top_k)
            return RoutingDecision(
                mode=RoutingMode.MULTI_HYPOTHESIS,
                candidate_languages=candidates,
                confidence=raw_top1_prob,
                reason=("Medium confidence" +
                        (", confusion partners added" if in_confusion else ""))
            )

        # ── Mode C: low confidence fallback ───────────────────────────────────
        return RoutingDecision(
            mode=RoutingMode.FALLBACK,
            candidate_languages=t_sorted_langs[:5],
            confidence=raw_top1_prob,
            reason="Low confidence, fallback triggered"
        )


def _inject_partners(candidates: list, top1: str, confusion_map: ConfusionMap,
                     fused_probs: dict, lang_map, top_k: int):
    """In-place: add confusion partners to candidate list if not already present.

    Uses fused_probs (not tempered) so all partners visible to LID are included.
    Caps the list at top_k + 2 (at most 2 extra partners), not at top_k + len(all_partners)
    which could be unbounded for large clusters.
    """
    partners = confusion_map.get_partners(top1)
    candidate_set = set(candidates)
    added = 0
    for p in partners:
        if added >= 2:  # cap: at most 2 extra partners beyond top_k
            break
        if (p not in candidate_set
                and lang_map.asr_capable(p)):
            candidates.append(p)
            candidate_set.add(p)
            added += 1
    # Hard cap: never exceed top_k + 2
    del candidates[top_k + 2:]
