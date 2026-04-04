"""
Routing Agent — the "agentic" core of the pipeline.

Wraps either the rule-based or learned policy and provides the
unified interface the pipeline calls.
"""
from typing import Dict, Optional

from src.utils import UncertaintySignals, get_logger, load_config
from src.routing.policy_rules import RuleBasedPolicy, RoutingDecision, RoutingMode
from src.routing.policy_learned import LearnedRoutingPolicy
from src.routing.confusion_map import ConfusionMap

log = get_logger("routing.agent")


class RoutingAgent:
    """Decides how to decode based on LID confidence signals.
    
    Supports two policies:
    - 'rules': threshold-based (interpretable baseline)
    - 'learned': MLP-based (trained on dev set oracle labels)
    """

    def __init__(self, policy: str = "rules", config: Optional[dict] = None):
        if config is None:
            config = load_config().get("routing", {})

        self.confusion_map = ConfusionMap()

        if policy == "rules":
            self._policy = RuleBasedPolicy(
                theta_high=config.get("theta_high", 0.85),
                delta_high=config.get("delta_high", 0.30),
                theta_med=config.get("theta_med", 0.40),
                entropy_med=config.get("entropy_med", 2.0),
                top_k=config.get("top_k", 3),
                confusion_map=self.confusion_map,
            )
        elif policy == "learned":
            self._policy = LearnedRoutingPolicy(
                input_dim=config.get("input_dim", 6),
                hidden_dim=config.get("hidden_dim", 32),
            )
        else:
            raise ValueError(f"Unknown policy: {policy}. Use 'rules' or 'learned'.")

        self.policy_name = policy
        log.info(f"RoutingAgent initialized with '{policy}' policy")

    def load_learned_policy(self, checkpoint_path: str):
        """Load a trained MLP policy from checkpoint."""
        if not isinstance(self._policy, LearnedRoutingPolicy):
            raise RuntimeError("Agent not configured with 'learned' policy")
        self._policy.load(checkpoint_path)

    def decide(self, fused_probs: Dict[str, float],
               uncertainty: UncertaintySignals) -> RoutingDecision:
        """Make a routing decision.
        
        Returns a RoutingDecision with:
        - mode: A (single), B (multi-hypothesis), C (fallback)
        - candidate_languages: ordered list of language codes to try
        - confidence: how confident the agent is in this decision
        - reason: human-readable explanation
        """
        decision = self._policy.decide(fused_probs, uncertainty)
        log.debug(f"Routing decision: {decision}")
        return decision
