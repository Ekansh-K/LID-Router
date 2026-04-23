"""
Confusion map: registry and lookup for known language confusion pairs.

Loaded from config/confusion_clusters.yaml and used by the routing agent
to override high-confidence Mode A decisions when the language belongs
to a cluster with known mutual intelligibility.

Phase 1+2 additions:
  - force_mode_b_threshold: per-cluster minimum confidence to use Mode A.
    Below this threshold Mode B is always used regardless of MLP/rules output.
  - temperature: per-cluster softmax temperature applied before gap/entropy calc.
  - rerank_by_script: whether multi_hypothesis should score by Unicode script.
  - script_expectations: global mapping lang -> expected Unicode script name.
"""
from typing import Dict, List, Optional, Set
from pathlib import Path

from src.utils import get_logger, load_yaml

log = get_logger("routing.confusion_map")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfusionMap:
    """Fast lookup for confusion clusters."""

    def __init__(self, config_path: Optional[str | Path] = None):
        if config_path is None:
            config_path = _PROJECT_ROOT / "config" / "confusion_clusters.yaml"
        self._clusters: List[dict] = []
        self._lang_to_partners: Dict[str, Set[str]] = {}
        self._lang_to_cluster_id: Dict[str, str] = {}
        self._lang_to_cluster: Dict[str, dict] = {}  # full cluster dict per lang
        self._script_expectations: Dict[str, str] = {}  # lang -> Unicode script name
        self._load(config_path)

    def _load(self, path: str | Path):
        data = load_yaml(path)
        self._clusters = data.get("clusters", [])
        self._script_expectations = data.get("script_expectations", {})

        for cluster in self._clusters:
            cid = cluster["id"]
            langs = cluster.get("languages", [])
            for lang in langs:
                partners = set(l for l in langs if l != lang)
                if lang in self._lang_to_partners:
                    self._lang_to_partners[lang] |= partners
                else:
                    self._lang_to_partners[lang] = partners
                self._lang_to_cluster_id[lang] = cid
                self._lang_to_cluster[lang] = cluster

        log.info(f"Loaded {len(self._clusters)} confusion clusters "
                 f"covering {len(self._lang_to_partners)} languages")

    def is_confused(self, lang: str) -> bool:
        """Does this language belong to any confusion cluster?"""
        return lang in self._lang_to_partners

    def get_partners(self, lang: str) -> List[str]:
        """Return list of languages that are known to confuse with `lang`."""
        return list(self._lang_to_partners.get(lang, set()))

    def get_cluster_id(self, lang: str) -> Optional[str]:
        return self._lang_to_cluster_id.get(lang)

    def get_cluster_languages(self, cluster_id: str) -> List[str]:
        for cluster in self._clusters:
            if cluster["id"] == cluster_id:
                return cluster.get("languages", [])
        return []

    def all_cluster_ids(self) -> List[str]:
        return [c["id"] for c in self._clusters]

    # ── Phase 1+2: New accessors ─────────────────────────────────────────────

    def force_mode_b_threshold(self, lang: str) -> float:
        """Minimum top1_prob required for Mode A to be used for this lang.

        If the fused top1_prob is below this value, Mode B is forced
        regardless of what the MLP or rules policy says.
        Returns 0.0 (never force B) for languages not in any cluster.
        """
        cluster = self._lang_to_cluster.get(lang)
        if cluster is None:
            return 0.0
        return float(cluster.get("force_mode_b_threshold", 0.0))

    def temperature(self, lang: str) -> float:
        """Softmax temperature for this language's cluster.

        tau > 1 softens the distribution (safer for confused clusters).
        tau < 1 sharpens it. Returns 1.0 (no scaling) for unclustered langs.
        """
        cluster = self._lang_to_cluster.get(lang)
        if cluster is None:
            return 1.0
        return float(cluster.get("temperature", 1.0))

    def rerank_by_script(self, lang: str) -> bool:
        """Should multi_hypothesis use Unicode script scoring for this lang?

        True for clusters where script mismatch causes catastrophic CER
        (urd vs hin -> Perso-Arabic vs Devanagari, srp vs hrv -> Cyrillic vs Latin).
        """
        cluster = self._lang_to_cluster.get(lang)
        if cluster is None:
            return False
        return bool(cluster.get("rerank_by_script", False))

    def expected_script(self, lang: str) -> Optional[str]:
        """Expected Unicode script name for this language.

        Used by the script-aware reranker.
        E.g. 'urd' -> 'Arab', 'srp' -> 'Cyrl', 'hin' -> 'Deva'.
        Returns None if not configured.
        """
        return self._script_expectations.get(lang)
