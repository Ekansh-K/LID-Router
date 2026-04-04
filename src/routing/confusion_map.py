"""
Confusion map: registry and lookup for known language confusion pairs.

Loaded from config/confusion_clusters.yaml and used by the routing agent
to override high-confidence Mode A decisions when the language belongs
to a cluster with known mutual intelligibility.
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
        self._load(config_path)

    def _load(self, path: str | Path):
        data = load_yaml(path)
        self._clusters = data.get("clusters", [])
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
