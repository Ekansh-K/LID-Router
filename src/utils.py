"""
Shared utilities: config loading, language code normalization, timing, logging.
"""
import os
import time
import logging
import functools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s %(name)s %(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

log = get_logger("utils")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config(config_name: str = "model_config.yaml") -> dict:
    return load_yaml(_PROJECT_ROOT / "config" / config_name)

# ---------------------------------------------------------------------------
# Language map (singleton)
# ---------------------------------------------------------------------------
class LanguageMap:
    """Bidirectional mapping between heterogeneous language-code systems."""

    def __init__(self, yaml_path: str | Path | None = None):
        if yaml_path is None:
            yaml_path = _PROJECT_ROOT / "config" / "language_map.yaml"
        self._data: Dict[str, dict] = load_yaml(yaml_path)
        self._build_reverse_indices()

    def _build_reverse_indices(self):
        self._whisper_to_canon: Dict[str, str] = {}
        self._mms_lid_to_canon: Dict[str, str] = {}
        self._mms_asr_to_canon: Dict[str, str] = {}
        self._fleurs_to_canon: Dict[str, str] = {}
        for canon, info in self._data.items():
            if info.get("whisper"):
                self._whisper_to_canon[info["whisper"]] = canon
            if info.get("mms_lid"):
                self._mms_lid_to_canon[info["mms_lid"]] = canon
            if info.get("mms_asr"):
                self._mms_asr_to_canon[info["mms_asr"]] = canon
            if info.get("fleurs"):
                self._fleurs_to_canon[info["fleurs"]] = canon

    # ── Forward lookups (canonical → system-specific) ──
    def to_whisper(self, canon: str) -> Optional[str]:
        entry = self._data.get(canon)
        return entry["whisper"] if entry else None

    def to_mms_lid(self, canon: str) -> Optional[str]:
        entry = self._data.get(canon)
        return entry["mms_lid"] if entry else None

    def to_mms_asr(self, canon: str) -> Optional[str]:
        entry = self._data.get(canon)
        return entry["mms_asr"] if entry else None

    def to_fleurs(self, canon: str) -> Optional[str]:
        entry = self._data.get(canon)
        return entry["fleurs"] if entry else None

    # ── Reverse lookups (system-specific → canonical) ──
    def from_whisper(self, code: str) -> Optional[str]:
        return self._whisper_to_canon.get(code)

    def from_mms_lid(self, code: str) -> Optional[str]:
        return self._mms_lid_to_canon.get(code)

    def from_mms_asr(self, code: str) -> Optional[str]:
        return self._mms_asr_to_canon.get(code)

    def from_fleurs(self, code: str) -> Optional[str]:
        return self._fleurs_to_canon.get(code)

    # ── Queries ──
    def get_info(self, canon: str) -> Optional[dict]:
        return self._data.get(canon)

    def all_canonical(self) -> List[str]:
        return list(self._data.keys())

    def whisper_supported(self, canon: str) -> bool:
        return canon in self._data and bool(self._data[canon].get("whisper"))

    def get_family(self, canon: str) -> Optional[str]:
        entry = self._data.get(canon)
        return entry.get("family") if entry else None

    def langs_in_family(self, family: str) -> List[str]:
        return [c for c, info in self._data.items() if info.get("family") == family]

# Module-level singleton — import from here everywhere
_lang_map_instance: Optional[LanguageMap] = None

def get_language_map() -> LanguageMap:
    global _lang_map_instance
    if _lang_map_instance is None:
        _lang_map_instance = LanguageMap()
    return _lang_map_instance

# ---------------------------------------------------------------------------
# Data classes shared across modules
# ---------------------------------------------------------------------------
@dataclass
class UncertaintySignals:
    top1_prob: float = 0.0
    gap: float = 0.0          # top1 - top2
    entropy: float = 0.0
    top3_concentration: float = 0.0
    in_confusion_cluster: bool = False
    single_lid_only: bool = False  # True when language not covered by both LIDs

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.top1_prob,
            self.gap,
            self.entropy,
            self.top3_concentration,
            float(self.in_confusion_cluster),
            float(self.single_lid_only),
        ], dtype=np.float32)

@dataclass
class TranscriptResult:
    text: str
    language: str              # canonical ISO 639-3
    confidence: float          # decoder-level confidence (avg log-prob → [0,1])
    backend: str = ""          # "whisper" or "mms"
    log_probs: Optional[float] = None

@dataclass
class PipelineOutput:
    transcript: str
    detected_language: str
    confidence: float
    routing_mode: str          # "A", "B", "C"
    candidates_considered: int
    backend_used: str
    lid_distribution: Dict[str, float] = field(default_factory=dict)
    all_transcripts: List[TranscriptResult] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        log.debug(f"{func.__qualname__} took {dt:.3f}s")
        return result
    return wrapper

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def get_device(preferred: str = "cuda") -> str:
    import torch
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def gpu_memory_mb() -> Optional[float]:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_mem / 1e6
    return None
