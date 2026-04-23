"""
F0 (Fundamental Frequency / Pitch) Feature Extractor — Phase 3.

Extracts a compact 4-dimensional pitch feature vector from audio.
These features help distinguish:
  - Tonal languages (cmn, lao, yue, yor): high F0 variance, distinct contours
  - Pitch-accent languages (jpn): low F0 variance, bounded range
  - Stress-timed (eng, deu): broad F0 range but infrequent tonal changes
  - Syllable-timed (spa, fra): moderate F0, regular

Feature vector (4 dimensions):
  [0] mean_f0        : mean fundamental frequency of voiced frames (Hz)
  [1] f0_std         : std dev of F0 (measures tonal variability)
  [2] f0_range       : max - min F0 of voiced frames (tonal range)
  [3] voiced_ratio   : fraction of frames that are voiced (0-1)

Usage:
  from src.lid.f0_features import extract_f0_features
  feats = extract_f0_features(audio, sr=16000)   # returns np.ndarray shape (4,)

Dependencies (install on Kaggle before Step 10):
  pip install pyworld
  -- OR --
  pip install torchcrepe  (deep learning alternative, more accurate but heavier)

Falls back to zero vector if pyworld is unavailable (safe — pipeline still works).
"""
import numpy as np
from typing import Optional

# Silence loudly if pyworld not installed — caller handles the fallback
try:
    import pyworld as pw
    _HAS_PYWORLD = True
except ImportError:
    _HAS_PYWORLD = False


# ── Constants ─────────────────────────────────────────────────────────────────
F0_FLOOR   = 60.0    # Hz — below this is sub-bass (not speech fundamental)
F0_CEIL    = 550.0   # Hz — above this is falsetto / noise
_ZERO_VEC  = np.zeros(4, dtype=np.float32)


def extract_f0_features(audio: np.ndarray,
                        sr: int = 16000,
                        frame_period_ms: float = 5.0) -> np.ndarray:
    """Extract 4-dim F0 feature vector using WORLD vocoder.

    Args:
        audio  : 1D float32 or float64 numpy array, mono, at `sr` Hz.
        sr     : sample rate (must match audio). Default 16000.
        frame_period_ms: analysis frame period in milliseconds. 5 ms is standard.

    Returns:
        np.ndarray of shape (4,) with [mean_f0, f0_std, f0_range, voiced_ratio].
        All zeros if pyworld is unavailable or audio is too short/silent.
    """
    if not _HAS_PYWORLD:
        return _ZERO_VEC.copy()

    if audio is None or len(audio) < sr * 0.1:
        # Less than 100 ms of audio — not enough for reliable F0
        return _ZERO_VEC.copy()

    try:
        # WORLD requires float64
        x = audio.astype(np.float64)

        # Normalise amplitude so WORLD doesn't fail on quiet audio
        peak = np.abs(x).max()
        if peak < 1e-6:
            return _ZERO_VEC.copy()
        x = x / peak * 0.99

        # Extract F0 using Harvest (more robust than DIO for continuous speech)
        f0, t = pw.harvest(x, sr,
                           f0_floor=F0_FLOOR,
                           f0_ceil=F0_CEIL,
                           frame_period=frame_period_ms)

        voiced = f0[f0 > 0]

        if len(voiced) < 5:
            # Less than 5 voiced frames — too short for meaningful F0 stats
            voiced_ratio = float(len(voiced)) / max(len(f0), 1)
            return np.array([0.0, 0.0, 0.0, voiced_ratio], dtype=np.float32)

        mean_f0      = float(voiced.mean())
        f0_std       = float(voiced.std())
        f0_range     = float(voiced.max() - voiced.min())
        voiced_ratio = float(len(voiced)) / len(f0)

        return np.array([mean_f0, f0_std, f0_range, voiced_ratio], dtype=np.float32)

    except Exception:
        return _ZERO_VEC.copy()


def is_available() -> bool:
    """Return True if pyworld is installed and F0 extraction is active."""
    return _HAS_PYWORLD


def dummy_features() -> np.ndarray:
    """Return a zero vector for use when pyworld is not available.

    The routing MLP is designed to treat [0,0,0,0] as 'no F0 signal'
    and fall back to the base 11-dim uncertainty features.
    """
    return _ZERO_VEC.copy()
