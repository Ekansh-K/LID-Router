"""
Audio preprocessing: loading, resampling, VAD, segmentation.

Runs fine on CPU (no GPU needed) — safe for local machine.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from src.utils import get_logger, load_config, timed

log = get_logger("preprocessing")

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(path: str | Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target_sr. Returns (waveform, sr)."""
    import torchaudio
    waveform, sr = torchaudio.load(str(path))
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy(), target_sr


def load_audio_from_array(audio_array: np.ndarray, source_sr: int,
                          target_sr: int = 16000) -> np.ndarray:
    """Resample an already-loaded numpy array."""
    if source_sr == target_sr:
        return audio_array.astype(np.float32)
    import torchaudio
    import torch
    waveform = torch.from_numpy(audio_array).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
    resampled = resampler(waveform)
    return resampled.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Voice Activity Detection (Silero VAD — loaded from torch.hub, ~1MB)
# ---------------------------------------------------------------------------
_vad_model = None
_vad_utils = None

def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
    return _vad_model, _vad_utils


def apply_vad(audio: np.ndarray, sr: int = 16000,
              threshold: float = 0.5,
              min_speech_ms: int = 250) -> np.ndarray:
    """Remove non-speech segments using Silero VAD.
    
    Returns trimmed audio containing only speech regions.
    Falls back to returning original audio if VAD fails.
    """
    import torch
    try:
        model, utils = _load_vad()
        get_speech_timestamps = utils[0]

        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.squeeze()

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
        )

        if not speech_timestamps:
            log.warning("VAD found no speech — returning original audio")
            return audio

        # Stitch speech regions together
        chunks = []
        for ts in speech_timestamps:
            chunks.append(audio[ts["start"]:ts["end"]])
        return np.concatenate(chunks)

    except Exception as e:
        log.warning(f"VAD failed ({e}), returning original audio")
        return audio


# ---------------------------------------------------------------------------
# Segmentation (split long audio for models with max-length constraints)
# ---------------------------------------------------------------------------
def segment_audio(audio: np.ndarray, sr: int = 16000,
                  max_duration_sec: float = 30.0,
                  overlap_sec: float = 1.0) -> List[np.ndarray]:
    """Split audio into overlapping chunks of at most max_duration_sec.
    
    For short audio (≤ max_duration_sec), returns a single-element list.
    """
    max_samples = int(max_duration_sec * sr)
    overlap_samples = int(overlap_sec * sr)

    if len(audio) <= max_samples:
        return [audio]

    segments = []
    start = 0
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        segments.append(audio[start:end])
        if end >= len(audio):
            break
        start = end - overlap_samples  # overlap for continuity
    return segments


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------
@timed
def preprocess(audio: np.ndarray, sr: int = 16000,
               apply_vad_flag: bool = True,
               segment: bool = True,
               config: Optional[dict] = None) -> List[np.ndarray]:
    """Full preprocessing: resample → VAD → segment.
    
    Args:
        audio: raw waveform (numpy float32)
        sr: sampling rate
        apply_vad_flag: whether to run VAD
        segment: whether to split long audio
        config: preprocessing config dict (optional)
    
    Returns:
        List of preprocessed audio segments ready for LID/ASR.
    """
    if config is None:
        try:
            config = load_config()["preprocessing"]
        except Exception:
            config = {"target_sr": 16000, "max_duration_sec": 30,
                      "vad_threshold": 0.5, "min_speech_duration_ms": 250}

    target_sr = config.get("target_sr", 16000)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        audio = load_audio_from_array(audio, sr, target_sr)
        sr = target_sr

    # VAD
    if apply_vad_flag:
        audio = apply_vad(
            audio, sr,
            threshold=config.get("vad_threshold", 0.5),
            min_speech_ms=config.get("min_speech_duration_ms", 250),
        )

    # Segment
    if segment:
        return segment_audio(
            audio, sr,
            max_duration_sec=config.get("max_duration_sec", 30),
        )
    return [audio]
