import os
import sys
import tempfile
import traceback
import json
import random
import io
import urllib.parse

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torchaudio
import numpy as np

# Ensure the repo root is on sys.path so both src/ and evaluation/ are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import Pipeline
from src.utils import get_logger, get_language_map
from evaluation.data_loader import load_fleurs, iterate_fleurs

log = get_logger("ui_backend")

app = FastAPI(title="LID Router UI API")

# Allow CORS for the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Expected-Text", "X-Stripped-Text"],
)

# ── Globals ────────────────────────────────────────────────────────────────────
pipeline_instance: Pipeline | None = None
_records_cache: list | None = None          # all per-utterance records (cached)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global pipeline_instance
    try:
        log.info("Initializing LID Router Pipeline …")
        pipeline_instance = Pipeline(routing_policy="learned")
        
        # Load the trained MLP policy checkpoint if using learned routing
        if pipeline_instance.routing_agent.policy_name == "learned":
            import glob
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            pt_files = glob.glob(os.path.join(root_dir, 'models', '*.pt'))
            if pt_files:
                # Use the first available checkpoint
                pipeline_instance.routing_agent.load_learned_policy(pt_files[0])
                log.info(f"Loaded learned routing policy from {pt_files[0]}")
            else:
                log.warning("No trained routing policy (.pt) found in models/. Evaluation may fail if required.")

        log.info("Loading models (first run downloads ~10 GB, subsequent runs are fast) …")
        pipeline_instance.load_models(sequential=False)
        log.info("Models loaded successfully.")
    except Exception:
        log.error(f"Error initialising pipeline:\n{traceback.format_exc()}")
        raise


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    ready = pipeline_instance is not None and getattr(pipeline_instance, "_models_loaded", False)
    return {"status": "ready" if ready else "loading_or_error"}


# ── Records helper ─────────────────────────────────────────────────────────────
def _get_all_records() -> list:
    """Load and cache per-utterance records from the best available results JSON."""
    global _records_cache
    if _records_cache is not None:
        return _records_cache

    candidates = [
        "results/step10_phase3_f0.json",
        "results/step9_phase12_learned.json",
        "results/step8_track3_learned.json",
        "results/eval_results.json",
    ]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for rel in candidates:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            if records:
                log.info(f"Loaded {len(records)} records from {rel}")
                _records_cache = records
                return _records_cache
        except Exception as exc:
            log.warning(f"Failed to read {rel}: {exc}")

    log.error("No per-utterance records found in any results file.")
    _records_cache = []
    return _records_cache


def _numpy_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Convert a float32 numpy audio array to a WAV byte-string via torchaudio."""
    import torch
    # torchaudio.save expects (channels, samples) tensor
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, tensor, sr, format="wav")
    buf.seek(0)
    return buf.read()


# ── Random audio ───────────────────────────────────────────────────────────────
@app.get("/random_audio")
def get_random_audio():
    records = _get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No evaluation records found.")

    # Separate easy (CER ≤ 0.1) from hard (CER > 0.1); fall back to all if empty
    easy = [r for r in records if r.get("cer", 1.0) <= 0.10]
    pool = easy if easy else records

    # Try up to 10 candidates — skip any whose language can't be resolved to a
    # FLEURS code (avoids silently returning None audio).
    lang_map = get_language_map()
    record = None
    fleurs_code = None
    for _ in range(10):
        cand = random.choice(pool)
        fcode = lang_map.to_fleurs(cand.get("true_lang", ""))
        if fcode:
            record = cand
            fleurs_code = fcode
            break

    if record is None or fleurs_code is None:
        raise HTTPException(
            status_code=404,
            detail="Could not resolve any sampled record to a FLEURS language code."
        )

    true_lang = record["true_lang"]
    reference = record.get("reference", "")
    log.info(f"Random audio: lang={true_lang} fleurs={fleurs_code} cer={record.get('cer'):.3f}")
    log.info(f"  Reference: {reference[:60]}…")

    try:
        # ── Find the matching audio clip from FLEURS ──
        target_audio: np.ndarray | None = None
        target_sr = 16000

        datasets = load_fleurs([fleurs_code], split="test", streaming=True)

        if not datasets:
            raise RuntimeError(
                f"load_fleurs returned empty dict for '{fleurs_code}'. "
                "Ensure datasets==2.20.0 is installed and the server was restarted."
            )

        # First pass: exact reference match (search up to 200 samples)
        for audio, sr, _, _, ref_text in iterate_fleurs(datasets, max_per_lang=200):
            if ref_text.strip().lower() == reference.strip().lower():
                target_audio = audio
                target_sr = sr
                log.info("  Exact match found.")
                break

        # Second pass: just grab the first available sample for this language
        if target_audio is None:
            log.warning("  Exact match not found — using first available sample.")
            datasets2 = load_fleurs([fleurs_code], split="test", streaming=True)
            for audio, sr, _, _, ref_text in iterate_fleurs(datasets2, max_per_lang=1):
                target_audio = audio
                target_sr = sr
                reference = ref_text  # update reference to match what we actually return
                break

        if target_audio is None:
            raise RuntimeError(
                f"No audio samples could be streamed for FLEURS code '{fleurs_code}'."
            )

        # ── Encode to WAV (using torchaudio — avoids scipy dtype issues) ──
        wav_bytes = _numpy_to_wav_bytes(target_audio, target_sr)

        # Stripped: lowercase + basic punctuation removal
        stripped = reference.lower()
        for ch in ".,!?;:'\"()-":
            stripped = stripped.replace(ch, "")
        stripped = " ".join(stripped.split())  # normalise whitespace

        headers = {
            "X-Expected-Text": urllib.parse.quote(reference.encode("utf-8")),
            "X-Stripped-Text": urllib.parse.quote(stripped.encode("utf-8")),
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    except HTTPException:
        raise
    except Exception:
        log.error(f"Error fetching random audio:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── Process audio ──────────────────────────────────────────────────────────────
@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    if pipeline_instance is None or not getattr(pipeline_instance, "_models_loaded", False):
        raise HTTPException(status_code=503, detail="Models are still loading or failed to load.")

    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    try:
        content = await file.read()
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)

        log.info(f"Received file '{file.filename}' ({len(content)/1024:.1f} KB)")

        # Load with torchaudio and convert to mono float32 numpy
        waveform, sr = torchaudio.load(temp_path)
        audio_np = waveform.numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)  # stereo → mono

        result = pipeline_instance.run(audio_np, sr=sr)

        # Safely serialise routing_mode (may be an enum)
        routing_mode = result.routing_mode
        if hasattr(routing_mode, "value"):
            routing_mode = routing_mode.value
        elif not isinstance(routing_mode, str):
            routing_mode = str(routing_mode)

        uncertainty = None
        u = getattr(result, "uncertainty", None)
        if u is not None:
            uncertainty = {
                "top1_prob": float(getattr(u, "top1_prob", 0.0)),
                "gap": float(getattr(u, "gap", 0.0)),
                "entropy": float(getattr(u, "entropy", 0.0)),
                "in_confusion_cluster": bool(getattr(u, "in_confusion_cluster", False)),
            }

        lang_map = get_language_map()
        lang_info = lang_map.get_info(result.detected_language)
        detected_language_name = lang_info.get("name", result.detected_language) if lang_info else result.detected_language

        return {
            "transcript": result.transcript,
            "detected_language": result.detected_language,
            "detected_language_name": detected_language_name,
            "confidence": float(result.confidence),
            "routing_mode": routing_mode,
            "backend_used": result.backend_used,
            "lid_distribution": result.lid_distribution or {},
            "all_transcripts": [
                {
                    "text": t.text,
                    "language": t.language,
                    "confidence": float(t.confidence),
                    "backend": t.backend,
                }
                for t in (result.all_transcripts or [])
            ],
            "uncertainty": uncertainty,
        }

    except HTTPException:
        raise
    except Exception:
        log.error(f"Error processing audio:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
