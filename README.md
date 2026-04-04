# Agentic Language-ID + ASR Routing for Multilingual Speech

Confidence-aware multilingual speech pipeline that performs Language Identification (LID)
and dynamically routes audio to the best ASR decoding strategy.

---

## Quick Start

```bash
# 1. Install dependencies (LOCAL — your Windows machine)
cd End_Sem_Project
pip install -r requirements.txt

# 2. Verify setup
python setup_verify.py --local

# 3. Run unit tests (no GPU needed)
python -m pytest tests/ -v
```

---

## What Runs Where

### LOCAL Machine (Windows, GTX 1650 4GB)
| Task | Command | When |
|---|---|---|
| Install dependencies | `pip install -r requirements.txt` | Phase 1, once |
| Verify setup | `python setup_verify.py --local` | Phase 1, once |
| Run unit tests | `python -m pytest tests/ -v` | After any code change |
| Generate dashboards | `python -c "from evaluation.dashboard import generate_full_dashboard; generate_full_dashboard('results/eval_results.json')"` | Phase 6 |
| Edit code & configs | VS Code | Always |

### KAGGLE Remote (2× T4 15.64 GB)
| Task | Command / Notebook | When |
|---|---|---|
| Full setup verification | `python setup_verify.py --gpu` | Phase 1 |
| Download & cache models | See "Model Download" below | Phase 1 |
| LID evaluation (dev set) | `notebooks/01_lid_exploration.ipynb` | Phase 2 |
| Fusion tuning (α, thresholds) | `notebooks/02_fusion_tuning.ipynb` | Phase 2 |
| Full pipeline evaluation | notebook or `python -m evaluation.evaluate` | Phase 3+ |
| Train learned routing MLP | `notebooks/03_routing_analysis.ipynb` | Phase 4 |
| Ablation experiments | `python -m evaluation.ablation` | Phase 5 |
| Full 102-language eval | `python -m evaluation.evaluate` | Phase 5 |

---

## Execution Order (Phase by Phase)

### Phase 1: Environment Setup (LOCAL + KAGGLE)

**Step 1 — LOCAL:**
```bash
pip install -r requirements.txt
python setup_verify.py --local
python -m pytest tests/ -v
```

**Step 2 — KAGGLE:**
```python
# In a Kaggle notebook connected to T4 GPU:
!pip install openai-whisper speechbrain jiwer
!python setup_verify.py --gpu

# Download & cache models (run once, they persist in Kaggle datasets)
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from transformers import Wav2Vec2ForCTC, AutoProcessor

# MMS-LID-4017 (~4 GB download)
AutoFeatureExtractor.from_pretrained("facebook/mms-lid-4017")
Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-4017")

# MMS-1b-all ASR (~4 GB download)
AutoProcessor.from_pretrained("facebook/mms-1b-all")
Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")

# Whisper large-v3 (~3 GB download)
import whisper
whisper.load_model("large-v3")

# Test FLEURS data access
from datasets import load_dataset, Audio
ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
sample = next(iter(ds))
print(f"Audio shape: {sample['audio']['array'].shape}, SR: {sample['audio']['sampling_rate']}")
```

### Phase 2: LID & Fusion (KAGGLE)

```python
# Quick LID test
from src.pipeline import Pipeline
pipe = Pipeline()
pipe.load_models()

# Test on one sample
import numpy as np
result = pipe.run_lid_only(sample['audio']['array'].astype(np.float32))
print(f"Top language: {result['top_language']}")
print(f"Routing: {result['routing_decision']}")

# Full LID evaluation on dev set
from evaluation.evaluate import evaluate_lid_only
results = evaluate_lid_only(pipe, split="validation", max_per_lang=50)
print(f"LID Accuracy: {results.lid_accuracy():.3f}")
```

### Phase 3: Full Pipeline (KAGGLE)

```python
# Full pipeline test
output = pipe.run(sample['audio']['array'].astype(np.float32))
print(f"Transcript: {output.transcript}")
print(f"Language: {output.detected_language}")
print(f"Mode: {output.routing_mode}")
print(f"Backend: {output.backend_used}")

# Full evaluation
from evaluation.evaluate import evaluate_full
results = evaluate_full(pipe, max_per_lang=30, save_path="results/eval_results.json")
```

### Phase 4: Learned Policy (KAGGLE)

```python
# Generate training data from dev set
from src.routing.policy_learned import generate_oracle_labels, LearnedRoutingPolicy
# ... (see notebooks/03_routing_analysis.ipynb)
```

### Phase 5: Ablations (KAGGLE)

```python
from evaluation.ablation import run_all_ablations
run_all_ablations(max_per_lang=30, save_dir="results/ablations")
```

### Phase 6: Dashboard (LOCAL)

```bash
# Download results/ folder from Kaggle, then:
python -c "from evaluation.dashboard import generate_full_dashboard; generate_full_dashboard('results/eval_results.json')"
```

---

## Project Structure

```
├── config/
│   ├── model_config.yaml       # All thresholds, model IDs, hyperparams
│   ├── language_map.yaml       # ISO code mappings (30 languages)
│   └── confusion_clusters.yaml # Known confusion pairs
│
├── src/
│   ├── utils.py                # Config, language map, data classes, logging
│   ├── preprocessing.py        # Load audio, VAD (Silero), segmentation
│   ├── lid/
│   │   ├── acoustic_lid.py     # MMS-LID-4017 wrapper (4017 langs)
│   │   ├── decoder_lid.py      # Whisper detect_language (99 langs)
│   │   ├── baseline_lid.py     # SpeechBrain ECAPA-TDNN (ablation only)
│   │   └── fusion.py           # Dual-signal fusion + uncertainty
│   ├── routing/
│   │   ├── agent.py            # Routing agent (wraps policy)
│   │   ├── policy_rules.py     # Threshold-based rules (Mode A/B/C)
│   │   ├── policy_learned.py   # MLP routing policy (Phase 4)
│   │   └── confusion_map.py    # Confusion cluster lookup
│   ├── asr/
│   │   ├── whisper_backend.py  # Whisper ASR (99 langs, high quality)
│   │   ├── mms_backend.py      # MMS-1b-all ASR (1162 langs, adapters)
│   │   └── backend_selector.py # Pick best backend per language
│   ├── decoding/
│   │   ├── single_decode.py    # Mode A: one language, one decode
│   │   ├── multi_hypothesis.py # Mode B: top-k langs, rerank
│   │   └── fallback_decode.py  # Mode C: tiered fallback
│   └── pipeline.py             # End-to-end orchestrator
│
├── evaluation/
│   ├── data_loader.py          # FLEURS dataset loading
│   ├── metrics.py              # CER, WER, LID accuracy, routing stats
│   ├── evaluate.py             # Main evaluation runner
│   ├── ablation.py             # A1–A8 ablation experiments
│   └── dashboard.py            # Visualization generation
│
├── tests/                      # Unit tests (run locally, no GPU)
├── results/                    # Generated results (gitignored)
├── notebooks/                  # Kaggle notebooks (created per phase)
├── requirements.txt
├── setup_verify.py             # Environment verification
├── Final_Plan.md               # Full project plan
└── README.md                   # This file
```

---

## Model Inventory

| Model | Role | VRAM (FP16) | Languages |
|---|---|---|---|
| `facebook/mms-lid-4017` | Primary LID | ~2 GB | 4,017 |
| `openai/whisper-large-v3` | Secondary LID + ASR | ~3 GB | 99 |
| `facebook/mms-1b-all` | Broad ASR | ~2 GB | 1,162 |
| `speechbrain/lang-id-voxlingua107-ecapa` | Ablation baseline | ~0.1 GB | 107 |

Total simultaneous: ~7 GB FP16 (fits on T4 with room to spare).

---

## Key Design Decisions (vs. the Plan)

1. **Whisper model is SHARED** between DecoderLID and WhisperBackend — saves 3 GB VRAM.
2. **VAD uses Silero** (from torch.hub, ~1 MB) — lightweight and excellent.
3. **Language map singleton** — normalized once, used everywhere via `get_language_map()`.
4. **Plan's FLEURS bug fixed** — `hi_in` was listed twice in the 30-language subset.
5. **Reranking in Mode B** uses decoder confidence + character plausibility heuristic
   (no external LM needed — avoids an extra dependency).
6. **All models support load/unload** for sequential loading on constrained GPUs.
