# Agentic Language-ID + ASR Routing for Multilingual Speech
## Final Implementation Plan

---

## 1. Executive Summary

This project builds a **confidence-aware multilingual speech pipeline** that performs Language Identification (LID) on unknown-language audio and dynamically routes it to the most suitable ASR decoding path. The core innovation is an **agentic routing layer** that decides between single-language, multi-hypothesis, and fallback decoding based on uncertainty signals — eliminating the brittle "guess-once-and-pray" problem of static LID→ASR cascades.

---

## 2. Critical Analysis of the Proposed Plan

The original proposed plan has a strong conceptual foundation but several gaps that would block actual implementation. Below is a summary of what was kept, what was changed, and why.

### 2.1 What Was Strong (Kept)

| Aspect | Why It's Good |
|---|---|
| Cascade error problem framing | Clearly identifies the real failure mode in multilingual ASR |
| Three-mode routing (single / multi-hypothesis / fallback) | Sound decision framework that no existing public system does well |
| Hybrid two-signal LID principle | Complementary signals reduce single-point-of-failure |
| Focus on routing, not training new ASR models | Right scope for a student project |
| Quality + efficiency dual evaluation | Shows the system isn't just accurate but also practical |

### 2.2 What Was Weak (Changed)

| Original Weakness | Problem | Fix in This Plan |
|---|---|---|
| **SpeechBrain ECAPA-TDNN as primary LID** | Only 107 languages; major coverage gap vs. the 141-language target | Replaced with **Meta MMS-LID-4017** (4017 languages, wav2vec2-based, 1B params) as primary acoustic LID |
| **"Generative LID" left unspecified** | No concrete model, no implementation path; would take weeks to build from scratch | Use **Whisper's built-in language detection** (decoder-based, naturally complementary to encoder-based MMS-LID) |
| **No concrete ASR backend selection** | "Available backends" is vague — which models, how many, how to switch? | Specify **MMS-1b-all** (1162 langs, adapter-switchable) + **Whisper large-v3/turbo** (99 langs, high quality) as the two ASR backends |
| **141 languages from day one** | Unrealistic to debug routing logic across 141 languages simultaneously | **Start with 30 strategically chosen languages** (diverse families + known confusion pairs), scale to full set in final phase |
| **No dataset access plan** | ML-SUPERB 2.0 data requires specific access; no fallback specified | Use **Google FLEURS** (102 languages, publicly available, standardized) as primary eval set, **Common Voice** as secondary |
| **Routing agent is purely rule-based** | Threshold-based rules are brittle; can't adapt to distribution shifts | Add a **learned routing policy** (lightweight MLP trained on validation signals) alongside rule-based baseline |
| **No code architecture** | 26 sections of prose, zero module design | Full module structure with class hierarchy, data flow, and interface contracts |
| **No timeline or phase deliverables** | Can't track progress or identify blockers | 6-phase plan with concrete deliverables per phase |
| **No compute/hardware planning** | Student laptop vs. 1B-parameter models is a real constraint | Explicit GPU memory estimates and model-loading strategies |
| **No ablation plan** | Can't prove which components matter | Structured ablation experiments included |
| **SpeechBrain ECAPA-TDNN dropped entirely?** | No — it's still useful | Repositioned as a **lightweight backup / ablation baseline**, not the primary acoustic LID |

### 2.3 Key Architectural Decisions Changed

1. **Primary Acoustic LID**: SpeechBrain ECAPA-TDNN → **Meta MMS-LID-4017**
   - *Why*: 4017 languages vs. 107; same inference API (HuggingFace Transformers); wav2vec2 backbone gives richer representations; directly covers essentially all ML-SUPERB target languages.

2. **Second LID Signal**: Vague "generative LID" → **Whisper Language Detection**
   - *Why*: Whisper uses a Transformer decoder to score language tokens — this is genuinely a "generative" approach (it generates language predictions from decoder logits). It's complementary to the encoder-only MMS-LID. No training needed. Supports 99 languages with probability outputs we can directly fuse.

3. **ASR Backends**: Unspecified → **Two concrete backends**
   - **MMS-1b-all**: Covers 1162 languages via adapter switching. Great for low-resource languages where Whisper struggles.
   - **Whisper large-v3 (or turbo)**: High quality for its supported languages (~99). Stronger on high-resource languages.
   - The routing agent decides *which backend* and *which language adapter* to use.

---

## 3. Final Architecture

### 3.1 System Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT AUDIO (16kHz)                       │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING MODULE                         │
│  • Resample to 16kHz mono                                        │
│  • VAD (Voice Activity Detection) — trim silence                 │
│  • Segment if > 30s                                              │
└─────────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   ACOUSTIC LID           │     │   DECODER-BASED LID      │
│   (MMS-LID-4017)         │     │   (Whisper detect_lang)   │
│                          │     │                           │
│  Input: raw waveform     │     │  Input: log-Mel features  │
│  Output: P_a(L|x) over   │     │  Output: P_d(L|x) over    │
│  4017 language classes   │     │  99 language classes       │
└────────────┬─────────────┘     └────────────┬──────────────┘
             │                                │
             └───────────────┬────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE FUSION MODULE                       │
│                                                                  │
│  • Align language codes (ISO 639-3 normalization)                │
│  • Weighted interpolation:                                       │
│    Score(L|x) = α · P_a(L|x) + (1-α) · P_d(L|x)               │
│  • Compute uncertainty signals:                                  │
│    - top1_prob, top1-top2 gap, entropy, top-k concentration      │
│  • α can be per-family or globally tuned                         │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ROUTING AGENT                               │
│                                                                  │
│  Inputs: fused LID distribution + uncertainty features           │
│                                                                  │
│  Policy (rule-based baseline + learned MLP):                     │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ IF top1_prob > θ_high AND gap > δ_high:                │      │
│  │   → MODE A: Single-Language Decode                     │      │
│  │ ELIF top1_prob > θ_med OR entropy < H_med:             │      │
│  │   → MODE B: Multi-Hypothesis Decode (top-k)            │      │
│  │ ELSE:                                                  │      │
│  │   → MODE C: Fallback Decode                            │      │
│  └────────────────────────────────────────────────────────┘      │
└──────┬──────────────────┬──────────────────┬─────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌─────────────────┐
│  MODE A:     │  │  MODE B:        │  │  MODE C:         │
│  Single-Lang │  │  Multi-Hypo     │  │  Fallback        │
│  Decode      │  │  Decode         │  │  Decode          │
│              │  │                 │  │                  │
│ Pick best    │  │ Run top-k langs │  │ Tier 1: Family   │
│ backend for  │  │ in parallel,    │  │   restriction    │
│ top-1 lang   │  │ rerank outputs  │  │ Tier 2: Broad    │
│              │  │                 │  │   MMS decode     │
│ Backend:     │  │ Backend:        │  │ Tier 3: Whisper  │
│ Whisper if   │  │ MMS adapters    │  │   --language=None│
│ supported,   │  │ for each lang   │  │   auto-detect    │
│ else MMS     │  │ + Whisper if    │  │                  │
│              │  │ applicable      │  │                  │
└──────┬───────┘  └───────┬────────┘  └────────┬─────────┘
       │                  │                     │
       └──────────────────┼─────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TRANSCRIPT SELECTION & OUTPUT                   │
│                                                                  │
│  • For Mode A: direct output                                     │
│  • For Mode B: rerank by decoder confidence + LM score           │
│  • For Mode C: output with low-confidence flag                   │
│                                                                  │
│  Output: {                                                       │
│    transcript: str,                                              │
│    detected_language: str,                                       │
│    confidence: float,                                            │
│    routing_mode: A|B|C,                                          │
│    candidates_considered: int,                                   │
│    backend_used: str,                                            │
│    lid_distribution: dict                                        │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Backend Selection Logic

| Condition | ASR Backend | Reason |
|---|---|---|
| Language is in Whisper's 99-lang set AND confidence is high | **Whisper large-v3** (or turbo for speed) | Whisper is generally stronger on high-resource languages |
| Language is NOT in Whisper's set OR is low-resource | **MMS-1b-all** with appropriate language adapter | MMS covers 1162 languages, much broader |
| Multi-hypothesis mode | **MMS adapters** for each candidate lang | Adapter switching is fast (~50ms per language); Whisper requires full re-decode |
| Fallback mode, Tier 3 | **Whisper** with `language=None` (auto) | Whisper's built-in auto-detection as a separate full-pipeline fallback |

---

## 4. Concrete Model Inventory

| Role | Model | Source | Params | Languages | Key Feature |
|---|---|---|---|---|---|
| Primary Acoustic LID | `facebook/mms-lid-4017` | HuggingFace | 1B | 4,017 | Broadest LID coverage available |
| Secondary Decoder-Based LID | `openai/whisper-large-v3` | HuggingFace/OpenAI | 1.55B | 99 | Decoder-based, complementary signal |
| ASR Backend 1 (broad) | `facebook/mms-1b-all` | HuggingFace | 1B | 1,162 | Adapter-switchable, covers low-resource |
| ASR Backend 2 (quality) | `openai/whisper-large-v3` | HuggingFace/OpenAI | 1.55B | 99 | Strong on high-resource languages |
| ASR Backend 2 (fast) | `openai/whisper-large-v3-turbo` | HuggingFace/OpenAI | 809M | 99 | 8x faster, minimal accuracy loss |
| Ablation LID Baseline | `speechbrain/lang-id-voxlingua107-ecapa` | HuggingFace | ~20M | 107 | Lightweight, good comparison point |

### 4.1 GPU Memory Estimates

| Component | VRAM (FP16) | VRAM (FP32) | Notes |
|---|---|---|---|
| MMS-LID-4017 | ~2 GB | ~4 GB | Can offload after LID inference |
| Whisper large-v3 | ~3 GB | ~6 GB | Dual-use: LID + ASR |
| MMS-1b-all | ~2 GB | ~4 GB | Adapters are ~1MB each, swapped in memory |
| SpeechBrain ECAPA | ~0.1 GB | ~0.2 GB | Very lightweight |
| **Total (simultaneous)** | **~7 GB** | **~14 GB** | Fits on a single 16GB GPU (FP16) or can be sequentially loaded |

**Practical strategy**: Use FP16 (half precision) throughout. Load MMS-LID for LID → cache result → load Whisper for secondary LID + ASR → load MMS-ASR only when needed. On a 16GB GPU (e.g., T4, RTX 4060), this is feasible with sequential loading.

---

## 5. Dataset and Evaluation Plan

### 5.1 Primary Evaluation Dataset: Google FLEURS

| Property | Value |
|---|---|
| Dataset | `google/fleurs` on HuggingFace |
| Languages | 102 languages |
| Size | ~12 hours per language (train+dev+test) |
| Format | 16kHz audio + text transcription + language label |
| Why chosen | Publicly available, standardized, covers diverse language families, used in ML-SUPERB evaluations |

### 5.2 Supplementary Dataset: Mozilla Common Voice

For languages where FLEURS coverage is insufficient or for additional robustness testing.

### 5.3 Language Subset Strategy (30 Languages for Development)

Instead of debugging on 141 languages simultaneously, we start with a **carefully chosen 30-language subset** that covers:

**Group A — High-Resource, Well-Separated (8 languages)**
- English (eng), Mandarin Chinese (cmn), Arabic (arb), Hindi (hin), Spanish (spa), French (fra), German (deu), Japanese (jpn)
- *Purpose*: Sanity check. Routing should confidently use Mode A here.

**Group B — Known Confusion Pairs (12 languages, 6 pairs)**
- Hindi (hin) ↔ Urdu (urd) — nearly identical spoken
- Serbian (srp) ↔ Croatian (hrv) — mutual intelligibility
- Indonesian (ind) ↔ Malay (msa) — very similar
- Norwegian Bokmål (nob) ↔ Norwegian Nynorsk (nno) — dialect variation
- Czech (ces) ↔ Slovak (slk) — closely related
- Portuguese-BR (por) ↔ Spanish (spa) — phonetic overlap
- *Purpose*: Stress-test routing. These should trigger Mode B (multi-hypothesis).

**Group C — Low-Resource Languages (10 languages)**
- Yoruba (yor), Swahili (swh), Amharic (amh), Welsh (cym), Basque (eus), Georgian (kat), Mongolian (mon), Nepali (nep), Sundanese (sun), Hausa (hau)
- *Purpose*: Test fallback behavior, MMS adapter quality, LID under data scarcity.

### 5.4 Confusion Map Construction

Build empirically from LID output on the dev set:
```python
# Pseudocode
confusion_matrix = {}
for audio, true_lang in fleurs_dev:
    lid_probs = run_fused_lid(audio)
    predicted = argmax(lid_probs)
    if predicted != true_lang:
        confusion_matrix[(true_lang, predicted)] += 1
```
This confusion map directly feeds into the routing agent's policy: known confusion partners are automatically considered in multi-hypothesis mode even if their individual LID score is below the normal threshold.

---

## 6. Module Design and Code Architecture

```
project/
├── config/
│   ├── model_config.yaml          # Model paths, thresholds, hyperparameters
│   ├── language_map.yaml          # ISO code mappings across models
│   └── confusion_clusters.yaml   # Known confusion pairs/families
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Audio loading, resampling, VAD, segmentation
│   ├── lid/
│   │   ├── __init__.py
│   │   ├── acoustic_lid.py        # MMS-LID-4017 wrapper
│   │   ├── decoder_lid.py         # Whisper language detection wrapper
│   │   ├── baseline_lid.py        # SpeechBrain ECAPA-TDNN (for ablation)
│   │   └── fusion.py              # Confidence fusion + uncertainty computation
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── agent.py               # Main routing agent (rule-based + learned)
│   │   ├── policy_rules.py        # Threshold-based routing rules
│   │   ├── policy_learned.py      # Lightweight MLP routing policy
│   │   └── confusion_map.py       # Confusion pair registry and lookup
│   │
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── whisper_backend.py     # Whisper ASR wrapper
│   │   ├── mms_backend.py         # MMS-1b-all ASR wrapper (adapter switching)
│   │   └── backend_selector.py    # Picks best backend for a given language
│   │
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── single_decode.py       # Mode A: single-language decode
│   │   ├── multi_hypothesis.py    # Mode B: parallel multi-lang decode + rerank
│   │   └── fallback_decode.py     # Mode C: tiered fallback
│   │
│   ├── pipeline.py                # End-to-end pipeline orchestrator
│   └── utils.py                   # Language code normalization, logging, timing
│
├── evaluation/
│   ├── evaluate.py                # Run full evaluation on FLEURS test set
│   ├── metrics.py                 # CER, WER, LID accuracy, routing stats
│   ├── ablation.py                # Ablation experiment runner
│   └── dashboard.py               # Confusion matrix visualization, routing plots
│
├── notebooks/
│   ├── 01_lid_exploration.ipynb   # Explore LID model outputs
│   ├── 02_fusion_tuning.ipynb     # Tune α and threshold parameters
│   ├── 03_routing_analysis.ipynb  # Visualize routing decisions
│   └── 04_final_results.ipynb     # Final evaluation and figures
│
├── tests/
│   ├── test_lid.py
│   ├── test_routing.py
│   ├── test_decoding.py
│   └── test_pipeline.py
│
├── requirements.txt
├── README.md
└── Final_Plan.md
```

### 6.1 Key Class Interfaces

```python
# --- LID Interface ---
class BaseLID(ABC):
    @abstractmethod
    def predict(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Returns {lang_code: probability} for all candidate languages."""
        pass

class AcousticLID(BaseLID):    # wraps MMS-LID-4017
class DecoderLID(BaseLID):     # wraps Whisper detect_language
class BaselineLID(BaseLID):    # wraps SpeechBrain ECAPA-TDNN

# --- Fusion ---
class LIDFusion:
    def fuse(self, acoustic_probs: dict, decoder_probs: dict, alpha: float = 0.6) -> dict:
        """Weighted interpolation of two LID distributions."""
        pass

    def compute_uncertainty(self, fused_probs: dict) -> UncertaintySignals:
        """Returns top1_prob, gap, entropy, top_k_concentration."""
        pass

# --- Routing Agent ---
class RoutingAgent:
    def decide(self, fused_probs: dict, uncertainty: UncertaintySignals) -> RoutingDecision:
        """Returns routing mode (A/B/C) + candidate language(s) + backend choice."""
        pass

# --- ASR Backend ---
class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str) -> TranscriptResult:
        pass

class WhisperBackend(ASRBackend):    # Whisper large-v3 / turbo
class MMSBackend(ASRBackend):        # MMS-1b-all with adapter switching
```

---

## 7. Confidence Fusion — Detailed Design

### 7.1 Language Code Alignment

MMS-LID uses ISO 639-3 codes (e.g., `eng`, `fra`, `cmn`). Whisper uses ISO 639-1/mix codes (e.g., `en`, `fr`, `zh`). A **normalization table** maps both to a common representation.

```python
# language_map.yaml (excerpt)
eng: {whisper: "en", mms_lid: "eng", mms_asr: "eng", name: "English"}
fra: {whisper: "fr", mms_lid: "fra", mms_asr: "fra", name: "French"}
cmn: {whisper: "zh", mms_lid: "cmn", mms_asr: "cmn", name: "Mandarin Chinese"}
hin: {whisper: "hi", mms_lid: "hin", mms_asr: "hin", name: "Hindi"}
```

### 7.2 Fusion Formula

For languages covered by **both** LID models:

$$\text{Score}(L \mid x) = \alpha \cdot P_{\text{acoustic}}(L \mid x) + (1-\alpha) \cdot P_{\text{decoder}}(L \mid x)$$

For languages covered **only** by MMS-LID (not in Whisper's 99):

$$\text{Score}(L \mid x) = P_{\text{acoustic}}(L \mid x)$$

with a small penalty factor since we lack confirmatory evidence from the second signal.

**Default α = 0.6** (slightly favoring the acoustic model since it covers more languages). Tuned per language family on the dev set.

### 7.3 Uncertainty Signals

| Signal | Formula | Use |
|---|---|---|
| `top1_prob` | max(Score(L\|x)) | Primary confidence |
| `gap` | Score(L₁\|x) - Score(L₂\|x) | How "clear" the winner is |
| `entropy` | -Σ Score(L\|x) · log(Score(L\|x)) | Overall distribution uncertainty |
| `top3_conc` | Σ top-3 Score(L\|x) | Whether mass is concentrated |
| `in_confusion_cluster` | Boolean: is top-1 in a known confusion pair? | Triggers cautious routing |

---

## 8. Routing Agent — Detailed Design

### 8.1 Rule-Based Policy (Baseline)

```python
def route(uncertainty: UncertaintySignals) -> RoutingMode:
    if uncertainty.top1_prob > 0.85 and uncertainty.gap > 0.3 and not uncertainty.in_confusion_cluster:
        return MODE_A  # Single-language decode
    elif uncertainty.top1_prob > 0.4 or uncertainty.entropy < 2.0:
        return MODE_B  # Multi-hypothesis (top-3)
    else:
        return MODE_C  # Fallback
```

Thresholds (`0.85`, `0.3`, `0.4`, `2.0`) are tuned on the dev set to minimize CER while keeping average decode calls per utterance low.

### 8.2 Learned Policy (Improvement)

A lightweight **2-layer MLP** (input: 6 uncertainty features → hidden: 32 → output: 3 modes) trained on dev-set oracle labels:

```
Oracle label for training:
- If top-1 language is correct → label = MODE_A
- If correct language is in top-3 → label = MODE_B
- Otherwise → label = MODE_C
```

This learned policy can potentially outperform fixed thresholds because it learns non-linear interactions between uncertainty features.

### 8.3 Confusion-Aware Override

Even if confidence is "high", if the top-1 language is a member of a **known confusion pair** (e.g., Hindi/Urdu, Serbian/Croatian), the agent downgrades to Mode B and includes the confusion partner in the top-k candidates.

```yaml
# confusion_clusters.yaml
- cluster: south_asian_script_neutral
  languages: [hin, urd]
  note: "Nearly identical spoken form, script differs"

- cluster: south_slavic
  languages: [srp, hrv, bos]
  note: "Former Serbo-Croatian, mutual intelligibility"

- cluster: malay_indonesian
  languages: [ind, msa]
  note: "Very similar vocabulary and phonetics"

- cluster: scandinavian
  languages: [nob, nno, swe, dan]
  note: "High phonetic overlap"

- cluster: iberian
  languages: [spa, por, cat, glg]
  note: "Romance language overlap"
```

---

## 9. Decoding Modes — Detailed Design

### 9.1 Mode A: Single-Language Decode

```
Input: top-1 language L₁, audio x
Logic:
  1. Check if L₁ is in Whisper's language set
  2. If yes → Whisper transcribe(x, language=L₁)
  3. If no → MMS transcribe(x, adapter=L₁)
Output: single transcript + confidence score
```

**Average decode calls: 1.0**

### 9.2 Mode B: Multi-Hypothesis Decode

```
Input: top-k languages [L₁, L₂, ..., Lₖ], audio x (k=3 default)
Logic:
  1. For each Lᵢ in top-k:
     a. Select best backend for Lᵢ
     b. transcripts[i] = backend.transcribe(x, language=Lᵢ)
  2. Score each transcript:
     - decoder_confidence: avg log-prob from the ASR model
     - lm_plausibility: character-level entropy (lower = more plausible)
     - lid_prior: original LID score for Lᵢ
  3. Rerank: combined_score = w₁·decoder_conf + w₂·lm_plaus + w₃·lid_prior
  4. Return best-scoring transcript
Output: best transcript + all candidates + ranking metadata
```

**Average decode calls: 2-3**

**Practical optimization**: For MMS-based decoding, adapter switching is very fast (~50ms). So running 3 MMS decodes (3 different adapters) is much cheaper than running 3 Whisper decodes.

### 9.3 Mode C: Fallback Decode

**Tier 1 — Family-Aware Restriction**
- Use the confusion map to identify the language family of the top-1 guess
- Restrict candidate set to that family
- Run Mode-B-style multi-hypothesis within the family

**Tier 2 — Broad MMS Decode**
- Use MMS-1b-all with the top-5 most probable language adapters
- Rerank by decoder confidence

**Tier 3 — Whisper Auto-Detect**
- Run Whisper with `language=None` (its own built-in auto-detection + transcription)
- This is a completely independent pipeline that may catch errors the primary pipeline misses
- Output is flagged as low-confidence

**Average decode calls: 3-6**

---

## 10. Evaluation Framework

### 10.1 Quality Metrics

| Metric | Description | How Computed |
|---|---|---|
| **LID Accuracy** | % of utterances where the final language prediction matches ground truth | Per-language and overall |
| **CER** | Character Error Rate of final transcript vs. ground truth | Standard CER computation |
| **WER** | Word Error Rate (for languages with word boundaries) | Where applicable |
| **CER@oracle** | CER when the correct language is given to ASR (upper bound) | Baseline comparison |
| **CER gain** | CER@oracle - CER@system (gap that routing is trying to close) | Key metric |

### 10.2 Routing Metrics

| Metric | Description | Target |
|---|---|---|
| **Mode distribution** | % utterances in Mode A / B / C | A: ~70%, B: ~20%, C: ~10% |
| **Avg decode calls** | Mean number of ASR calls per utterance | < 1.5 (close to 1 = efficient) |
| **Routing accuracy** | % of Mode A decisions that were correct | > 95% |
| **Recovery rate** | % of Mode B/C utterances where the correct language was recovered | Key metric |
| **Latency overhead** | Extra time from LID + routing vs. oracle-language direct decode | < 2x |

### 10.3 Ablation Experiments

| Experiment | What's Removed | What It Tests |
|---|---|---|
| A1: Single LID only (MMS-LID) | Whisper LID signal | Value of dual-signal LID |
| A2: Single LID only (Whisper) | MMS-LID signal | Value of broad-coverage acoustic LID |
| A3: SpeechBrain ECAPA as acoustic LID | MMS-LID replaced with ECAPA | Is the bigger model worth it? |
| A4: No routing agent (always Mode A) | Routing agent | Value of the routing layer |
| A5: No confusion-aware override | Confusion cluster logic | Value of confusion awareness |
| A6: Rule-based only vs. learned policy | MLP policy | Does learning improve over rules? |
| A7: MMS-only ASR (no Whisper) | Whisper ASR backend | Value of dual ASR backends |
| A8: Whisper-only ASR (no MMS) | MMS ASR backend | Value of broad-coverage backend |

### 10.4 Baselines for Comparison

| Baseline | Description |
|---|---|
| **B1: Oracle Language** | Ground-truth language given to best ASR → upper bound on CER |
| **B2: Whisper Auto** | Whisper with `language=None` → strong single-model baseline |
| **B3: MMS-LID → MMS-ASR (static)** | Static pipeline: acoustic LID → direct ASR routing, no agent |
| **B4: SpeechBrain LID → Whisper (static)** | Lightweight LID → Whisper, no agent |

---

## 11. Phase-Wise Implementation Plan

### Phase 1: Environment Setup & Data Pipeline (Week 1)

**Objective**: Set up the development environment, download models and datasets, build the data loading pipeline.

> **Workflow Note**: All GPU work runs on Kaggle T4 via VS Code remote runtime. For every new notebook created during this or any subsequent phase:
> 1. Create the `.ipynb` file locally or in the Kaggle web UI
> 2. **Connect the notebook to the Kaggle T4 remote runtime in VS Code** (bottom-right runtime selector)
> 3. Only then run any GPU-dependent cells
> Kaggle CLI (`kaggle kernels push/pull/output`) is used for batch submission when interactive development is not needed.

**Deliverables**:
- [ ] Python environment with all dependencies (`requirements.txt`)
- [x] GPU/CUDA verified on Kaggle T4: PyTorch 2.10.0+cu128, CUDA 12.8, 2× Tesla T4 @ 15.64 GB ✅
- [ ] Model download and caching scripts for all 4 models
- [ ] FLEURS dataset loader (30-language subset)
- [ ] Audio preprocessing module (resample, VAD, segmentation)
- [ ] Language code normalization table
- [ ] Basic project structure (all directories and `__init__.py` files)

**Implementation Details**:
```bash
# Key dependencies
pip install torch torchaudio transformers datasets speechbrain
pip install openai-whisper    # or use HuggingFace transformers for Whisper
pip install numpy scipy scikit-learn matplotlib seaborn
pip install jiwer              # for WER/CER computation
```

```python
# Data loading (evaluation/data_loader.py)
from datasets import load_dataset, Audio

SUBSET_LANGS = [
    "en_us", "zh_cn", "ar_eg", "hi_in", "es_419", "fr_fr", "de_de", "ja_jp",
    "hi_in", "ur_pk", "sr_rs", "hr_hr", "id_id", "ms_my",
    "nb_no", "nn_no", "cs_cz", "sk_sk", "pt_br",
    "yo_ng", "sw_ke", "am_et", "cy_gb", "eu_es",
    "ka_ge", "mn_mn", "ne_np", "su_id", "ha_ng"
]

def load_fleurs_subset(split="test"):
    datasets = {}
    for lang in SUBSET_LANGS:
        ds = load_dataset("google/fleurs", lang, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        datasets[lang] = ds
    return datasets
```

**Validation Checkpoint**: Can load audio from FLEURS, run it through each model individually, get outputs.

---

### Phase 2: LID Components & Fusion (Week 2)

**Objective**: Implement both LID models, the fusion module, and uncertainty computation. Validate LID accuracy on the 30-language subset.

**Deliverables**:
- [ ] `AcousticLID` class wrapping MMS-LID-4017
- [ ] `DecoderLID` class wrapping Whisper language detection
- [ ] `BaselineLID` class wrapping SpeechBrain ECAPA-TDNN
- [ ] `LIDFusion` class with weighted interpolation
- [ ] Uncertainty signal computation
- [ ] LID accuracy evaluation on FLEURS dev set (all three LID variants + fusion)
- [ ] Confusion matrix visualization (30×30 for subset)
- [ ] Initial α tuning on dev set

**Implementation Details**:

```python
# src/lid/acoustic_lid.py
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

class AcousticLID:
    def __init__(self, model_id="facebook/mms-lid-4017", device="cuda"):
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    def predict(self, audio: np.ndarray, sr: int = 16000) -> dict:
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # Map to language codes
        return {self.model.config.id2label[i]: p.item()
                for i, p in enumerate(probs) if p.item() > 0.001}

# src/lid/decoder_lid.py
import whisper

class DecoderLID:
    def __init__(self, model_size="large-v3", device="cuda"):
        self.model = whisper.load_model(model_size, device=device)
        self.device = device

    def predict(self, audio: np.ndarray, sr: int = 16000) -> dict:
        audio = whisper.pad_or_trim(audio.astype(np.float32))
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.device)
        _, probs = self.model.detect_language(mel)
        return probs  # dict of {lang_code: probability}
```

**Validation Checkpoint**: LID accuracy numbers for each model variant. Expected: MMS-LID ~85-93% on FLEURS, Whisper ~80-90%, Fused > either individual.

---

### Phase 3: Routing Agent & Decoding Modes (Week 3)

**Objective**: Implement the routing agent (rule-based) and all three decoding modes. End-to-end pipeline works.

**Deliverables**:
- [ ] `RoutingAgent` with rule-based policy
- [ ] `single_decode.py` — Mode A implementation
- [ ] `multi_hypothesis.py` — Mode B with reranking
- [ ] `fallback_decode.py` — Mode C tiered fallback
- [ ] `WhisperBackend` ASR wrapper
- [ ] `MMSBackend` ASR wrapper with adapter switching
- [ ] `BackendSelector` logic
- [ ] `pipeline.py` — full end-to-end orchestrator
- [ ] Confusion cluster configuration file
- [ ] Initial threshold tuning on dev set

**Implementation Details**:

```python
# src/routing/agent.py
from dataclasses import dataclass
from enum import Enum

class RoutingMode(Enum):
    SINGLE = "A"
    MULTI_HYPOTHESIS = "B"
    FALLBACK = "C"

@dataclass
class RoutingDecision:
    mode: RoutingMode
    candidate_languages: list  # [lang_code, ...]
    backends: list             # [backend_name, ...]
    confidence: float
    reason: str

class RoutingAgent:
    def __init__(self, config):
        self.theta_high = config.get("theta_high", 0.85)
        self.delta_high = config.get("delta_high", 0.3)
        self.theta_med = config.get("theta_med", 0.4)
        self.entropy_med = config.get("entropy_med", 2.0)
        self.confusion_map = load_confusion_map(config["confusion_clusters_path"])

    def decide(self, fused_probs, uncertainty):
        top1_lang = max(fused_probs, key=fused_probs.get)
        in_confusion = self.confusion_map.is_confused(top1_lang)

        if (uncertainty.top1_prob > self.theta_high
            and uncertainty.gap > self.delta_high
            and not in_confusion):
            return RoutingDecision(
                mode=RoutingMode.SINGLE,
                candidate_languages=[top1_lang],
                backends=[select_best_backend(top1_lang)],
                confidence=uncertainty.top1_prob,
                reason="High confidence, clear winner"
            )
        elif uncertainty.top1_prob > self.theta_med or uncertainty.entropy < self.entropy_med:
            top_k = sorted(fused_probs, key=fused_probs.get, reverse=True)[:3]
            if in_confusion:
                # Add confusion partners even if not in top-k
                partners = self.confusion_map.get_partners(top1_lang)
                top_k = list(set(top_k + partners))[:5]
            return RoutingDecision(
                mode=RoutingMode.MULTI_HYPOTHESIS,
                candidate_languages=top_k,
                backends=[select_best_backend(l) for l in top_k],
                confidence=uncertainty.top1_prob,
                reason="Medium confidence, exploring alternatives"
            )
        else:
            return RoutingDecision(
                mode=RoutingMode.FALLBACK,
                candidate_languages=sorted(fused_probs, key=fused_probs.get, reverse=True)[:5],
                backends=["mms", "whisper_auto"],
                confidence=uncertainty.top1_prob,
                reason="Low confidence, fallback triggered"
            )
```

```python
# src/asr/mms_backend.py
from transformers import Wav2Vec2ForCTC, AutoProcessor

class MMSBackend:
    def __init__(self, model_id="facebook/mms-1b-all", device="cuda"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        self.device = device
        self._current_lang = None

    def transcribe(self, audio, language):
        if language != self._current_lang:
            self.processor.tokenizer.set_target_lang(language)
            self.model.load_adapter(language)
            self._current_lang = language

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        transcription = self.processor.decode(ids)
        return TranscriptResult(text=transcription, language=language, confidence=compute_confidence(logits))
```

**Validation Checkpoint**: Full pipeline runs end-to-end on a single audio file. All three routing modes are triggered for different test cases.

---

### Phase 4: Learned Policy & Threshold Tuning (Week 4)

**Objective**: Train the learned routing MLP, optimize thresholds, run full evaluation on the 30-language subset.

**Deliverables**:
- [ ] Training data generation from dev set (uncertainty features → oracle routing labels)
- [ ] Trained MLP routing policy
- [ ] Optimized thresholds for rule-based policy
- [ ] Optimal α values per language family
- [ ] Comparative evaluation: rule-based vs. learned policy
- [ ] Full CER + routing metrics on 30-language test set

**Implementation Details**:

```python
# src/routing/policy_learned.py
import torch.nn as nn

class LearnedRoutingPolicy(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_modes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes)
        )

    def forward(self, uncertainty_features):
        # uncertainty_features: [top1_prob, gap, entropy, top3_conc, in_confusion, only_one_lid]
        return self.net(uncertainty_features)

# Training: generate labels from dev set oracle
def generate_training_data(dev_set, lid_system):
    X, y = [], []
    for audio, true_lang in dev_set:
        fused_probs, uncertainty = lid_system.run(audio)
        features = uncertainty.to_vector()
        # Oracle label
        sorted_langs = sorted(fused_probs, key=fused_probs.get, reverse=True)
        if sorted_langs[0] == true_lang:
            label = 0  # Mode A would have been correct
        elif true_lang in sorted_langs[:3]:
            label = 1  # Mode B would have recovered it
        else:
            label = 2  # Mode C needed
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)
```

**Validation Checkpoint**: Learned policy has > rule-based policy accuracy on held-out portion of dev set. CER measurements completed.

---

### Phase 5: Scale to Full Language Set & Ablation Studies (Week 5)

**Objective**: Extend from 30 to all available FLEURS languages. Run all ablation experiments. Generate complete evaluation results.

**Deliverables**:
- [ ] Full pipeline evaluation on all 102 FLEURS languages
- [ ] All 8 ablation experiments completed (A1–A8)
- [ ] All 4 baselines computed (B1–B4)
- [ ] Updated confusion map with full language coverage
- [ ] Per-language and per-family result tables
- [ ] Statistical significance tests (paired bootstrap)

**Key Work**:
1. Scale data pipeline to all FLEURS languages
2. Run full evaluation (will take several hours on GPU)
3. Run each ablation variant
4. Compile results into tables and figures

---

### Phase 6: Analysis, Dashboard & Report (Week 6)

**Objective**: Build analysis visualizations, write up findings, package final deliverables.

**Deliverables**:
- [ ] Interactive confusion matrix heatmap
- [ ] Routing mode distribution plots (per language, per family)
- [ ] CER vs. routing efficiency tradeoff curve (Pareto front)
- [ ] Language-level drill-down: "where does the agent help most?"
- [ ] Failure case analysis: "where does the agent still fail?"
- [ ] Final report / presentation with all results
- [ ] Clean, documented codebase on GitHub

**Dashboard Visualizations**:

1. **Confusion Heatmap**: 30×30 (or 102×102) matrix showing LID confusion rates
2. **Routing Sankey Diagram**: Flow from LID confidence buckets → routing modes → outcomes (correct/incorrect)
3. **CER Comparison Bar Chart**: Our system vs. all baselines and ablations, per language group
4. **Efficiency Plot**: CER improvement (y-axis) vs. average decode calls (x-axis)
5. **Confidence Calibration**: Are the confidence scores actually reliable? Reliability diagram.

---

## 12. Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| GPU memory overflow with multiple models | Pipeline crashes | Sequential model loading; use FP16; use Whisper turbo instead of large-v3 |
| MMS-LID-4017 download is 4+ GB | Slow setup | Cache models locally; use `TRANSFORMERS_CACHE` |
| FLEURS missing some target languages | Incomplete evaluation | Fall back to Common Voice or VoxLingua107 test sets |
| Multi-hypothesis decoding is too slow | Impractical system | Cap top-k at 3; use MMS (fast adapter switching) over Whisper for multi-hypo |
| Learned policy overfits on dev set | Misleading results | Use k-fold cross-validation; keep test set untouched |
| Confusion pairs dominate errors | Routing can't help | Acceptable — document as a known limitation; confusion pairs are fundamentally hard |
| Whisper's 99 languages miss many FLEURS languages | Second LID signal unavailable | Gracefully degrade to single-signal (MMS-only) for unsupported languages |

---

## 13. Requirements Summary

### Hardware

> **Confirmed Setup (April 2026)**
> - **Primary GPU environment**: Kaggle Notebooks — **2× Tesla T4 (15.64 GB each)** — connected directly to VS Code via the Kaggle remote runtime. Outputs stream to VS Code; Kaggle CLI is used for batch push/pull where needed.
> - **Local machine**: Windows, GTX 1650 4GB — used for code writing, light testing, and file management only (no GPU workloads locally).
> - **Runtime link protocol**: Whenever a new notebook is created for any phase, **connect it to the Kaggle T4 remote runtime in VS Code before executing any GPU cells**.

- **Actual hardware**: 2× Tesla T4 @ 15.64 GB VRAM each (Kaggle dual-GPU)
- **Minimum (reference)**: NVIDIA GPU with 8GB VRAM (sequential model loading, Whisper turbo instead of large-v3)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (concurrent loading, Whisper large-v3)
- **Achieved**: 2× 15.64 GB T4 — full concurrent model loading, Whisper large-v3 feasible
- **CPU-only fallback**: Possible for LID exploration only (very slow for ASR)

### Software
```
Python 3.9+
PyTorch 2.0+
transformers >= 4.30
datasets
speechbrain
openai-whisper
torchaudio
numpy, scipy, scikit-learn
matplotlib, seaborn, plotly
jiwer (CER/WER computation)
pyyaml (config files)
tqdm
```

### Time Estimate (Phases)
| Phase | Duration | Cumulative |
|---|---|---|
| Phase 1: Setup & Data | 1 week | Week 1 |
| Phase 2: LID & Fusion | 1 week | Week 2 |
| Phase 3: Routing & Decoding | 1 week | Week 3 |
| Phase 4: Learning & Tuning | 1 week | Week 4 |
| Phase 5: Scale & Ablation | 1 week | Week 5 |
| Phase 6: Analysis & Report | 1 week | Week 6 |

---

## 14. Summary of Contributions

This project makes the following contributions:

1. **Dual-signal LID fusion** combining encoder-based (MMS-LID-4017, 4017 languages) and decoder-based (Whisper, 99 languages) language identification with confidence-calibrated interpolation.

2. **Confidence-aware routing agent** that dynamically selects between single-language, multi-hypothesis, and tiered fallback decoding based on uncertainty features — the core novelty.

3. **Learned routing policy** (lightweight MLP) that outperforms static threshold-based routing by learning non-linear relationships between uncertainty features.

4. **Confusion-aware design** with empirically derived confusion maps that modify routing behavior for known hard language pairs.

5. **Dual ASR backend system** leveraging both Whisper (high quality, 99 languages) and MMS (broad coverage, 1162 languages) with intelligent backend selection.

6. **Comprehensive evaluation** on FLEURS across 102 languages with 8 ablation experiments, 4 baselines, and both quality and efficiency metrics.

7. **Practical analysis dashboard** with confusion matrices, routing flow visualizations, and CER-vs-efficiency tradeoff curves.

---

## 15. Key Improvement Over Proposed Plan — Summary

| Aspect | Proposed Plan | Final Plan |
|---|---|---|
| Acoustic LID | ECAPA-TDNN (107 langs) | **MMS-LID-4017 (4017 langs)** |
| Second LID | "Generative LID" (unspecified) | **Whisper detect_language (99 langs, concrete)** |
| ASR Backends | Unspecified | **MMS-1b-all + Whisper large-v3 (concrete)** |
| Language count | 141 from start | **30 → 102 (phased scaling)** |
| Dataset | ML-SUPERB 2.0 (access unclear) | **FLEURS (public, standardized)** |
| Routing policy | Rule-based only | **Rule-based + Learned MLP** |
| Code architecture | None | **Full module design with interfaces** |
| Ablation plan | None | **8 systematic ablation experiments** |
| Timeline | None | **6-phase, 6-week plan** |
| Hardware planning | None | **GPU memory estimates + loading strategy** |
