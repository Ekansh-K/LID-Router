"""
build_notebook.py
Reads 'Kaggle_Notebook copy.ipynb', appends Steps 9-11 (Phase 1+2+3),
and writes the result to 'Kaggle_Notebook_v2.ipynb'.
Run: python build_notebook.py   (from the project root)
"""
import json, pathlib, sys

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = pathlib.Path(__file__).resolve().parent
SRC  = ROOT / "Kaggle_Notebook copy.ipynb"
DST  = ROOT / "Kaggle_Notebook_v2.ipynb"

with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)

# ── Helper: make a markdown cell ─────────────────────────────────────────────
def md(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": source}

# ── Helper: make a code cell ─────────────────────────────────────────────────
def code(cell_id, source):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source}

# ─────────────────────────────────────────────────────────────────────────────
# NEW CELLS
# ─────────────────────────────────────────────────────────────────────────────

new_cells = []

# ── STEP 9 HEADER ────────────────────────────────────────────────────────────
new_cells.append(md("step9_md", """\
---
## Step 9 — Phase 1+2: Script-Aware Routing + Temperature Scaling (~80 min total)

**Changes already made to src/ (no extra downloads needed):**

| File | What changed |
|------|-------------|
| `config/confusion_clusters.yaml` | `force_mode_b_threshold` per cluster: urd=0.97, srp=0.92 |
| `src/routing/confusion_map.py` | Exposes `force_mode_b_threshold`, `temperature`, `rerank_by_script`, `expected_script` |
| `src/routing/policy_rules.py` | Phase 2 temperature scaling + Phase 1 forced-Mode-B override |
| `src/routing/policy_learned.py` | Same post-decision override after MLP prediction |
| `src/routing/agent.py` | Shares `ConfusionMap` between both policies |
| `src/decoding/multi_hypothesis.py` | Script-aware reranking: Perso-Arabic wins for urd, Cyrillic wins for srp |

**Run: 9-Setup → 9a → 9b (in order)**\
"""))

# ── STEP 9 SETUP ─────────────────────────────────────────────────────────────
new_cells.append(code("step9_setup", """\
# ── Step 9 Setup: Reload ALL Phase 1+2 modules (run before 9a and 9b) ───────
import importlib, sys

_to_reload = [
    'src.routing.confusion_map',
    'src.routing.policy_rules',
    'src.routing.policy_learned',
    'src.routing.agent',
    'src.asr.backend_selector',
    'src.decoding.multi_hypothesis',
    'src.decoding.single_decode',
    'src.pipeline',
]
for mod in _to_reload:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

import src.asr.backend_selector as _bs
_bs._backend_prefs = None   # reset cache so updated YAML is re-read

from src.routing.confusion_map import ConfusionMap
cmap = ConfusionMap()
print('Phase 1+2 modules reloaded.')
print(f'  urd  force_mode_b_threshold = {cmap.force_mode_b_threshold(\"urd\")}')
print(f'  srp  force_mode_b_threshold = {cmap.force_mode_b_threshold(\"srp\")}')
print(f'  urd  rerank_by_script       = {cmap.rerank_by_script(\"urd\")}')
print(f'  urd  expected_script        = {cmap.expected_script(\"urd\")}')
print(f'  srp  expected_script        = {cmap.expected_script(\"srp\")}')
print(f'  ind  force_mode_b_threshold = {cmap.force_mode_b_threshold(\"ind\")}')
print(f'  eng  force_mode_b_threshold = {cmap.force_mode_b_threshold(\"eng\")}')
print('Sanity check passed — ready.')
"""))

# ── STEP 9a ───────────────────────────────────────────────────────────────────
new_cells.append(md("step9a_md", "## Step 9a — Rules Policy + Phase 1+2 (~40 min)"))
new_cells.append(code("step9a_code", """\
# ── Step 9a: Rules policy + Phase 1+2 ───────────────────────────────────────
from src.pipeline import Pipeline
from evaluation.evaluate import evaluate_full
from evaluation.data_loader import SUBSET_30_FLEURS

pipe_9a = Pipeline(routing_policy='rules')
pipe_9a.load_models()

results_9a = evaluate_full(
    pipeline=pipe_9a,
    lang_codes=SUBSET_30_FLEURS,
    split='test',
    max_per_lang=30,
    save_path='results/step9_phase12_rules.json',
)
print('\\n=== Step 9a: Phase 1+2 + Rules Policy ===')
print('CER: %.4f' % results_9a.mean_cer())
print('WER: %.4f' % results_9a.mean_wer())
print('Routing:', results_9a.routing_distribution())

pipe_9a.unload_models()

import json
with open('results/step9_phase12_rules.json') as f:
    d = json.load(f)
per = d.get('per_language', {})
print('\\nProblem languages:')
for lang in ['urd', 'srp', 'hin', 'cmn', 'lao', 'jpn', 'yor']:
    if lang in per:
        print(f'  {lang}: CER={per[lang][\"mean_cer\"]:.4f}  LID={per[lang][\"lid_accuracy\"]:.2f}')
"""))

# ── STEP 9b ───────────────────────────────────────────────────────────────────
new_cells.append(md("step9b_md", """\
## Step 9b — Learned Policy + Phase 1+2 (~40 min)

> **Requires:** `models/routing_policy.pt` from Step 6.
> No retraining — the Phase 1+2 overrides are pure Python post-decision logic.\
"""))
new_cells.append(code("step9b_code", """\
# ── Step 9b: Learned policy + Phase 1+2 ─────────────────────────────────────
from src.pipeline import Pipeline
from evaluation.evaluate import evaluate_full
from evaluation.data_loader import SUBSET_30_FLEURS

pipe_9b = Pipeline(routing_policy='learned')
pipe_9b.routing_agent.load_learned_policy('models/routing_policy.pt')
pipe_9b.load_models()

results_9b = evaluate_full(
    pipeline=pipe_9b,
    lang_codes=SUBSET_30_FLEURS,
    split='test',
    max_per_lang=30,
    save_path='results/step9_phase12_learned.json',
)
print('\\n=== Step 9b: Phase 1+2 + Learned Policy ===')
print('CER: %.4f' % results_9b.mean_cer())
print('WER: %.4f' % results_9b.mean_wer())
print('Routing:', results_9b.routing_distribution())

pipe_9b.unload_models()

import json
with open('results/step9_phase12_learned.json') as f:
    d = json.load(f)
per = d.get('per_language', {})
print('\\nProblem languages after Phase 1+2:')
for lang in ['urd', 'srp', 'hin', 'cmn', 'lao', 'jpn', 'yor', 'amh']:
    if lang in per:
        print(f'  {lang}: CER={per[lang][\"mean_cer\"]:.4f}  LID={per[lang][\"lid_accuracy\"]:.2f}')
"""))

# ── STEP 9 COMPARISON ────────────────────────────────────────────────────────
new_cells.append(md("step9c_md", "## Step 9 — Comparison vs Step 8"))
new_cells.append(code("step9c_code", """\
# ── Step 9 Comparison ────────────────────────────────────────────────────────
import json

def _cer_wer(path):
    try:
        with open(path) as f: d = json.load(f)
        return (d.get('overall_mean_cer', d.get('overall', float('nan'))),
                d.get('overall_mean_wer', float('nan')))
    except Exception:
        return float('nan'), float('nan')

b3_cer, _ = _cer_wer('results/B3_static_mms.json')

print('=' * 70)
print(f'  {\"System\":<40} {\"CER\":>8} {\"WER\":>8}  Beat B3?')
print('-' * 70)
for name, path in [
    ('B3 Static MMS (target)',        'results/B3_static_mms.json'),
    ('A6 Learned (old baseline)',     'results/ablations/a6_learned_policy.json'),
    ('Step8 Track3 Learned',          'results/step8_track3_learned.json'),
    ('Step9a Phase1+2 Rules',         'results/step9_phase12_rules.json'),
    ('Step9b Phase1+2 Learned ★',    'results/step9_phase12_learned.json'),
]:
    cer, wer = _cer_wer(path)
    ok = '✅' if cer < b3_cer else '❌' if cer == cer else '⏳'
    print(f'  {name:<40} {cer:>8.4f} {wer:>8.4f}  {ok}')
print('=' * 70)
"""))

# ── STEP 10 HEADER ────────────────────────────────────────────────────────────
new_cells.append(md("step10_md", """\
---
## Step 10 — Phase 3: F₀ Pitch Features + MLP Retrain (~60 min)

**What this step does:**
1. Installs `pyworld` (WORLD vocoder — CPU only, ~1 min install)
2. Re-collects dev-set training data augmented with 4-dim F₀ features:
   `[mean_F₀, F₀_std, F₀_range, voiced_ratio]`
3. Retrains the routing MLP with `input_dim=15` (11 base + 4 F₀)
4. Evaluates the F₀-aware pipeline on the test set

**Why F₀ helps distinguish routing for tonal languages:**
- `cmn` (4 tones): high F₀_std → route confidently to MMS
- `lao` (6 tones): very high F₀_std → MMS preferred
- `jpn` (pitch-accent): low F₀_std → distinct from cmn/lao
- `yor` (tonal): elevated F₀_std vs European langs

> Run: 10-Setup → 10a → 10b → 10c → 10-Comparison\
"""))

# ── STEP 10 SETUP ─────────────────────────────────────────────────────────────
new_cells.append(code("step10_setup", """\
# ── Step 10 Setup: Install pyworld ──────────────────────────────────────────
import subprocess, sys, importlib

result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'pyworld', '--quiet'],
    capture_output=True, text=True
)
if result.returncode == 0:
    print('pyworld installed successfully.')
else:
    print('pyworld install warning:', result.stderr[:400])

# Reload f0_features now that pyworld may be available
if 'src.lid.f0_features' in sys.modules:
    importlib.reload(sys.modules['src.lid.f0_features'])

from src.lid.f0_features import extract_f0_features, is_available
print(f'F0 extraction active: {is_available()}')

if is_available():
    import numpy as np
    test_audio = np.random.randn(16000).astype(np.float32) * 0.01
    feats = extract_f0_features(test_audio, 16000)
    print(f'Test feature vector: {feats}  (shape={feats.shape})')
    print('Fields: [mean_f0, f0_std, f0_range, voiced_ratio]')
"""))

# ── STEP 10a ─────────────────────────────────────────────────────────────────
new_cells.append(md("step10a_md", "## Step 10a — Collect F₀-Augmented Training Data (~20 min)"))
new_cells.append(code("step10a_code", """\
# ── Step 10a: Collect F0-augmented training data ────────────────────────────
from src.pipeline import Pipeline
from src.routing.policy_learned import generate_oracle_labels
from src.lid.f0_features import extract_f0_features, is_available
from evaluation.data_loader import load_fleurs, iterate_fleurs, SUBSET_30_FLEURS
import numpy as np, pathlib

print(f'F0 extraction: {\"ACTIVE\" if is_available() else \"DISABLED (zero vectors used)\"}')

pipe = Pipeline()
pipe.load_models()

fused_probs_list, uncertainty_list, true_langs, f0_list = [], [], [], []
datasets = load_fleurs(SUBSET_30_FLEURS, split='validation', streaming=True)

n = 0
for audio, sr, fleurs_code, true_lang, ref_text in iterate_fleurs(datasets, max_per_lang=50):
    lid_out = pipe.run_lid_only(audio, sr)
    fused_probs_list.append(lid_out['fused_probs'])
    uncertainty_list.append(lid_out['uncertainty'])
    true_langs.append(true_lang)
    f0_list.append(extract_f0_features(audio, sr))
    n += 1
    if n % 100 == 0:
        print(f'  Processed {n} samples...')

pipe.unload_models()
print(f'Total samples: {n}')

# Build 11-dim base features + oracle labels
X_base, y = generate_oracle_labels(fused_probs_list, uncertainty_list, true_langs)
print(f'Base (11-dim): {X_base.shape}, labels: {np.bincount(y)}')

# Append 4 F0 dims → 15-dim total
F0_mat = np.array(f0_list, dtype=np.float32)
X_f0 = np.concatenate([X_base, F0_mat], axis=1)
print(f'F0-augmented (15-dim): {X_f0.shape}')

# Save
pathlib.Path('models').mkdir(exist_ok=True)
np.save('models/X_f0_train.npy', X_f0)
np.save('models/y_f0_train.npy', y)
np.save('models/X_f0_feats.npy', F0_mat)
print('Saved: models/X_f0_train.npy, models/y_f0_train.npy')
"""))

# ── STEP 10b ─────────────────────────────────────────────────────────────────
new_cells.append(md("step10b_md", "## Step 10b — Retrain MLP with F₀ Features (input_dim=15, ~5 min)"))
new_cells.append(code("step10b_code", """\
# ── Step 10b: Retrain F0-aware MLP ──────────────────────────────────────────
from src.routing.policy_learned import LearnedRoutingPolicy
import numpy as np

X_f0 = np.load('models/X_f0_train.npy')
y    = np.load('models/y_f0_train.npy')
print(f'Training data: {X_f0.shape}, labels: {np.bincount(y)}')

policy_f0 = LearnedRoutingPolicy(
    input_dim=15,    # 6 uncertainty + 5 top probs + 4 F0
    hidden_dim=128,  # larger hidden layer for extra features
)
history_f0 = policy_f0.train_policy(X_f0, y, epochs=120, lr=0.001)
best_val = max(history_f0['val_acc'])
print(f'Best val_acc: {best_val:.4f}  |  Final val_acc: {history_f0[\"val_acc\"][-1]:.4f}')

policy_f0.save('models/routing_policy_f0.pt')
print('Saved: models/routing_policy_f0.pt')
"""))

# ── STEP 10c ─────────────────────────────────────────────────────────────────
new_cells.append(md("step10c_md", "## Step 10c — Evaluate Phase 3 F₀-Aware Pipeline (~40 min)"))
new_cells.append(code("step10c_code", """\
# ── Step 10c: Evaluate F0-aware pipeline ────────────────────────────────────
import numpy as np, json, importlib, sys
import pathlib

from src.routing.policy_learned import LearnedRoutingPolicy
from src.lid.f0_features import extract_f0_features, is_available
from src.routing.confusion_map import ConfusionMap
from src.pipeline import Pipeline
from src.routing.policy_rules import RoutingMode
from src.decoding.single_decode import decode_single
from src.decoding.multi_hypothesis import decode_multi_hypothesis
from src.decoding.fallback_decode import decode_fallback
from src.utils import PipelineOutput
from src.preprocessing import preprocess
from evaluation.evaluate import evaluate_full
from evaluation.data_loader import SUBSET_30_FLEURS


class F0Pipeline(Pipeline):
    \"\"\"Pipeline subclass that injects F0 features into routing decisions.\"\"\"

    def __init__(self, f0_policy_path, **kwargs):
        super().__init__(routing_policy='learned', **kwargs)
        cmap = self.routing_agent.confusion_map
        self._f0_policy = LearnedRoutingPolicy(
            input_dim=15, hidden_dim=128, confusion_map=cmap
        )
        self._f0_policy.load(f0_policy_path)
        print(f'F0 policy loaded ({f0_policy_path})')

    def run(self, audio, sr=16000, apply_vad=True):
        segments = preprocess(audio, sr, apply_vad_flag=apply_vad)
        main_audio = segments[0] if segments else audio

        # Dual LID + fusion (unchanged)
        acoustic_probs = self.acoustic_lid.predict(main_audio)
        decoder_probs  = self.decoder_lid.predict(main_audio)
        fused_probs, uncertainty = self.fusion.fuse_and_analyze(
            acoustic_probs, decoder_probs
        )

        # Extract F0 and attach to uncertainty
        f0_feats = extract_f0_features(main_audio, sr)
        uncertainty.f0_features = f0_feats  # stored on UncertaintySignals

        # Override: build 15-dim feature manually and call decide()
        # policy_learned.decide() uses self.input_dim=15 automatically
        # because it reads from uncertainty.to_vector_extended() via feature_vec
        # We monkey-patch the feature building for this one call:
        import torch
        base_vec = uncertainty.to_vector()       # 6-dim
        sprobs   = sorted(fused_probs.values(), reverse=True)[:5]
        sprobs  += [0.0] * (5 - len(sprobs))     # pad to 5
        feat15   = np.concatenate([base_vec, sprobs, f0_feats]).astype('float32')
        # Directly call the model forward instead of going through decide()
        self._f0_policy._model.eval()
        with torch.no_grad():
            import torch as _t
            logits = self._f0_policy._model(_t.from_numpy(feat15).unsqueeze(0))
            probs  = _t.softmax(logits, dim=-1)[0]
            mode_idx = probs.argmax().item()

        from src.routing.policy_rules import RoutingDecision
        mode_labels = [RoutingMode.SINGLE, RoutingMode.MULTI_HYPOTHESIS, RoutingMode.FALLBACK]
        raw_mode    = mode_labels[mode_idx]

        from src.utils import get_language_map
        lang_map = get_language_map()
        sorted_langs = [l for l in sorted(fused_probs, key=fused_probs.get, reverse=True)
                        if lang_map.asr_capable(l)]

        # Apply Phase 1+2 overrides (reuse from policy_learned)
        from src.routing.policy_rules import _apply_temperature
        cmap = self.routing_agent.confusion_map
        top1 = sorted_langs[0] if sorted_langs else 'unk'
        tau  = cmap.temperature(top1)
        tempered = _apply_temperature(fused_probs, tau)
        t_sorted = [l for l in sorted(tempered, key=tempered.get, reverse=True)
                    if lang_map.asr_capable(l)]
        t1p = tempered.get(t_sorted[0], 0.0) if t_sorted else 0.0
        t2p = tempered.get(t_sorted[1], 0.0) if len(t_sorted) > 1 else 0.0
        tgap = t1p - t2p

        if raw_mode == RoutingMode.SINGLE:
            fb_thresh = cmap.force_mode_b_threshold(top1)
            if fb_thresh > 0 and t1p < fb_thresh:
                raw_mode = RoutingMode.MULTI_HYPOTHESIS
            elif tgap < 0.10 and cmap.is_confused(top1):
                raw_mode = RoutingMode.MULTI_HYPOTHESIS

        if raw_mode == RoutingMode.SINGLE:
            candidates = t_sorted[:1]
        elif raw_mode == RoutingMode.MULTI_HYPOTHESIS:
            candidates = t_sorted[:3]
            for p in cmap.get_partners(top1):
                if p not in candidates and lang_map.asr_capable(p): candidates.append(p)
            candidates = candidates[:5]
        else:
            candidates = t_sorted[:5]

        decision = RoutingDecision(mode=raw_mode, candidate_languages=candidates,
                                   confidence=float(probs[mode_idx]),
                                   reason=f'F0-aware MLP (tau={tau:.1f})')

        all_transcripts = []
        if decision.mode == RoutingMode.SINGLE:
            lang = decision.candidate_languages[0]
            best = decode_single(main_audio, lang, self.whisper_backend,
                                 self.mms_backend, lid_confidence=decision.confidence)
            all_transcripts = [best]
        elif decision.mode == RoutingMode.MULTI_HYPOTHESIS:
            best, all_transcripts = decode_multi_hypothesis(
                main_audio, decision.candidate_languages, fused_probs,
                self.whisper_backend, self.mms_backend)
        else:
            best, all_transcripts = decode_fallback(
                main_audio, decision.candidate_languages, fused_probs,
                self.whisper_backend, self.mms_backend)

        return PipelineOutput(
            transcript=best.text, detected_language=best.language,
            confidence=best.confidence, routing_mode=decision.mode,
            candidates_considered=len(decision.candidate_languages),
            backend_used=best.backend,
            lid_distribution=dict(sorted(fused_probs.items(),
                                         key=lambda x: x[1], reverse=True)[:10]),
            all_transcripts=all_transcripts, uncertainty=uncertainty,
        )


pipe_f0 = F0Pipeline(f0_policy_path='models/routing_policy_f0.pt')
pipe_f0.load_models()

results_f0 = evaluate_full(
    pipeline=pipe_f0,
    lang_codes=SUBSET_30_FLEURS,
    split='test',
    max_per_lang=30,
    save_path='results/step10_phase3_f0.json',
)
print('\\n=== Step 10c: Phase 3 F0-Aware Routing ===')
print('CER: %.4f' % results_f0.mean_cer())
print('WER: %.4f' % results_f0.mean_wer())
print('Routing:', results_f0.routing_distribution())

pipe_f0.unload_models()

with open('results/step10_phase3_f0.json') as f:
    d10 = json.load(f)
per = d10.get('per_language', {})
print('\\nProblem language CER after Phase 3:')
for lang in ['urd', 'srp', 'hin', 'cmn', 'lao', 'jpn', 'yor']:
    if lang in per:
        print(f'  {lang}: CER={per[lang][\"mean_cer\"]:.4f}  '
              f'LID={per[lang][\"lid_accuracy\"]:.2f}')
"""))

# ── STEP 11 ───────────────────────────────────────────────────────────────────
new_cells.append(md("step11_md", """\
---
## Step 11 — Grand Comparison: All Systems

Run any time after Steps 9b and 10c. Only reads saved JSON files — no pipeline.\
"""))
new_cells.append(code("step11_code", """\
# ── Step 11: Grand Comparison ────────────────────────────────────────────────
import json, pathlib

def _cer_wer(path):
    try:
        with open(path) as f: d = json.load(f)
        return (d.get('overall_mean_cer', d.get('overall', float('nan'))),
                d.get('overall_mean_wer', float('nan')))
    except Exception:
        return float('nan'), float('nan')

b3_cer, _ = _cer_wer('results/B3_static_mms.json')
s8_cer, _ = _cer_wer('results/step8_track3_learned.json')

SYSTEMS = [
    ('B3 Static MMS (target)',          'results/B3_static_mms.json'),
    ('A6 Learned (no Track3)',          'results/ablations/a6_learned_policy.json'),
    ('Step8 Track3 Learned (Phase 0)',  'results/step8_track3_learned.json'),
    ('Step9a Phase1+2 Rules',           'results/step9_phase12_rules.json'),
    ('Step9b Phase1+2 Learned ★',      'results/step9_phase12_learned.json'),
    ('Step10 Phase3 F0-Aware ★★',      'results/step10_phase3_f0.json'),
]

print('=' * 72)
print(f'  {\"SYSTEM\":<42} {\"CER\":>8} {\"WER\":>8}  vs B3   vs S8')
print('=' * 72)
for name, path in SYSTEMS:
    cer, wer = _cer_wer(path)
    beat_b3 = '✅' if cer < b3_cer else ('❌' if cer == cer else '⏳')
    if 'target' in name:
        print(f'  {name:<42} {cer:>8.4f} {wer:>8.4f}')
        print('-' * 72)
    else:
        delta_b3 = cer - b3_cer
        delta_s8 = cer - s8_cer
        print(f'  {name:<42} {cer:>8.4f} {wer:>8.4f}  '
              f'{delta_b3:>+.4f}  {delta_s8:>+.4f}  {beat_b3}')
print('=' * 72)

# Per-language table for best available result
for best_path in ['results/step10_phase3_f0.json',
                  'results/step9_phase12_learned.json',
                  'results/step8_track3_learned.json']:
    if pathlib.Path(best_path).exists():
        break

try:
    with open(best_path) as f: d = json.load(f)
    with open('results/step8_track3_learned.json') as f: d8 = json.load(f)
    per_best = d.get('per_language', {})
    per_s8   = d8.get('per_language', {})
    if per_best:
        print(f'\\nPer-language CER ({pathlib.Path(best_path).stem}) vs Step8:')
        print(f'  {\"Lang\":<6} {\"Best\":>8}  {\"Step8\":>8}  {\"Delta\":>8}  Flag')
        rows = sorted(per_best.items(), key=lambda x: x[1]['mean_cer'], reverse=True)
        for lang, info in rows:
            cv = info['mean_cer']
            s8v = per_s8.get(lang, {}).get('mean_cer', float('nan'))
            delta = cv - s8v
            flag = '🔴' if cv > 0.25 else ('🟡' if cv > 0.12 else '🟢')
            print(f'  {lang:<6} {cv:>8.4f}  {s8v:>8.4f}  {delta:>+8.4f}  {flag}')
except Exception as e:
    print(f'(Per-language breakdown: {e})')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# APPEND NEW CELLS AND SAVE
# ─────────────────────────────────────────────────────────────────────────────

nb["cells"].extend(new_cells)

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {len(new_cells)} new cells → {DST}")
print(f"Total cells in notebook: {len(nb['cells'])}")
