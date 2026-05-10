# ── Step 9-Quick: Validate script-reranking fix on srp + urd only ───────────
# Runs in ~5 minutes instead of 40. Tests ONLY the two failing languages.
import inspect
from src.decoding import multi_hypothesis as _mh
from src.pipeline import Pipeline
from evaluation.evaluate import evaluate_full

# 1. Confirm the fix is active
src_code = inspect.getsource(_mh.decode_multi_hypothesis)
if 'top1_lang = candidate_languages[0]' in src_code:
    print('✅ FIX ACTIVE: cluster-level script reranking loaded')
else:
    raise RuntimeError('❌ OLD code still loaded — re-run the patch cell first!')

# 2. Run only srp + urd + hin (hin is the sanity check — should stay ~0.14)
TARGET_LANGS = ['srp', 'urd', 'hin']

pipe_q = Pipeline(routing_policy='rules')
pipe_q.load_models()

results_q = evaluate_full(
    pipeline=pipe_q,
    lang_codes=TARGET_LANGS,
    split='test',
    max_per_lang=30,
    save_path='results/step9_quick_srp_urd.json',
)

pipe_q.unload_models()

import json
with open('results/step9_quick_srp_urd.json') as f:
    dq = json.load(f)
per = dq.get('per_language', {})

print()
print('=' * 58)
print(f'  {"Lang":<6}  {"New CER":>8}  {"Old CER":>8}  {"Delta":>8}  Result')
print('=' * 58)

OLD = {'srp': 0.8228, 'urd': 0.3513, 'hin': 0.1478}
for lang in TARGET_LANGS:
    if lang not in per:
        continue
    cer   = per[lang]['mean_cer']
    prev  = OLD[lang]
    delta = cer - prev
    if   delta < -0.05: status = 'IMPROVED ✅'
    elif delta < 0.01:  status = 'OK  ✅'
    else:               status = 'WORSE  ❌'
    print(f'  {lang:<6}  {cer:>8.4f}  {prev:>8.4f}  {delta:>+8.4f}  {status}')

print('=' * 58)
print()

srp_cer = per.get('srp', {}).get('mean_cer', 1.0)
urd_cer = per.get('urd', {}).get('mean_cer', 1.0)

if srp_cer < 0.50:
    print(f'srp ({srp_cer:.4f}): PASS — Cyrillic reranking is working')
else:
    print(f'srp ({srp_cer:.4f}): FAIL — still catastrophic, check script_target in logs')

if urd_cer < 0.30:
    print(f'urd ({urd_cer:.4f}): PASS — Arabic reranking is working')
else:
    print(f'urd ({urd_cer:.4f}): FAIL — still high, reranking not effective')
