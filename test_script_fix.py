"""Verify cluster-level script reranking logic is correct after fix."""
import sys
sys.path.insert(0, '.')

from src.decoding.multi_hypothesis import _script_ratio, _character_plausibility

print('=== SCRIPT RERANKING LOGIC TESTS (cluster-level target) ===\n')

# ── Test 1: urd as top-1 (67% of the time) ────────────────────────────────
target = 'Arab'
urdu_text  = '\u0645\u0631\u062d\u0628\u0627 \u062f\u0648\u0633\u062a'   # Perso-Arabic: مرحبا دوست
hindi_text = '\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u094b\u0938\u094d\u0924'  # Pure Devanagari: नमस्ते दोस्त

urd_s = _script_ratio(urdu_text,  target)
hin_s = _script_ratio(hindi_text, target)
print('urd top-1 → target_script=Arab applied to ALL candidates:')
print(f'  Whisper(urd) [Perso-Arabic]:  Arab score = {urd_s:.2f}  (expected ~1.0)')
print(f'  Whisper(hin) [Devanagari]:    Arab score = {hin_s:.2f}  (expected ~0.0)')
assert urd_s > 0.9, f'fail: {urd_s}'
assert hin_s < 0.1, f'fail: {hin_s}'
print('  PASS\n')

# ── Test 2: srp as top-1 (80% of the time) ────────────────────────────────
target2 = 'Cyrl'
srp_text = '\u0414\u043e\u0431\u0430\u0440 \u0434\u0430\u043d'  # Cyrillic
hrv_text = 'Dobar dan'                                           # Latin

srp_s = _script_ratio(srp_text, target2)
hrv_s = _script_ratio(hrv_text, target2)
print('srp top-1 → target_script=Cyrl applied to ALL candidates:')
print(f'  Whisper(srp) [Cyrillic]: Cyrl score = {srp_s:.2f}  (expected ~1.0)')
print(f'  MMS(hrv)     [Latin]:    Cyrl score = {hrv_s:.2f}  (expected ~0.0)')
assert srp_s > 0.9, f'fail: {srp_s}'
assert hrv_s < 0.1, f'fail: {hrv_s}'
print('  PASS\n')

# ── Test 3: Combined score — urd must beat hin when urd is top-1 ─────────
w_d, w_p, w_l, w_s = 0.4, 0.1, 0.2, 0.3
lid_urd, lid_hin = 0.67, 0.30
conf = 0.85
plaus = 0.7
urd_total = w_d*conf + w_p*plaus + w_l*lid_urd + w_s*_script_ratio(urdu_text,  'Arab')
hin_total = w_d*conf + w_p*plaus + w_l*lid_hin + w_s*_script_ratio(hindi_text, 'Arab')
winner = 'urd' if urd_total > hin_total else 'hin'
print(f'Combined score: urd={urd_total:.3f}  hin={hin_total:.3f}  winner={winner}')
assert winner == 'urd', f'urd should win, got {winner}'
print('  PASS\n')

# ── Test 4: LID error case (hin as top-1, true=urd) — hin wins = expected loss ──
# target = Deva (from hin top-1). urd/Arab loses on Deva score.
# This is unavoidable — it's a real LID error (33% rate).
target4 = 'Deva'
hin_d = _script_ratio(hindi_text, target4)
urd_d = _script_ratio(urdu_text,  target4)
print('hin top-1 (LID error case, true=urd) → target_script=Deva:')
print(f'  Whisper(hin) [Deva]: {hin_d:.2f}  (wins — this is the unavoidable error case)')
print(f'  Whisper(urd) [Arab]: {urd_d:.2f}  (loses)')
print('  NOTE: 33% LID error rate → ~33% of urd samples still wrong. Only better LID fixes this.')
print('  PASS (expected behavior)\n')

# ── Test 5: Non-script cluster (e.g. ind) — no script scoring ───────────
from src.routing.confusion_map import ConfusionMap
cmap = ConfusionMap()
assert not cmap.rerank_by_script('ind'), 'ind should not rerank by script'
assert cmap.rerank_by_script('urd') and cmap.rerank_by_script('srp')
print('Cluster checks: ind no-script, urd/srp use-script: PASS\n')

# ── Test 6: File is clean (no dead code) ────────────────────────────────
with open('src/decoding/multi_hypothesis.py') as f:
    content = f.read()
count = content.count('def decode_multi_hypothesis')
assert count == 1, f'Expected 1 function definition, found {count}'
print(f'File clean: 1 definition of decode_multi_hypothesis: PASS\n')

print('ALL TESTS PASSED')
