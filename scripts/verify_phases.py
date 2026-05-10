"""Final phase verification script."""
import sys, pathlib, yaml, json, numpy as np
sys.path.insert(0, '.')

OK = 0
FAIL = 0

def chk(label, cond, detail=''):
    global OK, FAIL
    if cond:
        print(f'  OK  {label}')
        OK += 1
    else:
        print(f'  !! FAIL  {label}  {detail}')
        FAIL += 1

print('=== PHASE-BY-PHASE IMPLEMENTATION VERIFICATION ===\n')

# ── PHASE 0 ─────────────────────────────────────────────────────────────────
print('PHASE 0 — Metric Normalization (runs locally, no pipeline re-run)')
chk('results_analysis.ipynb exists', pathlib.Path('results_analysis.ipynb').exists())
print()

# ── PHASE 1 — Script-Aware Routing ──────────────────────────────────────────
print('PHASE 1 — Script-Aware Routing Fix')
with open('config/confusion_clusters.yaml') as f:
    cfg = yaml.safe_load(f)

urd_cl  = next(c for c in cfg['clusters'] if 'urd' in c['languages'])
srp_cl  = next(c for c in cfg['clusters'] if 'srp' in c['languages'])
ind_cl  = next(c for c in cfg['clusters'] if 'ind' in c['languages'])
script_exp = cfg.get('script_expectations', {})

chk('urd force_mode_b_threshold=0.97', urd_cl.get('force_mode_b_threshold') == 0.97)
chk('srp force_mode_b_threshold=0.92', srp_cl.get('force_mode_b_threshold') == 0.92)
chk('ind force_mode_b_threshold=0.80', ind_cl.get('force_mode_b_threshold') == 0.80)
chk('urd rerank_by_script=True',       urd_cl.get('rerank_by_script') is True)
chk('srp rerank_by_script=True',       srp_cl.get('rerank_by_script') is True)
chk('urd script expectation=Arab',     script_exp.get('urd') == 'Arab')
chk('srp script expectation=Cyrl',     script_exp.get('srp') == 'Cyrl')
chk('hin script expectation=Deva',     script_exp.get('hin') == 'Deva')

from src.routing.confusion_map import ConfusionMap
from src.routing.policy_rules import RuleBasedPolicy, RoutingMode, _apply_temperature
from src.utils import UncertaintySignals
from src.decoding.multi_hypothesis import _script_ratio

cmap = ConfusionMap()
policy = RuleBasedPolicy(confusion_map=cmap)

# urd at ANY confidence below 0.97 AND above 0.97 → always Mode B
d_lo = policy.decide({'urd':0.80,'hin':0.15,'eng':0.05},
                     UncertaintySignals(top1_prob=0.80, gap=0.65, entropy=0.8))
d_hi = policy.decide({'urd':0.99,'hin':0.01},
                     UncertaintySignals(top1_prob=0.99, gap=0.98, entropy=0.1))

chk('urd 0.80 -> Mode B (forced)',  d_lo.mode == 'B', f'got {d_lo.mode}')
chk('urd 0.99 -> Mode B (forced)',  d_hi.mode == 'B', f'got {d_hi.mode}')
chk('hin injected when urd top-1',  'hin' in d_lo.candidate_languages, str(d_lo.candidate_languages))

eng_d = policy.decide({'eng':0.95,'fra':0.03,'deu':0.02},
                      UncertaintySignals(top1_prob=0.95, gap=0.92, entropy=0.3))
chk('eng 0.95 stays Mode A (not in cluster)', eng_d.mode == 'A', f'got {eng_d.mode}')

chk('Arabic text Arab score > 0.9',  _script_ratio('\u0645\u0631\u062d\u0628\u0627', 'Arab') > 0.9)
chk('Latin text Arab score < 0.1',   _script_ratio('hello world', 'Arab') < 0.1)
chk('Empty text neutral 0.5',        _script_ratio('', 'Arab') == 0.5)
chk('Unknown script neutral 0.5',    _script_ratio('x', 'NOTEXIST') == 0.5)
print()

# ── PHASE 2 — Temperature Scaling ───────────────────────────────────────────
print('PHASE 2 — Softmax Temperature Scaling')
chk('urd tau=1.6',  cmap.temperature('urd') == 1.6)
chk('srp tau=1.4',  cmap.temperature('srp') == 1.4)
chk('ind tau=1.2',  cmap.temperature('ind') == 1.2)
chk('eng tau=1.0 (no cluster)', cmap.temperature('eng') == 1.0)

probs = {'urd':0.70,'hin':0.25,'eng':0.05}
t = _apply_temperature(probs, 1.6)
orig_gap = 0.70 - 0.25
s = sorted(t.values(), reverse=True)
new_gap = s[0] - s[1]
chk(f'Temperature softens gap {orig_gap:.3f}->{new_gap:.3f}', new_gap < orig_gap)

# flat-gap guard: urd=0.55, hin=0.45 -> gap=0.10 which is exactly the threshold
# After temperature tau=1.6, gap becomes even smaller -> triggers flat-gap guard
flat_d = policy.decide({'urd':0.55,'hin':0.45},
                       UncertaintySignals(top1_prob=0.55, gap=0.10, entropy=1.8))
chk('Flat-gap guard: urd=0.55/hin=0.45 -> Mode B', flat_d.mode == 'B', f'got {flat_d.mode}')
print()

# ── PHASE 3 — F0 Features ───────────────────────────────────────────────────
print('PHASE 3 — F0 Pitch Features')
from src.lid.f0_features import extract_f0_features, is_available, dummy_features

f0 = extract_f0_features(np.zeros(16000, dtype='float32'), 16000)
chk('f0_features returns shape (4,)',     f0.shape == (4,), str(f0.shape))
chk('dummy_features returns shape (4,)',  dummy_features().shape == (4,))

us = UncertaintySignals()
us.f0_features = np.array([150.0, 30.0, 80.0, 0.7], dtype='float32')
v_ext = us.to_vector_extended()
chk('to_vector_extended() returns (10,)', v_ext.shape == (10,), str(v_ext.shape))

v_base = us.to_vector()
chk('to_vector() still returns (6,)',     v_base.shape == (6,), str(v_base.shape))

from src.routing.policy_learned import LearnedRoutingPolicy
p15 = LearnedRoutingPolicy(input_dim=15)
chk('LearnedRoutingPolicy input_dim=15 for Phase 3 MLP', p15.input_dim == 15)
print()

# ── NOTEBOOK ────────────────────────────────────────────────────────────────
print('NOTEBOOK — Kaggle_Notebook_v2.ipynb')
with open('Kaggle_Notebook_v2.ipynb') as f:
    nb = json.load(f)
cell_ids = [c['id'] for c in nb['cells']]
needed = [
    ('step9_setup',  'Step 9 Setup: reload modules'),
    ('step9a_code',  'Step 9a: Rules+Phase1+2 evaluation'),
    ('step9b_code',  'Step 9b: Learned+Phase1+2 evaluation'),
    ('step9c_code',  'Step 9c: Comparison table'),
    ('step10_setup', 'Step 10 Setup: install pyworld'),
    ('step10a_code', 'Step 10a: collect F0 training data'),
    ('step10b_code', 'Step 10b: retrain MLP input_dim=15'),
    ('step10c_code', 'Step 10c: F0 pipeline evaluation'),
    ('step11_code',  'Step 11: grand comparison'),
]
for cid, desc in needed:
    chk(f'{cid} ({desc})', cid in cell_ids)
chk(f'Total cells = 45', len(nb['cells']) == 45, f'actual={len(nb["cells"])}')
print()

print(f'======================================')
print(f'TOTAL: {OK} OK, {FAIL} FAILED')
if FAIL == 0:
    print('ALL PHASES VERIFIED — ready to run in Kaggle')
else:
    sys.exit(1)
