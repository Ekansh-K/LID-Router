"""Integrity check for all Phase 1+2+3 changes."""
import sys, numpy as np
sys.path.insert(0, '.')

PASS = 0
FAIL = 0

def check(label, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  OK   {label}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        FAIL += 1

# 1 ConfusionMap
def t1():
    from src.routing.confusion_map import ConfusionMap
    c = ConfusionMap()
    assert c.force_mode_b_threshold("urd") == 0.97, "urd threshold"
    assert c.force_mode_b_threshold("srp") == 0.92, "srp threshold"
    assert c.force_mode_b_threshold("eng") == 0.0,  "eng not in cluster"
    assert c.temperature("urd") == 1.6, "urd temperature"
    assert c.rerank_by_script("urd") == True, "urd rerank"
    assert c.expected_script("urd") == "Arab", "urd script"
    assert c.expected_script("srp") == "Cyrl", "srp script"
    assert c.expected_script("hin") == "Deva", "hin script"
check("ConfusionMap new fields", t1)

# 2 Temperature scaling
def t2():
    from src.routing.policy_rules import _apply_temperature
    probs = {"urd": 0.5, "hin": 0.4, "eng": 0.1}
    scaled = _apply_temperature(probs, 1.6)
    assert abs(sum(scaled.values()) - 1.0) < 1e-5, "sum != 1"
    assert scaled["urd"] < 0.5, "tau>1 should lower top prob"
check("_apply_temperature", t2)

# 3 RuleBasedPolicy accepts flat_gap_override
def t3():
    from src.routing.policy_rules import RuleBasedPolicy
    from src.routing.confusion_map import ConfusionMap
    p = RuleBasedPolicy(flat_gap_override=0.10, confusion_map=ConfusionMap())
    assert p.flat_gap_override == 0.10
check("RuleBasedPolicy flat_gap_override", t3)

# 4 LearnedRoutingPolicy accepts confusion_map
def t4():
    from src.routing.policy_learned import LearnedRoutingPolicy, _inject_partners_learned
    from src.routing.confusion_map import ConfusionMap
    p = LearnedRoutingPolicy(input_dim=11, confusion_map=ConfusionMap())
    assert p._flat_gap_override == 0.10
check("LearnedRoutingPolicy confusion_map", t4)

# 5 RoutingAgent wires confusion_map to learned policy
def t5():
    from src.routing.agent import RoutingAgent
    a = RoutingAgent(policy="rules")
    assert a.confusion_map.force_mode_b_threshold("urd") == 0.97
check("RoutingAgent shared confusion_map", t5)

# 6 UncertaintySignals extensions
def t6():
    from src.utils import UncertaintySignals
    us = UncertaintySignals(top1_prob=0.9, gap=0.3, entropy=1.2, top3_concentration=0.95)
    v6 = us.to_vector()
    assert v6.shape == (6,), f"expected 6-dim, got {v6.shape}"
    us.f0_features = np.array([150.0, 30.0, 80.0, 0.7], dtype="float32")
    v10 = us.to_vector_extended()
    assert v10.shape == (10,), f"expected 10-dim, got {v10.shape}"
check("UncertaintySignals f0_features + to_vector_extended", t6)

# 7 F0 extractor
def t7():
    from src.lid.f0_features import extract_f0_features, dummy_features
    silence = np.zeros(16000, dtype="float32")
    feats = extract_f0_features(silence, 16000)
    assert feats.shape == (4,), f"bad shape {feats.shape}"
    d = dummy_features()
    assert d.shape == (4,)
check("f0_features module", t7)

# 8 Script ratio
def t8():
    from src.decoding.multi_hypothesis import _script_ratio
    assert _script_ratio("", "Arab") == 0.5, "empty neutral"
    arabic = "\u0645\u0631\u062d\u0628\u0627\u0628\u0627\u0644\u0639\u0627\u0644\u0645"  # مرحبا بالعالم
    assert _script_ratio(arabic, "Arab") > 0.8, "Arabic text in Arab script"
    assert _script_ratio("hello world", "Arab") < 0.1, "Latin text in Arab script"
    cyrillic = "\u041f\u0440\u0438\u0432\u0435\u0442"  # Привет
    assert _script_ratio(cyrillic, "Cyrl") > 0.8, "Cyrillic in Cyrl"
    devanagari = "\u0928\u092e\u0938\u094d\u0924\u0947"  # नमस्ते
    assert _script_ratio(devanagari, "Deva") > 0.8, "Devanagari in Deva"
check("_script_ratio (unicode script detection)", t8)

# 9 multi_hypothesis imports cleanly
def t9():
    from src.decoding.multi_hypothesis import decode_multi_hypothesis, _character_plausibility
    assert _character_plausibility("") == 0.0
    assert _character_plausibility("hello world test sentence") > 0.5
check("multi_hypothesis imports and helpers", t9)

# 10 Notebook exists
def t10():
    import pathlib
    nb = pathlib.Path("Kaggle_Notebook_v2.ipynb")
    assert nb.exists(), "Kaggle_Notebook_v2.ipynb not found"
    import json
    with open(nb) as f:
        d = json.load(f)
    cell_ids = [c["id"] for c in d["cells"]]
    assert "step9_setup" in cell_ids, "step9_setup cell missing"
    assert "step9b_code" in cell_ids, "step9b_code cell missing"
    assert "step10_setup" in cell_ids, "step10_setup cell missing"
    assert "step10c_code" in cell_ids, "step10c_code cell missing"
    assert "step11_code" in cell_ids, "step11_code cell missing"
    assert len(d["cells"]) == 45, f"expected 45 cells, got {len(d['cells'])}"
check("Kaggle_Notebook_v2.ipynb completeness", t10)

print()
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL CHECKS PASSED - ready to upload to Kaggle")
else:
    print("SOME CHECKS FAILED - see above")
    sys.exit(1)
