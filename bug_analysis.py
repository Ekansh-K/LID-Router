# Critical Analysis: All Bugs Found

## BUG 1 — CRITICAL: `_inject_partners` in policy_rules.py cap formula is wrong
## Line 195: `del candidates[top_k + len(partners):]`
## `partners` here includes ALL cluster partners, not just newly injected ones.
## If top_k=3 and hin/urd cluster has 1 partner, but that partner was already in
## candidates, len(partners)=1 still fires, allowing 4 items instead of bounded list.
## The cap should be top_k + 2 (original intent) or dynamically computed.

## BUG 2 — CRITICAL: `multi_hypothesis.py` line 206: `orig_total` reference in alt-backend branch
## In the `else` block of `if use_script_score` (alt backend scoring, line 206-208),
## the variable `orig_total` is used but it is ONLY defined inside `if use_script_score` above
## at line 175. When `use_script_score=True`, `orig_total` was never set, so
## if Python reaches line 206 it raises NameError. Actually re-read - if use_script_score
## is True, we take the `if` branch (line 200-204). If use_script_score is False,
## orig_total WAS set at line 175. So the alt-backend else branch IS reachable only
## when use_script_score=False, and orig_total IS defined. Actually no bug here.
## Wait - re-check: orig_total is set at line 175 in the primary candidate scoring:
## `else: orig_total = w_decoder + w_plausibility + w_lid`
## This runs for the PRIMARY candidate. For the ALT backend (i==0 block), we check
## `if use_script_score` again at line 200. If use_script_score=True, takes if-branch OK.
## If use_script_score=False, takes else-branch and uses orig_total from line 175.
## orig_total IS defined because in the primary candidate loop, when use_script_score=False,
## orig_total was set at line 175. HOWEVER: if the primary candidate threw an exception
## and we're now in the alt-backend try block... no, alt-backend is inside the primary try.
## OK so orig_total is always set when use_script_score=False. No bug here.

## BUG 3 — CRITICAL: `_apply_temperature` uses p^(1/tau) NOT standard softmax temperature
## Standard softmax temperature: logit / tau, then softmax
## p^(1/tau) is equivalent only for uniform prior — in practice it's an approximation
## that works for probability distributions (not logits) and is called "sharpening/smoothing"
## It's mathematically WRONG for proper temperature scaling but FUNCTIONALLY acceptable
## as an approximation. However p^(1/tau) for p=0 gives 0^inf = 0 which is fine.
## The bigger issue: when tau=1.6 and top-prob is 0.97 (urd gets 0.97 confidence),
## 0.97^(1/1.6) = 0.97^0.625 ≈ 0.981 — barely changes anything!
## So for VERY high confidence (>0.95), temperature scaling does almost nothing.
## The force_mode_b_threshold=0.97 + temperature combo means:
## If urd raw prob = 0.90, tempered = 0.90^0.625 ≈ 0.937 < 0.97 → triggers override ✓
## If urd raw prob = 0.98, tempered = 0.98^0.625 ≈ 0.987 > 0.97 → Mode A NOT forced ✗
## This is a real logic gap: high-confidence urd samples (prob>0.975) would escape
## the Mode B override. Since urd LID is only 67% accurate, we ALWAYS want Mode B.

## BUG 4 — CRITICAL: force_mode_b_threshold compares TEMPERED prob, not RAW prob
## In policy_rules.py line 128: `top1_prob < force_b_threshold`
## where top1_prob = tempered_probs.get(t_top1, ...) — this is AFTER temperature scaling.
## Temperature scaling with tau=1.6 RAISES the post-temp prob (p^0.625 > p for 0<p<1).
## So we're comparing a HIGHER value against the threshold.
## For urd: if raw_prob=0.90, tempered=0.937, threshold=0.97 → 0.937 < 0.97 → forced ✓
## For urd: if raw_prob=0.99, tempered=0.994, threshold=0.97 → NOT forced → MODE A ✗
## The intent is to use the RAW prob, not the tempered one, for the threshold check.

## BUG 5 — CRITICAL: `_script_ratio` docstring says returns 1.0 for no-keywords,
## but the actual use case for "no expectation" is wrong neutral value
## Line 75-76: `if not keywords: return 1.0` — this makes every language without
## a configured script get score=1.0, which means it would BEAT a correct-script
## hypothesis that has score<1.0. Should return 0.5 (neutral) consistently.

## BUG 6 — MODERATE: `multi_hypothesis.py` loads_yaml import is unused
## Line 24: `from src.utils import get_logger, TranscriptResult, load_yaml`
## `load_yaml` is never used inside multi_hypothesis.py

## BUG 7 — MODERATE: `UncertaintySignals.to_vector_extended()` docstring says "15-dim"
## but actually returns 10-dim (6 base + 4 F0). The docstring is wrong.
## Also the comment on line 167: "5 top-prob slots (filled by policy) + 4 F0 = 9 extra dims"
## contradicts "10-dim" total. The method only concatenates 6+4=10 dims.

## BUG 8 — MODERATE: In `policy_learned.py` decide(), the MLP feature vector at line 180
## uses `[:self.input_dim]` which clips. For the Phase 3 retrain (input_dim=15),
## the 15-dim input is: [6 uncertainty + 5 probs + 4 F0]. But the F0 features are
## stored in uncertainty.f0_features — they are NOT included in the feature_vec built
## at line 178-180. So when calling the base policy's decide(), Phase 3 F0 data is
## IGNORED. The F0Pipeline in the notebook builds the 15-dim vector manually, which
## is correct, but the base LearnedRoutingPolicy.decide() can't use F0 at all.
## This is intentional design (Phase 3 uses F0Pipeline subclass) but not documented.

## BUG 9 — MODERATE: In `policy_rules.py` _inject_partners(), line 191-192:
## `p in tempered_probs` — this checks tempered_probs, not fused_probs.
## Since _apply_temperature drops keys where v==0 (line 53: `if v > 0`),
## a confusion partner with prob=0 in fused_probs won't be in tempered_probs either.
## But a partner that exists in fused_probs with tiny prob (e.g. 0.001) WILL be in
## tempered_probs. This is fine — but it's inconsistent with the original code
## which checked fused_probs directly. Low-prob partners get temperature-scaled too.

## BUG 10 — MODERATE: `multi_hypothesis.py` _confusion_map singleton is module-level.
## If the module is reloaded (which Step 9 setup does), the singleton is NOT reset
## because `_confusion_map = None` only runs once at module load. But on reload,
## it re-runs, setting _confusion_map=None again, which is correct.
## Actually importlib.reload() re-executes the module top level, so _confusion_map=None 
## gets re-set. On the NEXT call to decode_multi_hypothesis(), _get_confusion_map()
## creates a new ConfusionMap. This is fine BUT the new ConfusionMap is a DIFFERENT
## instance than the one held by RoutingAgent. They share the same YAML but are
## separate objects. This means if someone passed a custom ConfusionMap to RoutingAgent,
## multi_hypothesis wouldn't use it. This is a design issue but not a runtime bug.

## SUMMARY:
## Critical fixes needed: BUG 3+4 (temperature + threshold logic), BUG 5 (script neutral)
## Moderate fixes: BUG 1 (cap formula), BUG 6 (unused import), BUG 7 (docstring)
print("Analysis complete")
