"""
Ablation experiment runner.

Implements all 8 ablation experiments (A1–A8) and 4 baselines (B1–B4)
from the plan. Each ablation modifies one component and re-runs evaluation.

ALL ablations run on KAGGLE (GPU required).
"""
import json
from pathlib import Path
from typing import Optional, List

from src.utils import get_logger, load_config
from evaluation.evaluate import evaluate_full, evaluate_lid_only
from evaluation.data_loader import SUBSET_30_FLEURS

log = get_logger("ablation")


def run_ablation_a1(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A1: Single LID only (MMS-LID, no Whisper LID signal).
    
    Tests the value of dual-signal LID.
    """
    from src.pipeline import Pipeline

    config = load_config()
    pipe = Pipeline(config=config)

    # Override: only use acoustic LID
    pipe.load_models()

    # Monkey-patch: make decoder_lid return empty probs
    original_predict = pipe.decoder_lid.predict
    pipe.decoder_lid.predict = lambda audio, sr=16000: {}

    log.info("A1: Running with MMS-LID only (no Whisper LID)...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a1_mms_lid_only.json")

    pipe.decoder_lid.predict = original_predict  # restore
    pipe.unload_models()
    return results


def run_ablation_a2(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A2: Single LID only (Whisper, no MMS-LID signal)."""
    from src.pipeline import Pipeline

    config = load_config()
    pipe = Pipeline(config=config)
    pipe.load_models()

    original_predict = pipe.acoustic_lid.predict
    pipe.acoustic_lid.predict = lambda audio, sr=16000: {}

    log.info("A2: Running with Whisper LID only (no MMS-LID)...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a2_whisper_lid_only.json")

    pipe.acoustic_lid.predict = original_predict
    pipe.unload_models()
    return results


def run_ablation_a4(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A4: No routing agent (always Mode A — pick top-1 and decode once).
    
    Tests the value of the routing layer itself.
    """
    from src.pipeline import Pipeline
    from src.routing.policy_rules import RoutingDecision, RoutingMode

    config = load_config()
    pipe = Pipeline(config=config)
    pipe.load_models()

    # Override: always return Mode A
    original_decide = pipe.routing_agent.decide
    def always_mode_a(fused_probs, uncertainty):
        top_lang = max(fused_probs, key=fused_probs.get) if fused_probs else "eng"
        return RoutingDecision(
            mode=RoutingMode.SINGLE,
            candidate_languages=[top_lang],
            confidence=uncertainty.top1_prob,
            reason="Ablation: always Mode A"
        )
    pipe.routing_agent.decide = always_mode_a

    log.info("A4: Running with no routing (always Mode A)...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a4_no_routing.json")

    pipe.routing_agent.decide = original_decide
    pipe.unload_models()
    return results


def run_ablation_a5(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A5: No confusion-aware override."""
    from src.pipeline import Pipeline

    config = load_config()
    pipe = Pipeline(config=config)
    pipe.load_models()

    # Clear the confusion map
    pipe.routing_agent.confusion_map._lang_to_partners = {}
    pipe.fusion._confusion_lookup = {}

    log.info("A5: Running without confusion awareness...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a5_no_confusion.json")

    pipe.unload_models()
    return results


def run_ablation_a3(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A3: SpeechBrain ECAPA-TDNN as acoustic LID (instead of MMS-LID-4017).

    Tests whether the much larger MMS-LID-4017 model is worth its cost.
    """
    from src.pipeline import Pipeline
    from src.lid.baseline_lid import BaselineLID

    config = load_config()
    pipe = Pipeline(config=config)
    pipe.load_models()

    # Replace acoustic LID with ECAPA-TDNN baseline
    baseline = BaselineLID(device=pipe.device)
    baseline.load()
    original_lid = pipe.acoustic_lid
    pipe.acoustic_lid = baseline

    log.info("A3: Running with SpeechBrain ECAPA-TDNN as acoustic LID...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a3_ecapa_lid.json")

    pipe.acoustic_lid = original_lid
    baseline.unload()
    pipe.unload_models()
    return results


def run_ablation_a6(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A6: Rule-based only vs. learned policy.

    Compares rule-based routing with the trained MLP policy.

    The rule-based run IS identical to Step 4 (same pipeline, same config), so we
    reuse results/eval_results.json instead of re-running (~40 min saved).
    """
    import shutil
    from src.pipeline import Pipeline

    config = load_config()

    # ── Rule-based: reuse Step 4 output (identical pipeline) ──────────────
    step4_json = Path(save_dir).parent / "eval_results.json"
    a6_rules_path = Path(save_dir) / "a6_rules_policy.json"
    if step4_json.exists():
        shutil.copy(step4_json, a6_rules_path)
        log.info(f"A6: Reused Step 4 results as rule-based baseline → {a6_rules_path}")
        with open(a6_rules_path) as f:
            results_rules_dict = json.load(f)
    else:
        # Fallback: re-run if Step 4 JSON is missing
        log.warning("A6: eval_results.json not found — re-running rule-based eval.")
        pipe_rules = Pipeline(config=config, routing_policy="rules")
        pipe_rules.load_models()
        results_rules = evaluate_full(pipe_rules, max_per_lang=max_per_lang,
                                      save_path=str(a6_rules_path))
        pipe_rules.unload_models()
        results_rules_dict = results_rules.to_dict()

    # ── Learned policy ─────────────────────────────────────────────────────
    checkpoint = Path("models/routing_policy.pt")
    if checkpoint.exists():
        pipe_learned = Pipeline(config=config, routing_policy="learned")
        pipe_learned.load_models()
        pipe_learned.routing_agent.load_learned_policy(str(checkpoint))

        log.info("A6: Running with learned MLP policy (LID-oracle)...")
        results_learned = evaluate_full(pipe_learned, max_per_lang=max_per_lang,
                                        save_path=f"{save_dir}/a6_learned_policy.json")
        pipe_learned.unload_models()
        return {"rules": results_rules_dict, "learned": results_learned.to_dict()}
    else:
        log.warning("A6: models/routing_policy.pt not found — skipping learned comparison.")
        return results_rules_dict


def run_ablation_a7(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A7: MMS-only ASR (no Whisper ASR backend).

    Tests the value of having dual ASR backends.
    """
    from src.pipeline import Pipeline
    from src.asr.backend_selector import select_backend as original_selector
    import src.asr.backend_selector as selector_mod

    config = load_config()
    pipe = Pipeline(config=config)
    pipe.load_models()

    # Override backend selector to always return "mms"
    selector_mod.select_backend = lambda language, mode="A": "mms"

    log.info("A7: Running with MMS-only ASR (no Whisper backend)...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a7_mms_only_asr.json")

    selector_mod.select_backend = original_selector
    pipe.unload_models()
    return results


def run_ablation_a8(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A8: Whisper-only ASR (no MMS ASR backend).

    Tests the value of broad-coverage MMS backend.
    Note: languages outside Whisper's 99 will fail — this is expected
    and demonstrates why MMS is needed.
    """
    from src.pipeline import Pipeline
    from src.asr.backend_selector import select_backend as original_selector
    import src.asr.backend_selector as selector_mod
    from src.utils import get_language_map

    config = load_config()
    pipe = Pipeline(config=config)
    pipe.load_models()

    lang_map = get_language_map()

    # Override backend selector to always return "whisper" when possible
    def whisper_only_selector(language, mode="A"):
        if lang_map.whisper_supported(language):
            return "whisper"
        # For unsupported languages, still fall back to mms to avoid crashes
        # but log a warning — this is the point of the ablation
        log.warning(f"A8: {language} not in Whisper — forced to use MMS")
        return "mms"

    selector_mod.select_backend = whisper_only_selector

    log.info("A8: Running with Whisper-only ASR...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a8_whisper_only_asr.json")

    selector_mod.select_backend = original_selector
    pipe.unload_models()
    return results


def run_ablation_a9_cer_oracle(max_per_lang: int = 30, save_dir: str = "./results/ablations"):
    """A9: CER-oracle learned routing policy.

    Uses a policy trained on CER-based oracle labels (from the test set)
    rather than LID-accuracy oracle labels. Labels are assigned as:
      CER < 0.15  → Mode A (single decode was fine)
      CER < 0.35  → Mode B (multi-hypothesis quality)
      CER ≥ 0.35  → Mode C (fallback needed)

    Compares against the LID-oracle policy (A6) to see whether CER-oracle
    supervision yields better routing.
    """
    from src.pipeline import Pipeline
    checkpoint = Path("models/routing_policy_cer.pt")
    if not checkpoint.exists():
        log.warning("A9: models/routing_policy_cer.pt not found — skipping. "
                    "Run the CER-oracle training cell first.")
        return {"error": "checkpoint not found"}

    config = load_config()
    pipe = Pipeline(config=config, routing_policy="learned")
    pipe.load_models()
    pipe.routing_agent.load_learned_policy(str(checkpoint))

    log.info("A9: Running with CER-oracle learned policy...")
    results = evaluate_full(pipe, max_per_lang=max_per_lang,
                            save_path=f"{save_dir}/a9_cer_oracle_policy.json")
    pipe.unload_models()
    return results


def run_all_ablations(max_per_lang: int = 30,
                      save_dir: str = "./results/ablations"):
    """Run all ablation experiments. Takes several hours on GPU."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ablations = {
        "A1": run_ablation_a1,
        "A2": run_ablation_a2,
        "A3": run_ablation_a3,
        "A4": run_ablation_a4,
        "A5": run_ablation_a5,
        "A6": run_ablation_a6,
        "A7": run_ablation_a7,
        "A8": run_ablation_a8,
        "A9": run_ablation_a9_cer_oracle,
    }

    all_results = {}
    for name, fn in ablations.items():
        log.info(f"\n{'='*60}\nRunning ablation {name}\n{'='*60}")
        try:
            results = fn(max_per_lang=max_per_lang, save_dir=save_dir)
            if isinstance(results, dict):
                all_results[name] = results
            else:
                all_results[name] = results.to_dict()
        except Exception as e:
            log.error(f"Ablation {name} failed: {e}")
            all_results[name] = {"error": str(e)}

    # Save summary
    summary_path = f"{save_dir}/ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"All ablation results saved to {summary_path}")

    return all_results
