"""
Setup Verification Script
==========================
Run this to verify your environment is correctly configured.

Usage:
    python setup_verify.py          # check everything
    python setup_verify.py --local  # check local-only deps (no GPU)
    python setup_verify.py --gpu    # check GPU + model availability
"""
import sys
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def check_python_version():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 9, f"Need Python 3.9+, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


def check_import(module_name):
    mod = importlib.import_module(module_name)
    version = getattr(mod, "__version__", "installed")
    return version


def check_project_structure():
    required = [
        "config/model_config.yaml",
        "config/language_map.yaml",
        "config/confusion_clusters.yaml",
        "src/__init__.py",
        "src/utils.py",
        "src/preprocessing.py",
        "src/lid/acoustic_lid.py",
        "src/lid/decoder_lid.py",
        "src/lid/fusion.py",
        "src/routing/agent.py",
        "src/routing/policy_rules.py",
        "src/routing/policy_learned.py",
        "src/routing/confusion_map.py",
        "src/asr/whisper_backend.py",
        "src/asr/mms_backend.py",
        "src/asr/backend_selector.py",
        "src/decoding/single_decode.py",
        "src/decoding/multi_hypothesis.py",
        "src/decoding/fallback_decode.py",
        "src/pipeline.py",
        "evaluation/data_loader.py",
        "evaluation/metrics.py",
        "evaluation/evaluate.py",
        "evaluation/ablation.py",
        "evaluation/dashboard.py",
    ]
    missing = [f for f in required if not (PROJECT_ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}")
    return f"{len(required)} files OK"


def check_language_map():
    from src.utils import get_language_map
    lang_map = get_language_map()
    langs = lang_map.all_canonical()
    assert len(langs) >= 28, f"Expected >=28 languages, got {len(langs)}"
    # Verify a few critical mappings
    assert lang_map.to_whisper("eng") == "en"
    assert lang_map.from_whisper("en") == "eng"
    assert lang_map.to_mms_lid("fra") == "fra"
    return f"{len(langs)} languages mapped"


def check_confusion_map():
    from src.routing.confusion_map import ConfusionMap
    cmap = ConfusionMap()
    assert cmap.is_confused("hin")
    assert "urd" in cmap.get_partners("hin")
    return f"{len(cmap.all_cluster_ids())} clusters loaded"


def check_gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    return f"{dev} ({mem:.1f} GB)"


def check_config():
    from src.utils import load_config
    cfg = load_config()
    assert "lid" in cfg
    assert "routing" in cfg
    assert "asr" in cfg
    return "model_config.yaml valid"


def check_fusion_logic():
    from src.lid.fusion import LIDFusion
    fusion = LIDFusion(alpha=0.6)
    acoustic = {"eng": 0.8, "fra": 0.2}
    decoder = {"eng": 0.9, "fra": 0.1}
    fused = fusion.fuse(acoustic, decoder)
    assert max(fused, key=fused.get) == "eng"
    assert abs(sum(fused.values()) - 1.0) < 0.01
    return "fusion logic OK"


def check_routing_logic():
    from src.routing.policy_rules import RuleBasedPolicy, RoutingMode
    from src.utils import UncertaintySignals
    policy = RuleBasedPolicy()
    u = UncertaintySignals(top1_prob=0.95, gap=0.9, entropy=0.2,
                           top3_concentration=1.0)
    d = policy.decide({"eng": 0.95, "fra": 0.05}, u)
    assert d.mode == RoutingMode.SINGLE
    return "routing logic OK"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"

    print("=" * 60)
    print("Agentic LID+ASR Pipeline — Setup Verification")
    print("=" * 60)

    results = []

    # Always check these (local-safe)
    print("\n--- Core Environment ---")
    results.append(check("Python version", check_python_version))
    results.append(check("PyYAML", lambda: check_import("yaml")))
    results.append(check("NumPy", lambda: check_import("numpy")))
    results.append(check("tqdm", lambda: check_import("tqdm")))

    print("\n--- ML Libraries ---")
    results.append(check("PyTorch", lambda: check_import("torch")))
    results.append(check("torchaudio", lambda: check_import("torchaudio")))
    results.append(check("transformers", lambda: check_import("transformers")))
    results.append(check("datasets", lambda: check_import("datasets")))
    results.append(check("jiwer", lambda: check_import("jiwer")))

    if mode != "--local":
        results.append(check("Whisper (openai-whisper)", lambda: check_import("whisper")))
        results.append(check("SpeechBrain", lambda: check_import("speechbrain")))

    print("\n--- Visualization ---")
    results.append(check("matplotlib", lambda: check_import("matplotlib")))
    results.append(check("seaborn", lambda: check_import("seaborn")))

    print("\n--- Project Structure ---")
    results.append(check("File structure", check_project_structure))
    results.append(check("Config loading", check_config))
    results.append(check("Language map", check_language_map))
    results.append(check("Confusion map", check_confusion_map))

    print("\n--- Logic Verification ---")
    results.append(check("Fusion logic", check_fusion_logic))
    results.append(check("Routing logic", check_routing_logic))

    if mode in ("--gpu", "--all"):
        print("\n--- GPU ---")
        results.append(check("CUDA GPU", check_gpu))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed! Environment is ready.")
    else:
        print(f"{total - passed} check(s) failed. Fix these before proceeding.")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
