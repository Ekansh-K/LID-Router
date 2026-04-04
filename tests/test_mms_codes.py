"""
Fast local test: verifies all language_map.yaml mms_asr codes are valid
by calling set_target_lang() on the MMS tokenizer (no audio, no inference).

Run on Kaggle after Step 1 (models downloaded):
    pytest tests/test_mms_codes.py -v

Or run standalone:
    python tests/test_mms_codes.py
"""
import pytest


def get_all_mms_asr_codes():
    """Return list of (canonical, mms_asr_code) from language_map.yaml."""
    from src.utils import get_language_map, load_yaml
    from pathlib import Path
    raw = load_yaml(Path(__file__).parent.parent / "config" / "language_map.yaml")
    return [(lang, entry["mms_asr"]) for lang, entry in raw.items()
            if isinstance(entry, dict) and "mms_asr" in entry]


@pytest.mark.parametrize("canonical,mms_code", get_all_mms_asr_codes())
def test_mms_asr_code_valid(canonical, mms_code):
    """Each mms_asr code must be accepted by the MMS tokenizer without error."""
    try:
        from transformers import AutoProcessor
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    except Exception:
        pytest.skip("MMS model not cached — run Step 1 first")

    try:
        proc.tokenizer.set_target_lang(mms_code)
    except ValueError as e:
        pytest.fail(
            f"Language '{canonical}' → mms_asr='{mms_code}' is INVALID: {e}"
        )


if __name__ == "__main__":
    # Standalone run without pytest — prints pass/fail for each language
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    except Exception as e:
        print(f"Cannot load MMS processor: {e}")
        print("Run this on Kaggle after Step 1 (model download).")
        raise SystemExit(1)

    codes = get_all_mms_asr_codes()
    print(f"Testing {len(codes)} mms_asr codes...\n")
    passed, failed = [], []

    for canonical, mms_code in codes:
        try:
            proc.tokenizer.set_target_lang(mms_code)
            print(f"  PASS  {canonical:<10} -> {mms_code}")
            passed.append(canonical)
        except ValueError as e:
            print(f"  FAIL  {canonical:<10} -> {mms_code}  ({e})")
            failed.append((canonical, mms_code))

    print(f"\n{'='*60}")
    print(f"Passed: {len(passed)}/{len(codes)}")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for lang, code in failed:
            print(f"  {lang} -> {code}")
    else:
        print("All MMS-ASR codes are valid!")
