"""
Data loader for FLEURS evaluation dataset.

FLEURS provides audio + transcription + language labels for 102 languages.
We use the HuggingFace `datasets` library with streaming to avoid downloading
everything upfront.

Runs on CPU — safe for local machine (download happens on first access).
"""
from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np

from src.utils import get_logger, get_language_map, load_yaml
from pathlib import Path

log = get_logger("data_loader")

# 30-language development subset (plan Section 5.3, with hi_in duplicate fixed)
SUBSET_30_FLEURS = [
    # Group A: High-Resource
    "en_us", "cmn_hans_cn", "ar_eg", "hi_in", "es_419", "fr_fr", "de_de", "ja_jp",
    # Group B: Confusion Pairs (hin already in A, no duplicate)
    "ur_pk", "sr_rs", "hr_hr", "id_id", "ms_my",
    "nb_no", "da_dk", "cs_cz", "sk_sk", "pt_br",
    # Group C: Low-Resource
    "yo_ng", "sw_ke", "am_et", "cy_gb", "jv_id",
    "ka_ge", "mn_mn", "ne_np", "lo_la", "ha_ng",
]


def _datasets_version():
    """Return (major, minor) tuple for the installed datasets library."""
    try:
        import importlib.metadata
        ver = importlib.metadata.version("datasets")
        parts = ver.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


def load_fleurs(lang_codes: Optional[List[str]] = None,
                split: str = "test",
                streaming: bool = True,
                max_samples_per_lang: Optional[int] = None
                ) -> Dict[str, any]:
    """Load FLEURS dataset for specified languages.
    
    Requires datasets < 4.0.0 (google/fleurs uses a loading script).
    On Kaggle, run: !pip install 'datasets==2.20.0' --quiet  then restart kernel.

    Args:
        lang_codes: list of FLEURS config names (e.g., ["en_us", "fr_fr"]).
                   If None, uses the 30-language subset.
        split: "train", "validation", or "test"
        streaming: if True, use streaming mode (no full download)
        max_samples_per_lang: limit samples per language (for quick testing)
    
    Returns:
        dict mapping fleurs_code → dataset iterator
    """
    from datasets import load_dataset, Audio

    ver = _datasets_version()
    if ver >= (4, 0):
        raise RuntimeError(
            f"datasets {'.'.join(str(x) for x in ver)} no longer supports "
            "script-based datasets like google/fleurs.\n"
            "Fix: restart your Kaggle kernel after running:\n"
            "  !pip install 'datasets==2.20.0' --quiet"
        )

    if lang_codes is None:
        lang_codes = SUBSET_30_FLEURS

    datasets = {}
    for lang in lang_codes:
        try:
            ds = load_dataset(
                "google/fleurs", lang,
                split=split,
                streaming=streaming,
                trust_remote_code=True,   # required for script-based FLEURS in datasets 2.x
            )
            # Ensure audio is 16kHz
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            if max_samples_per_lang and not streaming:
                ds = ds.select(range(min(len(ds), max_samples_per_lang)))
            datasets[lang] = ds
            log.info(f"Loaded FLEURS {lang} ({split})")
        except Exception as e:
            log.warning(f"Failed to load FLEURS {lang}: {e}")

    log.info(f"Loaded {len(datasets)}/{len(lang_codes)} FLEURS languages")
    return datasets


def iterate_fleurs(datasets: Dict[str, any],
                   max_per_lang: Optional[int] = None
                   ) -> Iterator[Tuple[np.ndarray, int, str, str, str]]:
    """Iterate over loaded FLEURS datasets yielding individual samples.
    
    Yields:
        (audio_array, sample_rate, fleurs_lang_code, canonical_lang_code, reference_text)
    """
    lang_map = get_language_map()

    for fleurs_code, ds in datasets.items():
        canon = lang_map.from_fleurs(fleurs_code)
        if not canon:
            log.warning(f"No canonical code for FLEURS '{fleurs_code}' — skipping")
            continue

        count = 0
        for sample in ds:
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            # FLEURS has 'transcription' or 'raw_transcription'
            ref_text = sample.get("transcription", sample.get("raw_transcription", ""))

            yield audio.astype(np.float32), sr, fleurs_code, canon, ref_text

            count += 1
            if max_per_lang and count >= max_per_lang:
                break


def get_fleurs_lang_list(subset: str = "30") -> List[str]:
    """Get list of FLEURS config names for a predefined subset."""
    if subset == "30":
        return SUBSET_30_FLEURS
    elif subset == "all":
        return ALL_FLEURS_LANGS
    else:
        raise ValueError(f"Unknown subset: {subset}")


# Full FLEURS language list (102 languages)
ALL_FLEURS_LANGS = [
    "af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az",
    "be_by", "bg_bg", "bn_in", "bs_ba", "ca_es", "ceb_ph",
    "ckb_iq", "cmn_hans_cn", "cs_cz", "cy_gb", "da_dk", "de_de",
    "el_gr", "en_us", "es_419", "et_ee", "eu_es", "fa_ir",
    "ff_sn", "fi_fi", "fil_ph", "fr_fr", "ga_ie", "gl_es",
    "gu_in", "ha_ng", "he_il", "hi_in", "hr_hr", "hu_hu",
    "hy_am", "id_id", "ig_ng", "is_is", "it_it", "ja_jp",
    "jv_id", "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh",
    "kn_in", "ko_kr", "ky_kg", "lb_lu", "lg_ug", "ln_cd",
    "lo_la", "lt_lt", "luo_ke", "lv_lv", "mi_nz", "mk_mk",
    "ml_in", "mn_mn", "mr_in", "ms_my", "mt_mt", "my_mm",
    "nb_no", "ne_np", "nl_nl", "nn_no", "ny_mw", "oc_fr",
    "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br",
    "ro_ro", "ru_ru", "sd_in", "sk_sk", "sl_si", "sn_zw",
    "so_so", "sr_rs", "su_id", "sv_se", "sw_ke", "ta_in",
    "te_in", "tg_tj", "th_th", "tr_tr", "uk_ua", "umb_ao",
    "ur_pk", "uz_uz", "vi_vn", "wo_sn", "xh_za", "yo_ng",
    "yue_hant_hk", "zu_za",
]
