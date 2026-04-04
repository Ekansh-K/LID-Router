"""
Main evaluation runner — runs the full pipeline on FLEURS test set
and computes all metrics.

This is GPU-heavy: run on KAGGLE, not locally.
"""
import json
import time
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from src.utils import get_logger, load_config, PipelineOutput
from src.pipeline import Pipeline
from evaluation.data_loader import load_fleurs, iterate_fleurs, SUBSET_30_FLEURS
from evaluation.metrics import (
    EvaluationResults, LIDResult, ASRResult,
    compute_cer, compute_wer,
)

log = get_logger("evaluate")


def evaluate_lid_only(pipeline: Pipeline,
                      lang_codes: Optional[List[str]] = None,
                      split: str = "validation",
                      max_per_lang: int = 50,
                      ) -> EvaluationResults:
    """Evaluate just the LID pipeline (no ASR). Faster for LID tuning.
    
    Args:
        pipeline: loaded Pipeline instance
        lang_codes: FLEURS language codes to evaluate
        split: dataset split
        max_per_lang: max samples per language (for speed)
    """
    if lang_codes is None:
        lang_codes = SUBSET_30_FLEURS

    datasets = load_fleurs(lang_codes, split=split, streaming=True)
    results = EvaluationResults()

    for audio, sr, fleurs_code, true_lang, ref_text in tqdm(
            iterate_fleurs(datasets, max_per_lang=max_per_lang),
            desc="LID Evaluation"):

        lid_output = pipeline.run_lid_only(audio, sr)
        fused = lid_output["fused_probs"]
        predicted = lid_output["top_language"]
        uncertainty = lid_output["uncertainty"]
        decision = lid_output["routing_decision"]

        sorted_langs = sorted(fused, key=fused.get, reverse=True)
        in_top3 = true_lang in sorted_langs[:3]

        results.lid_results.append(LIDResult(
            true_lang=true_lang,
            predicted_lang=predicted,
            top1_prob=uncertainty.top1_prob,
            correct=(predicted == true_lang),
            in_top3=in_top3,
            routing_mode=decision.mode,
        ))

    return results


def evaluate_full(pipeline: Pipeline,
                  lang_codes: Optional[List[str]] = None,
                  split: str = "test",
                  max_per_lang: int = 30,
                  save_path: Optional[str] = None,
                  ) -> EvaluationResults:
    """Full pipeline evaluation: LID + routing + ASR.
    
    Args:
        pipeline: loaded Pipeline instance (all models loaded)
        lang_codes: FLEURS codes
        split: dataset split
        max_per_lang: max samples per language
        save_path: if set, save results as JSON
    
    Returns:
        EvaluationResults with all metrics
    """
    if lang_codes is None:
        lang_codes = SUBSET_30_FLEURS

    datasets = load_fleurs(lang_codes, split=split, streaming=True)
    results = EvaluationResults()
    t0 = time.time()

    for audio, sr, fleurs_code, true_lang, ref_text in tqdm(
            iterate_fleurs(datasets, max_per_lang=max_per_lang),
            desc="Full Evaluation"):

        try:
            output: PipelineOutput = pipeline.run(audio, sr)

            # LID result
            sorted_fused = sorted(output.lid_distribution,
                                  key=output.lid_distribution.get, reverse=True)
            in_top3 = true_lang in sorted_fused[:3]

            results.lid_results.append(LIDResult(
                true_lang=true_lang,
                predicted_lang=output.detected_language,
                top1_prob=output.confidence,
                correct=(output.detected_language == true_lang),
                in_top3=in_top3,
                routing_mode=output.routing_mode,
            ))

            # ASR result
            cer_val = compute_cer(ref_text, output.transcript)
            wer_val = compute_wer(ref_text, output.transcript)

            results.asr_results.append(ASRResult(
                true_lang=true_lang,
                predicted_lang=output.detected_language,
                reference_text=ref_text,
                hypothesis_text=output.transcript,
                cer=cer_val,
                wer=wer_val,
                routing_mode=output.routing_mode,
                backend=output.backend_used,
                candidates_considered=output.candidates_considered,
            ))

        except Exception as e:
            log.error(f"Failed on sample ({fleurs_code}): {e}")
            continue

    elapsed = time.time() - t0
    n = len(results.lid_results)
    log.info(f"Evaluation complete: {n} samples in {elapsed:.1f}s "
             f"({elapsed/max(n,1):.2f}s/sample)")

    # Print summary
    summary = results.to_dict()
    log.info(f"LID Accuracy: {summary['overall_lid_accuracy']:.3f}")
    log.info(f"Mean CER: {summary['overall_mean_cer']:.3f}")
    log.info(f"Routing: {summary['routing_distribution']}")
    log.info(f"Avg Decode Calls: {summary['avg_decode_calls']:.2f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info(f"Results saved to {save_path}")

    return results


def evaluate_baseline_oracle(lang_codes: Optional[List[str]] = None,
                             split: str = "test",
                             max_per_lang: int = 30,
                             whisper_model=None,
                             ) -> dict:
    """Baseline B1: Oracle language → best ASR backend. Upper bound on CER."""
    from src.asr.whisper_backend import WhisperBackend
    from src.asr.mms_backend import MMSBackend
    from src.utils import get_language_map

    if lang_codes is None:
        lang_codes = SUBSET_30_FLEURS

    lang_map = get_language_map()
    whisper_be = WhisperBackend(whisper_model=whisper_model)
    if not whisper_be.is_loaded():
        whisper_be.load()

    mms_be = MMSBackend()
    mms_be.load()

    datasets = load_fleurs(lang_codes, split=split, streaming=True)

    cer_by_lang = {}
    for audio, sr, fleurs_code, true_lang, ref_text in tqdm(
            iterate_fleurs(datasets, max_per_lang=max_per_lang),
            desc="Oracle Baseline (B1)"):
        try:
            # Use Whisper if the language is in its set, otherwise MMS
            if lang_map.whisper_supported(true_lang):
                result = whisper_be.transcribe(audio, true_lang)
            else:
                result = mms_be.transcribe(audio, true_lang)
            cer_val = compute_cer(ref_text, result.text)
            cer_by_lang.setdefault(true_lang, []).append(cer_val)
        except Exception as e:
            log.warning(f"Oracle baseline failed for {true_lang}: {e}")

    whisper_be.unload()
    mms_be.unload()

    summary = {lang: float(sum(vals)/len(vals)) for lang, vals in cer_by_lang.items()}
    summary["overall"] = float(sum(v for vals in cer_by_lang.values() for v in vals) /
                               max(sum(len(v) for v in cer_by_lang.values()), 1))
    return summary


def evaluate_baseline_whisper_auto(lang_codes: Optional[List[str]] = None,
                                   split: str = "test",
                                   max_per_lang: int = 30,
                                   whisper_model=None,
                                   ) -> dict:
    """Baseline B2: Whisper with language=None (its own auto-detect).

    Strong single-model baseline — no external LID at all.
    """
    from src.asr.whisper_backend import WhisperBackend

    if lang_codes is None:
        lang_codes = SUBSET_30_FLEURS

    backend = WhisperBackend(whisper_model=whisper_model)
    if not backend.is_loaded():
        backend.load()

    datasets = load_fleurs(lang_codes, split=split, streaming=True)

    cer_by_lang = {}
    lang_detection = {"correct": 0, "total": 0}

    for audio, sr, fleurs_code, true_lang, ref_text in tqdm(
            iterate_fleurs(datasets, max_per_lang=max_per_lang),
            desc="Whisper Auto Baseline (B2)"):
        try:
            result = backend.transcribe_auto(audio)
            cer_val = compute_cer(ref_text, result.text)
            cer_by_lang.setdefault(true_lang, []).append(cer_val)
            lang_detection["total"] += 1
            if result.language == true_lang:
                lang_detection["correct"] += 1
        except Exception as e:
            log.warning(f"Whisper auto baseline failed for {true_lang}: {e}")

    backend.unload()

    summary = {lang: float(sum(vals)/len(vals)) for lang, vals in cer_by_lang.items()}
    summary["overall"] = float(sum(v for vals in cer_by_lang.values() for v in vals) /
                               max(sum(len(v) for v in cer_by_lang.values()), 1))
    summary["whisper_lid_accuracy"] = (lang_detection["correct"] / max(lang_detection["total"], 1))
    return summary


def evaluate_baseline_static_mms(lang_codes: Optional[List[str]] = None,
                                  split: str = "test",
                                  max_per_lang: int = 30,
                                  ) -> dict:
    """Baseline B3: MMS-LID → MMS-ASR (static cascade, no routing agent).

    Acoustic LID picks the top-1 language, and MMS-ASR directly decodes
    with that language. No fusion, no multi-hypothesis, no fallback.
    """
    from src.lid.acoustic_lid import AcousticLID
    from src.asr.mms_backend import MMSBackend
    from src.utils import get_language_map

    if lang_codes is None:
        lang_codes = SUBSET_30_FLEURS

    lang_map = get_language_map()
    lid = AcousticLID(device="cuda")
    lid.load()
    asr = MMSBackend(device="cuda")
    asr.load()

    datasets = load_fleurs(lang_codes, split=split, streaming=True)

    cer_by_lang = {}
    lid_correct = 0
    lid_total = 0

    for audio, sr, fleurs_code, true_lang, ref_text in tqdm(
            iterate_fleurs(datasets, max_per_lang=max_per_lang),
            desc="Static MMS Baseline (B3)"):
        try:
            probs = lid.predict(audio, sr)
            predicted_lang = max(probs, key=probs.get) if probs else "eng"
            # Map to canonical
            canon = lang_map.from_mms_lid(predicted_lang) or predicted_lang
            lid_total += 1
            if canon == true_lang:
                lid_correct += 1
            result = asr.transcribe(audio, canon)
            cer_val = compute_cer(ref_text, result.text)
            cer_by_lang.setdefault(true_lang, []).append(cer_val)
        except Exception as e:
            log.warning(f"Static MMS baseline failed for {true_lang}: {e}")

    lid.unload()
    asr.unload()

    summary = {lang: float(sum(vals)/len(vals)) for lang, vals in cer_by_lang.items()}
    summary["overall"] = float(sum(v for vals in cer_by_lang.values() for v in vals) /
                               max(sum(len(v) for v in cer_by_lang.values()), 1))
    summary["mms_lid_accuracy"] = lid_correct / max(lid_total, 1)
    return summary


def evaluate_baseline_static_sb_whisper(lang_codes: Optional[List[str]] = None,
                                         split: str = "test",
                                         max_per_lang: int = 30,
                                         ) -> dict:
    """Baseline B4: SpeechBrain ECAPA LID → Whisper ASR (static cascade).

    Lightweight LID + quality ASR, no routing agent.
    Only works for languages in both SpeechBrain and Whisper sets.
    """
    from src.lid.baseline_lid import BaselineLID
    from src.asr.whisper_backend import WhisperBackend
    from src.utils import get_language_map

    if lang_codes is None:
        lang_codes = SUBSET_30_FLEURS

    lang_map = get_language_map()
    lid = BaselineLID(device="cuda")
    lid.load()
    asr = WhisperBackend()
    asr.load()

    datasets = load_fleurs(lang_codes, split=split, streaming=True)

    cer_by_lang = {}
    lid_correct = 0
    lid_total = 0

    for audio, sr, fleurs_code, true_lang, ref_text in tqdm(
            iterate_fleurs(datasets, max_per_lang=max_per_lang),
            desc="Static SB+Whisper Baseline (B4)"):
        try:
            probs = lid.predict(audio, sr)
            predicted_lang = max(probs, key=probs.get) if probs else "eng"
            # SpeechBrain codes may differ; try to normalize
            canon = lang_map.from_whisper(predicted_lang) or predicted_lang
            lid_total += 1
            if canon == true_lang:
                lid_correct += 1
            # Only use Whisper if the language is supported
            if lang_map.whisper_supported(canon):
                result = asr.transcribe(audio, canon)
            else:
                # Skip non-Whisper languages
                continue
            cer_val = compute_cer(ref_text, result.text)
            cer_by_lang.setdefault(true_lang, []).append(cer_val)
        except Exception as e:
            log.warning(f"Static SB+Whisper baseline failed for {true_lang}: {e}")

    lid.unload()
    asr.unload()

    summary = {lang: float(sum(vals)/len(vals)) for lang, vals in cer_by_lang.items()}
    summary["overall"] = float(sum(v for vals in cer_by_lang.values() for v in vals) /
                               max(sum(len(v) for v in cer_by_lang.values()), 1))
    summary["sb_lid_accuracy"] = lid_correct / max(lid_total, 1)
    return summary
