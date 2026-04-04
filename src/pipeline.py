"""
End-to-end Pipeline Orchestrator.

This is the main entry point that chains:
  preprocessing → dual LID → fusion → routing agent → decoding → output

It manages model lifecycle (load/unload) to fit within GPU memory
and exposes a simple .run(audio) interface.
"""
import numpy as np
from typing import Optional, List

from src.utils import (
    get_logger, load_config, get_device, PipelineOutput, timed
)
from src.preprocessing import preprocess
from src.lid.acoustic_lid import AcousticLID
from src.lid.decoder_lid import DecoderLID
from src.lid.fusion import LIDFusion
from src.routing.agent import RoutingAgent
from src.routing.policy_rules import RoutingMode
from src.asr.whisper_backend import WhisperBackend
from src.asr.mms_backend import MMSBackend
from src.decoding.single_decode import decode_single
from src.decoding.multi_hypothesis import decode_multi_hypothesis
from src.decoding.fallback_decode import decode_fallback

log = get_logger("pipeline")


class Pipeline:
    """Full LID → Route → ASR pipeline.
    
    Usage:
        pipe = Pipeline()
        pipe.load_models()
        output = pipe.run(audio_array)
        print(output.transcript, output.detected_language, output.routing_mode)
    
    Model loading strategy:
    - Whisper model is SHARED between DecoderLID and WhisperBackend (saves ~3GB).
    - Models can be loaded all at once (enough VRAM) or sequentially.
    """

    def __init__(self, config: Optional[dict] = None,
                 routing_policy: str = "rules"):
        self.config = config or load_config()
        self.device = get_device(self.config.get("lid", {}).get("acoustic", {}).get("device", "cuda"))

        # ── LID ──
        lid_cfg = self.config.get("lid", {})
        self.acoustic_lid = AcousticLID(
            model_id=lid_cfg.get("acoustic", {}).get("model_id", "facebook/mms-lid-4017"),
            device=self.device,
            precision=lid_cfg.get("acoustic", {}).get("precision", "fp16"),
            min_prob=lid_cfg.get("acoustic", {}).get("min_prob_threshold", 0.001),
        )
        self.decoder_lid = DecoderLID(
            model_size=lid_cfg.get("decoder", {}).get("model_size", "large-v3"),
            device=self.device,
        )

        # ── Fusion ──
        fusion_cfg = self.config.get("fusion", {})
        self.fusion = LIDFusion(
            alpha=fusion_cfg.get("default_alpha", 0.6),
            single_signal_penalty=fusion_cfg.get("single_signal_penalty", 0.85),
        )

        # ── Routing ──
        self.routing_agent = RoutingAgent(
            policy=routing_policy,
            config=self.config.get("routing", {}),
        )

        # ── ASR ──
        asr_cfg = self.config.get("asr", {})
        self.whisper_backend = WhisperBackend(
            model_size=asr_cfg.get("whisper", {}).get("model_size", "large-v3"),
            device=self.device,
            beam_size=asr_cfg.get("whisper", {}).get("beam_size", 5),
        )
        self.mms_backend = MMSBackend(
            model_id=asr_cfg.get("mms", {}).get("model_id", "facebook/mms-1b-all"),
            device=self.device,
            precision=asr_cfg.get("mms", {}).get("precision", "fp16"),
        )

        self._models_loaded = False

    def load_models(self, sequential: bool = False):
        """Load all models into GPU memory.
        
        Args:
            sequential: If True, load models one at a time
                       (for constrained VRAM). If False, load all.
        """
        log.info("Loading pipeline models...")

        # 1. MMS-LID (acoustic) — ~2GB FP16
        self.acoustic_lid.load()

        # 2. Whisper (shared between DecoderLID and WhisperBackend) — ~3GB
        self.decoder_lid.load()
        # Share the model instance with ASR backend
        whisper_model = self.decoder_lid.get_model()
        self.whisper_backend.set_model(whisper_model)

        if sequential:
            # On constrained GPUs: unload MMS-LID before loading MMS-ASR
            # (they don't run simultaneously)
            log.info("Sequential mode: MMS-ASR will be loaded on demand")
        else:
            # 3. MMS-ASR — ~2GB FP16
            self.mms_backend.load()

        self._models_loaded = True
        log.info("Pipeline models loaded")

    def unload_models(self):
        # Unload Whisper ASR FIRST (it holds a borrowed reference),
        # then DecoderLID (which owns the model) — this ensures
        # VRAM is actually freed after decoder_lid.unload().
        self.whisper_backend.unload()
        self.decoder_lid.unload()
        self.acoustic_lid.unload()
        self.mms_backend.unload()
        self._models_loaded = False

    @timed
    def run(self, audio: np.ndarray, sr: int = 16000,
            apply_vad: bool = True) -> PipelineOutput:
        """Process a single audio input through the full pipeline.
        
        Args:
            audio: raw waveform (numpy float32 or int16)
            sr: sample rate
            apply_vad: whether to apply Voice Activity Detection
        
        Returns:
            PipelineOutput with transcript, language, confidence, routing info.
        """
        if not self._models_loaded:
            raise RuntimeError("Call pipeline.load_models() before pipeline.run()")

        # ── Step 1: Preprocess ──
        segments = preprocess(audio, sr, apply_vad_flag=apply_vad)
        # For now, use the first (or only) segment for LID.
        # Full segmentation support would process each and merge.
        main_audio = segments[0] if segments else audio

        # ── Step 2: Dual LID ──
        acoustic_probs = self.acoustic_lid.predict(main_audio)
        decoder_probs = self.decoder_lid.predict(main_audio)

        # ── Step 3: Fusion ──
        fused_probs, uncertainty = self.fusion.fuse_and_analyze(
            acoustic_probs, decoder_probs
        )

        # ── Step 4: Route ──
        decision = self.routing_agent.decide(fused_probs, uncertainty)
        log.info(f"Routing: mode={decision.mode}, "
                 f"candidates={decision.candidate_languages}, "
                 f"conf={decision.confidence:.3f}")

        # ── Step 5: Decode (based on routing mode) ──
        all_transcripts = []

        if decision.mode == RoutingMode.SINGLE:
            # Mode A
            lang = decision.candidate_languages[0]
            best = decode_single(main_audio, lang,
                                 self.whisper_backend, self.mms_backend)
            all_transcripts = [best]

        elif decision.mode == RoutingMode.MULTI_HYPOTHESIS:
            # Mode B
            best, all_transcripts = decode_multi_hypothesis(
                main_audio,
                decision.candidate_languages,
                fused_probs,
                self.whisper_backend,
                self.mms_backend,
            )

        elif decision.mode == RoutingMode.FALLBACK:
            # Mode C
            best, all_transcripts = decode_fallback(
                main_audio,
                decision.candidate_languages,
                fused_probs,
                self.whisper_backend,
                self.mms_backend,
            )
        else:
            log.error(f"Unknown routing mode: {decision.mode}")
            best = decode_single(main_audio, decision.candidate_languages[0],
                                 self.whisper_backend, self.mms_backend)

        # ── Step 6: Build output ──
        return PipelineOutput(
            transcript=best.text,
            detected_language=best.language,
            confidence=best.confidence,
            routing_mode=decision.mode,
            candidates_considered=len(decision.candidate_languages),
            backend_used=best.backend,
            lid_distribution=dict(sorted(fused_probs.items(),
                                         key=lambda x: x[1], reverse=True)[:10]),
            all_transcripts=all_transcripts,
        )

    def run_lid_only(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Run just the LID pipeline (no ASR). Useful for evaluation."""
        segments = preprocess(audio, sr)
        main_audio = segments[0] if segments else audio

        acoustic_probs = self.acoustic_lid.predict(main_audio)
        decoder_probs = self.decoder_lid.predict(main_audio)
        fused_probs, uncertainty = self.fusion.fuse_and_analyze(
            acoustic_probs, decoder_probs
        )

        top_lang = max(fused_probs, key=fused_probs.get) if fused_probs else "unk"

        return {
            "acoustic_probs": acoustic_probs,
            "decoder_probs": decoder_probs,
            "fused_probs": fused_probs,
            "uncertainty": uncertainty,
            "top_language": top_lang,
            "routing_decision": self.routing_agent.decide(fused_probs, uncertainty),
        }
