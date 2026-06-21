
from orchestrator.state import MIaZAIState

from whiteear.contracts.audio_input import AudioInput

from whiteear.contracts.handoff_payload import WhiteEarPayload

from whiteear.contracts.transcription import TranscriptionResult

from whiteear.contracts.emotion_profile import EmotionProfile

from whiteego.contracts.psycho_context import PsychoContext

from whiteear.modules.emotion_fusion import EmotionFusionEngine

from whiteear.modules.transcriber import Transcriber

from whiteego.modules.llm_caller import generate_lyrics, generate_psycho_reflection
from whiteego.modules.lyric_validator import validate_lyrics
from orchestrator.model_manager import ensure_mode

def initialize_state(audio_input: AudioInput) -> MIaZAIState:

    return {

        "audio_input": audio_input,

        "status": "initialized",

        "error": None,

    }

def attach_whiteear_payload(

    state: MIaZAIState,

    payload: WhiteEarPayload,

) -> MIaZAIState:

    state["whiteear_payload"] = payload

    state["status"] = "whiteear_complete"

    return state

def attach_psycho_context(

    state: MIaZAIState,

    psycho_context: PsychoContext,

) -> MIaZAIState:

    state["psycho_context"] = psycho_context

    state["status"] = "psycho_context_complete"

    return state

def build_placeholder_whiteear_payload(audio_input: AudioInput) -> WhiteEarPayload:

    transcription = TranscriptionResult(

        text="",

        language=audio_input.language_hint or "hu",

        confidence=0.0,

        source_path=audio_input.source_path,

    )

    emotion_profile = EmotionProfile(

        primary_emotion="neutral",

        emotion_scores={"neutral": 1.0},

        valence=0.0,

        arousal=0.0,

        dominance=0.0,

        fusion_method="placeholder",

        acoustic_weight=0.65,

        text_weight=0.35,

        overall_confidence=0.0,

    )

    return WhiteEarPayload(

        source_path=audio_input.source_path,

        transcription=transcription,

        emotion_profile=emotion_profile,

    )


def build_placeholder_psycho_context(state: MIaZAIState) -> PsychoContext:

    payload = state["whiteear_payload"]

    return PsychoContext(

        dominant_emotion=payload.emotion_profile.primary_emotion,

        intensity=payload.emotion_profile.overall_confidence,

        polarity="mixed",

        narrative_frame="placeholder_from_whiteear",

    )


def run_placeholder_pipeline(audio_input: AudioInput) -> MIaZAIState:

    state = initialize_state(audio_input)

    payload = build_placeholder_whiteear_payload(audio_input)

    state = attach_whiteear_payload(state, payload)

    psycho_context = build_placeholder_psycho_context(state)

    state = attach_psycho_context(state, psycho_context)

    psycho_context_dict = psycho_context.model_dump()

    ensure_mode("qwen_only")
    state["psycho_reflection"] = generate_psycho_reflection(psycho_context_dict)

    lyric_output = "Emma lyric generation skipped during qwen-first dry run."
    lyric_validation = {
        "line_count_ok": False,
        "all_rhymes_ok": False,
        "skipped": True,
    }
    lyric_validation_passed = False

    state["lyric_output"] = lyric_output
    state["lyric_validation"] = lyric_validation
    state["lyric_validation_passed"] = lyric_validation_passed

    state["status"] = "dry_run_complete"

    return state
