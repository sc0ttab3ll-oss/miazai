from unittest.mock import patch

from orchestrator.pipeline import (
    attach_psycho_context,
    attach_whiteear_payload,
    build_placeholder_psycho_context,
    build_placeholder_whiteear_payload,
    initialize_state,
    run_placeholder_pipeline,
)
from whiteear.contracts.audio_input import AudioInput


def test_pipeline_state_can_be_initialized_and_extended():
    audio_input = AudioInput(
        source_path="sample.wav",
        language_hint="hu",
        reference_id="pipeline-001",
    )

    state = initialize_state(audio_input)

    assert state["status"] == "initialized"
    assert state["audio_input"].source_path == "sample.wav"

    payload = build_placeholder_whiteear_payload(audio_input)
    state = attach_whiteear_payload(state, payload)

    assert state["status"] == "whiteear_complete"
    assert state["whiteear_payload"].source_path == "sample.wav"

    psycho_context = build_placeholder_psycho_context(state)
    state = attach_psycho_context(state, psycho_context)

    assert state["status"] == "psycho_context_complete"
    assert state["psycho_context"].dominant_emotion == "neutral"
    assert state["psycho_context"].narrative_frame == "placeholder_from_whiteear"


@patch("orchestrator.pipeline.validate_lyrics")
@patch("orchestrator.pipeline.generate_lyrics")
@patch("orchestrator.pipeline.generate_psycho_reflection")
@patch("orchestrator.pipeline.ensure_mode")
def test_run_placeholder_pipeline_completes_dry_run(
    mock_ensure_mode,
    mock_generate_psycho_reflection,
    mock_generate_lyrics,
    mock_validate_lyrics,
):
    mock_generate_psycho_reflection.return_value = "mock reflection"
    mock_generate_lyrics.return_value = "első sor\nmásodik sor\nharmadik sor\nnegyedik sor"
    mock_validate_lyrics.return_value = {
        "line_count_ok": True,
        "all_rhymes_ok": True,
    }

    audio_input = AudioInput(
        source_path="sample.wav",
        language_hint="hu",
        reference_id="pipeline-002",
    )

    state = run_placeholder_pipeline(audio_input)

    assert state["status"] == "dry_run_complete"
    assert state["audio_input"].source_path == "sample.wav"
    assert state["whiteear_payload"].source_path == "sample.wav"
    assert state["psycho_context"].narrative_frame == "placeholder_from_whiteear"
    assert state["psycho_reflection"] == "mock reflection"
    assert state["lyric_output"] == "Emma lyric generation skipped during qwen-first dry run."
    assert state["lyric_validation"]["skipped"] is True
    assert state["lyric_validation_passed"] is False
