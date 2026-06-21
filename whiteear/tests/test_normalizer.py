
from pathlib import Path

from whiteear.contracts.audio_input import AudioInput

from whiteear.modules.normalizer import normalize_audio_input

def test_normalizer_returns_expected_output_for_existing_file(tmp_path):

    audio_file = tmp_path / "sample.wav"

    audio_file.write_bytes(b"fake-audio-data")

    audio_input = AudioInput(

        source_path=str(audio_file),

        language_hint="hu",

        reference_id="test-001",

    )

    result = normalize_audio_input(audio_input)

    assert result.source_path == str(audio_file)

    assert result.normalized_path.endswith("sample.normalized.wav")

    assert result.sample_rate == 16000

    assert result.channels == 1

