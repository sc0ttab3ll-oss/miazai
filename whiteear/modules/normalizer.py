
from pathlib import Path

from whiteear.contracts.audio_input import AudioInput, NormalizedAudio

def normalize_audio_input(audio_input: AudioInput) -> NormalizedAudio:

    source = Path(audio_input.source_path)

    if not source.exists():

        raise FileNotFoundError(f"Source audio not found: {source}")

    normalized_path = source.with_name(f"{source.stem}.normalized{source.suffix}")

    return NormalizedAudio(

        source_path=str(source),

        normalized_path=str(normalized_path),

        sample_rate=16000,

        channels=1,

        duration_seconds=None,

    )

