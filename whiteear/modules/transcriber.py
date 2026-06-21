
from whiteear.contracts.transcription import TranscriptionResult

class Transcriber:

    def transcribe(self, audio_path: str, language: str = "hu") -> TranscriptionResult:

        """

        v1-lite placeholder transcriber.

        Later this will call a real Whisper-based backend.

        """

        return TranscriptionResult(

            text="",

            language=language,

            confidence=0.0,

            source_path=audio_path,

        )

