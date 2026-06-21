
from whiteear.modules.transcriber import Transcriber

def test_transcriber_returns_transcription_result():

    transcriber = Transcriber()

    result = transcriber.transcribe("sample.wav", language="hu")

    assert result.text == ""

    assert result.language == "hu"

    assert result.confidence == 0.0

    assert result.source_path == "sample.wav"

