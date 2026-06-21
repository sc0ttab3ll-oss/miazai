
from whiteear.contracts.transcription import TranscriptionResult

from whiteear.modules.emotion_fusion import EmotionFusionEngine

def test_emotion_fusion_returns_profile():

    engine = EmotionFusionEngine()

    transcription = TranscriptionResult(

        text="boldog vagyok és tele vagyok örömmel",

        language="hu",

        confidence=0.9,

        source_path="sample.wav",

    )

    profile = engine.analyze("sample.wav", transcription)

    assert profile.primary_emotion in profile.emotion_scores

    assert 0.0 <= profile.overall_confidence <= 1.0

    assert profile.fusion_method == "weighted_acoustic_priority"

