
from whiteear.contracts.emotion_profile import EmotionProfile

from whiteear.contracts.handoff_payload import WhiteEarPayload

from whiteear.contracts.transcription import TranscriptionResult

def test_handoff_payload_accepts_transcription_and_emotion_profile():

    transcription = TranscriptionResult(

        text="ez egy próba",

        language="hu",

        confidence=0.9,

        source_path="sample.wav",

    )

    emotion_profile = EmotionProfile(

        primary_emotion="joy",

        emotion_scores={"joy": 0.7, "neutral": 0.3},

        valence=0.4,

        arousal=0.2,

        dominance=0.1,

        fusion_method="weighted_acoustic_priority",

        acoustic_weight=0.65,

        text_weight=0.35,

        overall_confidence=0.8,

    )

    payload = WhiteEarPayload(

        source_path="sample.wav",

        transcription=transcription,

        emotion_profile=emotion_profile,

    )

    assert payload.source_path == "sample.wav"

    assert payload.transcription.text == "ez egy próba"

    assert payload.emotion_profile.primary_emotion == "joy"

