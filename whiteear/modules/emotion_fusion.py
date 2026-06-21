from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from whiteear.contracts.emotion_profile import EmotionProfile, VALID_EMOTIONS
from whiteear.contracts.transcription import TranscriptionResult


@dataclass
class AcousticEmotionAdapter:
    def analyze(self, audio_path: str) -> Dict[str, float]:
        """
        Placeholder acoustic inference.
        Returns dimensional scores and categorical hints.
        """
        return {
            "valence": 0.1,
            "arousal": 0.2,
            "dominance": 0.0,
            "joy": 0.18,
            "sadness": 0.14,
            "anger": 0.10,
            "fear": 0.08,
            "surprise": 0.12,
            "disgust": 0.06,
            "neutral": 0.32,
            "confidence": 0.70,
        }


@dataclass
class TextEmotionAdapter:
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Placeholder text-based emotion inference.
        Simple keyword heuristics for now.
        """
        lowered = text.lower().strip()

        base = {
            "joy": 0.10,
            "sadness": 0.10,
            "anger": 0.10,
            "fear": 0.10,
            "surprise": 0.10,
            "disgust": 0.10,
            "neutral": 0.40,
            "confidence": 0.60,
        }

        if not lowered:
            return base

        if any(word in lowered for word in ["öröm", "boldog", "happy", "love"]):
            base.update({"joy": 0.55, "neutral": 0.10, "confidence": 0.80})
        elif any(word in lowered for word in ["szomorú", "sad", "loss", "cry"]):
            base.update({"sadness": 0.55, "neutral": 0.10, "confidence": 0.80})
        elif any(word in lowered for word in ["harag", "angry", "düh", "rage"]):
            base.update({"anger": 0.55, "neutral": 0.10, "confidence": 0.80})
        elif any(word in lowered for word in ["félelem", "fear", "panic", "retteg"]):
            base.update({"fear": 0.55, "neutral": 0.10, "confidence": 0.80})

        return base


class EmotionFusionEngine:
    def __init__(
        self,
        acoustic_adapter: AcousticEmotionAdapter | None = None,
        text_adapter: TextEmotionAdapter | None = None,
        acoustic_weight: float = 0.65,
        text_weight: float = 0.35,
    ) -> None:
        self.acoustic_adapter = acoustic_adapter or AcousticEmotionAdapter()
        self.text_adapter = text_adapter or TextEmotionAdapter()
        self.acoustic_weight = acoustic_weight
        self.text_weight = text_weight

    def analyze(self, audio_path: str, transcription: TranscriptionResult) -> EmotionProfile:
        acoustic = self.acoustic_adapter.analyze(audio_path)
        text = self.text_adapter.analyze(transcription.text)

        emotion_scores = self._merge_emotion_scores(acoustic, text)

        primary_emotion = max(emotion_scores, key=emotion_scores.get)

        overall_confidence = self._compute_confidence(
            acoustic_confidence=acoustic.get("confidence", 0.0),
            text_confidence=text.get("confidence", 0.0),
            transcription_confidence=transcription.confidence,
        )

        return EmotionProfile(
            primary_emotion=primary_emotion,
            emotion_scores=emotion_scores,
            valence=self._weighted_value(acoustic.get("valence", 0.0), 0.0),
            arousal=self._weighted_value(acoustic.get("arousal", 0.0), 0.0),
            dominance=self._weighted_value(acoustic.get("dominance", 0.0), 0.0),
            fusion_method="weighted_acoustic_priority",
            acoustic_weight=self.acoustic_weight,
            text_weight=self.text_weight,
            overall_confidence=overall_confidence,
        )

    def _merge_emotion_scores(
        self,
        acoustic: Dict[str, float],
        text: Dict[str, float],
    ) -> Dict[str, float]:
        merged = {}
        for emotion in VALID_EMOTIONS:
            acoustic_score = acoustic.get(emotion, 0.0)
            text_score = text.get(emotion, 0.0)
            merged[emotion] = round(
                (acoustic_score * self.acoustic_weight)
                + (text_score * self.text_weight),
                4,
            )
        return merged

    def _weighted_value(self, acoustic_value: float, text_value: float) -> float:
        value = (acoustic_value * self.acoustic_weight) + (text_value * self.text_weight)
        return max(-1.0, min(1.0, round(value, 4)))

    def _compute_confidence(
        self,
        acoustic_confidence: float,
        text_confidence: float,
        transcription_confidence: float,
    ) -> float:
        confidence = (
            acoustic_confidence * 0.5
            + text_confidence * 0.2
            + transcription_confidence * 0.3
        )
        return max(0.0, min(1.0, round(confidence, 4)))