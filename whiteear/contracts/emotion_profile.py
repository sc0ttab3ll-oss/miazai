
from pydantic import BaseModel, Field

from typing import Dict

VALID_EMOTIONS = [

    "joy",

    "sadness",

    "anger",

    "fear",

    "surprise",

    "disgust",

    "neutral",

]

class EmotionProfile(BaseModel):

    primary_emotion: str = Field(..., description="Top fused emotion label")

    emotion_scores: Dict[str, float] = Field(default_factory=dict, description="Per-emotion fused scores")

    valence: float = Field(default=0.0, ge=-1.0, le=1.0)

    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)

    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)

    fusion_method: str = Field(default="weighted_acoustic_priority")

    acoustic_weight: float = Field(default=0.65, ge=0.0, le=1.0)

    text_weight: float = Field(default=0.35, ge=0.0, le=1.0)

    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

