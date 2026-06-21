from dataclasses import dataclass
from typing import List


DISTORTION_REGISTRY = {
    "all_or_nothing": "fekete-fehér gondolkodás",
    "catastrophising": "katasztrofizálás",
    "mind_reading": "gondolatolvasás",
    "fortune_telling": "jövőbejóslás",
    "emotional_reasoning": "érzelmi következtetés",
    "should_statements": "'kellene' kijelentések",
    "labelling": "cimkézés",
    "personalisation": "perszonalizáció",
    "magnification": "felnagyítás",
    "minimisation": "kicsinyítés",
    "mental_filter": "mentális szűrő",
    "disqualifying_positive": "pozitívumok kizárása",
    "overgeneralisation": "túláltalánosítás",
    "jumping_to_conclusions": "elhamarkodott következtetés",
}


@dataclass
class ThoughtRecord:
    situation: str
    automatic_thought: str
    emotion: str
    emotion_intensity: float
    evidence_for: List[str]
    evidence_against: List[str]
    balanced_thought: str
    re_rated_intensity: float
    cognitive_distortion: str
