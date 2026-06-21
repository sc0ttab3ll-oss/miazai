from whiteego.contracts.thought_record import DISTORTION_REGISTRY, ThoughtRecord


class ValidationError(ValueError):
    pass


def _ensure_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")


def validate_thought_record(record: ThoughtRecord) -> ThoughtRecord:
    _ensure_non_empty(record.situation, "situation")
    _ensure_non_empty(record.automatic_thought, "automatic_thought")
    _ensure_non_empty(record.emotion, "emotion")
    _ensure_non_empty(record.balanced_thought, "balanced_thought")
    _ensure_non_empty(record.cognitive_distortion, "cognitive_distortion")

    if len(record.evidence_against) < 2:
        raise ValidationError("evidence_against must contain at least 2 items")

    if record.re_rated_intensity > record.emotion_intensity - 0.05:
        raise ValidationError(
            "re_rated_intensity must be less than or equal to emotion_intensity - 0.05"
        )

    if record.cognitive_distortion not in DISTORTION_REGISTRY:
        raise ValidationError("cognitive_distortion must be a valid DISTORTION_REGISTRY key")

    return record
