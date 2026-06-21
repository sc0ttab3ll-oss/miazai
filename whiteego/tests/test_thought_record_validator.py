import pytest

from whiteego.contracts.thought_record import ThoughtRecord
from whiteego.modules.thought_record_validator import ValidationError, validate_thought_record


def build_valid_record() -> ThoughtRecord:
    return ThoughtRecord(
        situation="Megláttam egy hibát a kódban.",
        automatic_thought="Biztos elrontottam mindent.",
        emotion="szorongás",
        emotion_intensity=0.8,
        evidence_for=["Van benne hiba."],
        evidence_against=["Már javítottam hasonlót.", "Volt működő része is a változtatásnak."],
        balanced_thought="Van hiba, de ez javítható és nem jelenti azt, hogy minden rossz.",
        re_rated_intensity=0.7,
        cognitive_distortion="catastrophising",
    )


def test_valid_thought_record_passes_all_assertions():
    record = build_valid_record()

    result = validate_thought_record(record)

    assert result is record


def test_evidence_against_with_one_item_raises_validation_error():
    record = build_valid_record()
    record.evidence_against = ["Csak egy ellenbizonyíték van."]

    with pytest.raises(ValidationError):
        validate_thought_record(record)


def test_re_rated_intensity_greater_than_or_equal_to_original_raises_validation_error():
    record = build_valid_record()
    record.re_rated_intensity = record.emotion_intensity

    with pytest.raises(ValidationError):
        validate_thought_record(record)


def test_unknown_distortion_key_raises_validation_error():
    record = build_valid_record()
    record.cognitive_distortion = "unknown_distortion"

    with pytest.raises(ValidationError):
        validate_thought_record(record)
