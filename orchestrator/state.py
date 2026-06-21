from typing import Optional, TypedDict

from whiteear.contracts.audio_input import AudioInput
from whiteear.contracts.handoff_payload import WhiteEarPayload
from whiteego.contracts.psycho_context import PsychoContext


class MIaZAIState(TypedDict, total=False):
    audio_input: AudioInput
    whiteear_payload: WhiteEarPayload
    psycho_context: PsychoContext
    psycho_reflection: str
    lyric_output: str
    lyric_validation: dict
    lyric_validation_passed: bool
    image_prompt: str
    status: str
    error: Optional[str]
