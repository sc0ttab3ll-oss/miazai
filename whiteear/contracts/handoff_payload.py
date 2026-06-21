
from pydantic import BaseModel, Field

from whiteear.contracts.emotion_profile import EmotionProfile

from whiteear.contracts.transcription import TranscriptionResult

class WhiteEarPayload(BaseModel):

    source_path: str = Field(..., description="Original source audio path")

    transcription: TranscriptionResult

    emotion_profile: EmotionProfile

