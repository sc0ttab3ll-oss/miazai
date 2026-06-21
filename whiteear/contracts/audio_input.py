
from pydantic import BaseModel, Field

from typing import Optional

class AudioInput(BaseModel):

    source_path: str = Field(..., description="Path to the source audio file")

    language_hint: Optional[str] = Field(default="hu", description="Optional language hint")

    reference_id: Optional[str] = Field(default=None, description="Optional external reference")

class NormalizedAudio(BaseModel):

    source_path: str

    normalized_path: str

    sample_rate: int = 16000

    channels: int = 1

    duration_seconds: Optional[float] = None

