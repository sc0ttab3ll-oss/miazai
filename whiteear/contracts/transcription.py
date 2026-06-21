
from pydantic import BaseModel, Field

from typing import Optional

class TranscriptionResult(BaseModel):

    text: str = Field(..., description="Transcribed text output")

    language: str = Field(default="hu", description="Detected or assigned language")

    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Transcription confidence")

    source_path: Optional[str] = Field(default=None, description="Optional source audio path")

