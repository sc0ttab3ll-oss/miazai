
from pydantic import BaseModel, Field

from typing import Optional

class PsychoContext(BaseModel):

    dominant_emotion: str = Field(..., description="Primary interpreted emotional state")

    intensity: float = Field(..., ge=0.0, le=1.0, description="Emotional intensity from 0 to 1")

    polarity: str = Field(..., description="Overall polarity, e.g. positive, negative, mixed")

    narrative_frame: Optional[str] = Field(default=None, description="Optional interpretive frame")

