from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class OpenAISpeechRequest(BaseModel):
    model: str = "mira-tts"
    input: str
    voice: str
    response_format: Literal["mp3", "wav", "pcm"] = "wav"

class VoiceInfo(BaseModel):
    id: str
    name: str
    path: str

class VoicesResponse(BaseModel):
    voices: List[VoiceInfo]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    available_voices: int