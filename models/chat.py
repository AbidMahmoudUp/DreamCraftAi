from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[str]] = None
    detected_disease: Optional[str] = None
    country: Optional[str] = None
    image: Optional[bytes] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    language: Optional[str] = None