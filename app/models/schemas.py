from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Employee's message to the assistant")
    thread_id: str = Field(..., description="Conversation/session identifier")


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
