from pydantic import BaseModel, Field


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class AssistantChatResponse(BaseModel):
    response: str
    model_used: str