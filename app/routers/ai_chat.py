from fastapi import Request
from fastapi.responses import HTMLResponse

from app.ai import generate_assistant_response
from app.dependencies.auth import AuthDep
from app.dependencies.session import SessionDep
from app.schemas.ai import AssistantChatRequest, AssistantChatResponse

from . import api_router, router, templates


@router.get("/assistant", response_class=HTMLResponse)
async def assistant_view(request: Request, user: AuthDep):
    return templates.TemplateResponse(
        request=request,
        name="assistant.html",
        context={"user": user},
    )


@api_router.post("/assistant/chat", response_model=AssistantChatResponse)
async def assistant_chat(payload: AssistantChatRequest, user: AuthDep, db: SessionDep):
    return generate_assistant_response(payload.message.strip(), user=user, db=db)