from typing import Dict

from sqlmodel import Session, select

from app.config import get_settings
from app.models import Routine, RoutineWorkout, User, Workout

try:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_ollama import ChatOllama
except ImportError:
    create_agent = None
    tool = None
    ChatOllama = None


def _build_tools(db: Session, user_id: int):
    if tool is None:
        return []

    @tool
    def list_my_routines() -> str:
        """List all routines that belong to the current user."""
        routines = db.exec(select(Routine).where(Routine.user_id == user_id)).all()
        if not routines:
            return "No routines found for this user."

        rows = [f"- {routine.id}: {routine.name}" for routine in routines]
        return "\n".join(rows)

    @tool
    def get_routine_details(routine_name: str) -> str:
        """Get workouts inside one of the current user's routines by routine name."""
        target = routine_name.strip().lower()
        routines = db.exec(select(Routine).where(Routine.user_id == user_id)).all()
        selected = next((routine for routine in routines if routine.name.lower() == target), None)
        if selected is None:
            return "Routine not found for current user."

        rows = db.exec(
            select(RoutineWorkout, Workout)
            .join(Workout, Workout.id == RoutineWorkout.workout_id)
            .where(RoutineWorkout.routine_id == selected.id)
            .order_by(RoutineWorkout.order)
        ).all()

        if not rows:
            return f"Routine '{selected.name}' has no workouts yet."

        lines = [f"Routine: {selected.name}"]
        for association, workout in rows:
            lines.append(f"{association.order}. {workout.title} ({workout.body_part}, {workout.level})")
        return "\n".join(lines)

    @tool
    def suggest_workouts(body_part: str = "", level: str = "", workout_type: str = "") -> str:
        """Suggest workouts filtered by body part, level, and type."""
        query = select(Workout)
        if body_part.strip():
            query = query.where(Workout.body_part.ilike(body_part.strip()))
        if level.strip():
            query = query.where(Workout.level.ilike(level.strip()))
        if workout_type.strip():
            query = query.where(Workout.type.ilike(workout_type.strip()))

        workouts = db.exec(query.limit(20)).all()
        if not workouts:
            return "No workouts matched those filters."

        rows = [
            f"- {workout.title} | type: {workout.type} | body_part: {workout.body_part} | level: {workout.level}"
            for workout in workouts
        ]
        return "\n".join(rows)

    return [list_my_routines, get_routine_details, suggest_workouts]


def _extract_assistant_message(result: Dict[str, Any]) -> str:
    messages = result.get("messages") if isinstance(result, dict) else None
    if isinstance(messages, list):
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if content:
                return str(content)
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"])
    return "I could not generate a response right now."


def generate_assistant_response(message: str, user: User, db: Session) -> Dict[str, str]:
    settings = get_settings()
    if create_agent is None or tool is None or ChatOllama is None:
        return {
            "response": "LangChain/Ollama dependencies are missing. Install langchain and langchain-ollama.",
            "model_used": settings.ollama_model_name,
        }

    model = ChatOllama(
        model=settings.ollama_model_name,
        temperature=settings.ollama_temperature,
        base_url=settings.ollama_base_url,
    )

    assistant = create_agent(
        model=model,
        tools=_build_tools(db, user.id),
        system_prompt=(
            "You are Workout Master Assistant. Help users plan and improve workouts. "
            "Prefer using tools when users ask about their routines or workout suggestions. "
            "Keep responses practical and concise."
        ),
    )

    try:
        result = assistant.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": f"user-{user.id}"}},
        )
    except Exception as exc:
        return {
            "response": f"I could not reach Ollama at {settings.ollama_base_url}. Error: {exc}",
            "model_used": settings.ollama_model_name,
        }

    return {
        "response": _extract_assistant_message(result),
        "model_used": settings.ollama_model_name,
    }
