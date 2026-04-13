from typing import Dict, List

from google import genai
from sqlalchemy import or_
from sqlmodel import Session, select

from app.config import get_settings
from app.models import Routine, RoutineWorkout, User, Workout

def _list_my_routines(db: Session, user_id: int) -> str:
    routines = db.exec(select(Routine).where(Routine.user_id == user_id)).all()
    if not routines:
        return "No routines found for this user."

    rows = []
    for routine in routines:
        workout_count = db.exec(
            select(RoutineWorkout).where(RoutineWorkout.routine_id == routine.id)
        ).all()
        rows.append(
            f"- {routine.name}: {len(workout_count)} workouts. Notes: {routine.description or 'None'}"
        )
    return "\n".join(rows)


def _get_routine_details(db: Session, user_id: int, message: str) -> str:
    routines = db.exec(select(Routine).where(Routine.user_id == user_id)).all()
    matched = None
    lowered_message = message.lower()
    for routine in routines:
        if routine.name.lower() in lowered_message:
            matched = routine
            break

    if matched is None:
        return "No specific routine referenced in the user request."

    rows = db.exec(
        select(RoutineWorkout, Workout)
        .join(Workout, Workout.id == RoutineWorkout.workout_id)
        .where(RoutineWorkout.routine_id == matched.id)
        .order_by(RoutineWorkout.order)
    ).all()

    if not rows:
        return f"Routine '{matched.name}' exists but has no workouts yet."

    lines = [f"Routine details for {matched.name}:"]
    for association, workout in rows:
        lines.append(
            f"{association.order}. {workout.title} | body part: {workout.body_part} | type: {workout.type} | "
            f"level: {workout.level} | sets: {association.sets} | reps: {association.reps} | note: {association.note or 'None'}"
        )
    return "\n".join(lines)


def _search_workouts(db: Session, message: str) -> str:
    tokens = [token.strip(" ,.!?;:").lower() for token in message.split() if len(token.strip(" ,.!?;:")) >= 3]
    if not tokens:
        workouts = db.exec(select(Workout).limit(12)).all()
    else:
        clauses = []
        for token in tokens[:8]:
            pattern = f"%{token}%"
            clauses.extend([
                Workout.title.ilike(pattern),
                Workout.description.ilike(pattern),
                Workout.type.ilike(pattern),
                Workout.body_part.ilike(pattern),
                Workout.equipment.ilike(pattern),
                Workout.level.ilike(pattern),
            ])
        workouts = db.exec(select(Workout).where(or_(*clauses)).limit(20)).all()

    if not workouts:
        return "No workouts matched the request terms."

    rows = []
    for workout in workouts:
        rows.append(
            f"- {workout.title} | type: {workout.type} | body part: {workout.body_part} | "
            f"equipment: {workout.equipment} | level: {workout.level} | description: {workout.description or 'None'}"
        )
    return "\n".join(rows)


def _build_prompt(message: str, user: User, db: Session) -> str:
    routines_context = _list_my_routines(db, user.id)
    routine_details_context = _get_routine_details(db, user.id, message)
    workout_context = _search_workouts(db, message)

    return (
        "You are Workout Master Assistant. Use the provided database context to help the user build routines, "
        "adjust sets and reps, and explain workouts accurately. Only use workouts from the provided context when "
        "making specific routine recommendations. If the context is insufficient, say so clearly. Keep responses concise and practical.\n\n"
        f"Current user: {user.username}\n\n"
        "User routines:\n"
        f"{routines_context}\n\n"
        "Referenced routine details:\n"
        f"{routine_details_context}\n\n"
        "Relevant workout library matches:\n"
        f"{workout_context}\n\n"
        "User request:\n"
        f"{message}"
    )


def generate_assistant_response(message: str, user: User, db: Session) -> Dict[str, str]:
    settings = get_settings()
    if not settings.gemini_api_key:
        return {
            "response": "Gemini is not configured yet. Add GEMINI_API_KEY to your .env file.",
            "model_used": settings.gemini_model_name,
        }

    prompt = _build_prompt(message, user, db)

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=prompt,
        )
    except Exception as exc:
        return {
            "response": f"I could not reach Gemini right now. Error: {exc}",
            "model_used": settings.gemini_model_name,
        }

    return {
        "response": getattr(response, "text", None) or "I could not generate a response right now.",
        "model_used": settings.gemini_model_name,
    }
