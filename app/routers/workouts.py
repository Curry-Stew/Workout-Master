from typing import Optional

from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlmodel import select
from app.dependencies.session import SessionDep
from app.dependencies.auth import AuthDep
from app.models import Routine, Workout
from . import router, templates


def _normalize_distinct_values(rows):
    values = []
    for row in rows:
        if isinstance(row, tuple):
            value = row[0]
        else:
            value = row
        if value:
            values.append(value)
    return values


@router.get("/workouts", response_class=HTMLResponse)
async def workouts_view(
    request: Request,
    user: AuthDep,
    db: SessionDep,
    workout_type: Optional[str] = None,
    body_part: Optional[str] = None,
    equipment: Optional[str] = None,
    level: Optional[str] = None,
):
    query = select(Workout)
    if workout_type:
        query = query.where(Workout.type == workout_type)
    if body_part:
        query = query.where(Workout.body_part == body_part)
    if equipment:
        query = query.where(Workout.equipment == equipment)
    if level:
        query = query.where(Workout.level == level)

    workouts = db.exec(query.order_by(Workout.title)).all()
    routines = db.exec(select(Routine).where(Routine.user_id == user.id)).all()

    types = _normalize_distinct_values(db.exec(select(Workout.type).distinct().order_by(Workout.type)).all())
    body_parts = _normalize_distinct_values(db.exec(select(Workout.body_part).distinct().order_by(Workout.body_part)).all())
    equipments = _normalize_distinct_values(db.exec(select(Workout.equipment).distinct().order_by(Workout.equipment)).all())
    levels = _normalize_distinct_values(db.exec(select(Workout.level).distinct().order_by(Workout.level)).all())

    return templates.TemplateResponse(
        request=request,
        name="workouts.html",
        context={
            "user": user,
            "workouts": workouts,
            "routines": routines,
            "types": types,
            "body_parts": body_parts,
            "equipments": equipments,
            "levels": levels,
            "selected_workout_type": workout_type,
            "selected_body_part": body_part,
            "selected_equipment": equipment,
            "selected_level": level,
        },
    )
