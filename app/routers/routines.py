from typing import Optional

from fastapi import Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select
from app.dependencies.auth import AuthDep
from app.dependencies.session import SessionDep
from app.models import Routine, RoutineWorkout, Workout
from app.utilities.flash import flash
from . import router, templates


@router.get("/routines", response_class=HTMLResponse)
async def routines_view(request: Request, user: AuthDep, db: SessionDep):
    routines = db.exec(select(Routine).where(Routine.user_id == user.id)).all()
    return templates.TemplateResponse(
        request=request,
        name="routines.html",
        context={
            "user": user,
            "routines": routines,
        },
    )


@router.post("/routines/create")
async def create_routine(
    request: Request,
    user: AuthDep,
    db: SessionDep,
    name: str = Form(...),
    description: str = Form(""),
):
    title = name.strip()
    if not title:
        flash(request, "Please provide a routine name.", "danger")
        return RedirectResponse(url=request.url_for("routines_view"), status_code=status.HTTP_303_SEE_OTHER)

    routine = Routine(name=title, description=description.strip() or None, user_id=user.id)
    db.add(routine)
    db.commit()
    db.refresh(routine)

    flash(request, f"Routine '{routine.name}' created.")
    return RedirectResponse(
        url=request.url_for("routine_detail_view", routine_id=routine.id),
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/routines/{routine_id}/delete")
async def delete_routine(
    request: Request,
    routine_id: int,
    user: AuthDep,
    db: SessionDep,
):
    routine = db.get(Routine, routine_id)
    if not routine:
        flash(request, "Routine not found.", "danger")
        return RedirectResponse(url=request.url_for("routines_view"), status_code=status.HTTP_303_SEE_OTHER)

    if routine.user_id != user.id:
        flash(request, "You are not allowed to delete this routine.", "danger")
        return RedirectResponse(url=request.url_for("routines_view"), status_code=status.HTTP_303_SEE_OTHER)

    associations = db.exec(select(RoutineWorkout).where(RoutineWorkout.routine_id == routine.id)).all()
    for association in associations:
        db.delete(association)

    routine_name = routine.name
    db.delete(routine)
    db.commit()

    flash(request, f"Routine '{routine_name}' deleted.")
    return RedirectResponse(url=request.url_for("routines_view"), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/routines/add")
async def add_workout_to_routine(
    request: Request,
    user: AuthDep,
    db: SessionDep,
    workout_id: int = Form(...),
    routine_id: int = Form(...),
    sets: int = Form(3),
    reps: str = Form("10"),
    note: str = Form(""),
):
    routine = db.get(Routine, routine_id)
    if not routine or routine.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Routine not found")

    workout = db.get(Workout, workout_id)
    if not workout:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workout not found")

    existing = db.exec(
        select(RoutineWorkout)
        .where(RoutineWorkout.routine_id == routine.id)
        .where(RoutineWorkout.workout_id == workout.id)
    ).first()
    if existing:
        flash(request, f"{workout.title} is already in {routine.name}.", "warning")
        return RedirectResponse(request.url_for("routine_detail_view", routine_id=routine.id), status_code=status.HTTP_303_SEE_OTHER)

    max_order_row = db.exec(
        select(RoutineWorkout.order)
        .where(RoutineWorkout.routine_id == routine.id)
        .order_by(RoutineWorkout.order.desc())
    ).first()
    max_order = max_order_row if isinstance(max_order_row, int) else (max_order_row[0] if max_order_row else 0)
    position = (max_order or 0) + 1
    clean_sets = max(1, sets)
    clean_reps = reps.strip() or "10"
    clean_note = note.strip() or None
    association = RoutineWorkout(
        routine_id=routine.id,
        workout_id=workout.id,
        order=position,
        sets=clean_sets,
        reps=clean_reps,
        note=clean_note,
    )
    db.add(association)
    db.commit()

    flash(request, f"Added {workout.title} to {routine.name} with {clean_sets} sets of {clean_reps}.")
    return RedirectResponse(request.url_for("routine_detail_view", routine_id=routine.id), status_code=status.HTTP_303_SEE_OTHER)


@router.get("/routines/{routine_id}", response_class=HTMLResponse)
async def routine_detail_view(
    request: Request,
    routine_id: int,
    user: AuthDep,
    db: SessionDep,
    workout_type: Optional[str] = None,
    body_part: Optional[str] = None,
    equipment: Optional[str] = None,
):
    routine = db.get(Routine, routine_id)
    if not routine or routine.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Routine not found")

    routine_rows = db.exec(
        select(RoutineWorkout, Workout)
        .join(Workout, Workout.id == RoutineWorkout.workout_id)
        .where(RoutineWorkout.routine_id == routine_id)
        .order_by(RoutineWorkout.order)
    ).all()

    existing_workout_ids = [association.workout_id for association, _ in routine_rows]
    available_query = select(Workout)
    if existing_workout_ids:
        available_query = available_query.where(Workout.id.notin_(existing_workout_ids))
    if workout_type:
        available_query = available_query.where(Workout.type == workout_type)
    if body_part:
        available_query = available_query.where(Workout.body_part == body_part)
    if equipment:
        available_query = available_query.where(Workout.equipment == equipment)

    available_workouts = db.exec(available_query.order_by(Workout.title)).all()

    alternative_workouts = {}
    for association, workout in routine_rows:
        alternative_query = select(Workout).where(
            Workout.body_part == workout.body_part,
            Workout.id != workout.id,
        )
        if existing_workout_ids:
            alternative_query = alternative_query.where(Workout.id.notin_(existing_workout_ids))
        alternative_workouts[association.id] = db.exec(alternative_query).all()

    types = [row[0] for row in db.exec(select(Workout.type).distinct().order_by(Workout.type)).all()]
    body_parts = [row[0] for row in db.exec(select(Workout.body_part).distinct().order_by(Workout.body_part)).all()]
    equipments = [row[0] for row in db.exec(select(Workout.equipment).distinct().order_by(Workout.equipment)).all()]

    return templates.TemplateResponse(
        request=request,
        name="routine_detail.html",
        context={
            "user": user,
            "routine": routine,
            "routine_rows": routine_rows,
            "available_workouts": available_workouts,
            "alternative_workouts": alternative_workouts,
            "types": types,
            "body_parts": body_parts,
            "equipments": equipments,
            "selected_workout_type": workout_type,
            "selected_body_part": body_part,
            "selected_equipment": equipment,
        },
    )


@router.post("/routines/{routine_id}/remove")
async def remove_workout(
    request: Request,
    routine_id: int,
    user: AuthDep,
    db: SessionDep,
    association_id: int = Form(...),
):
    routine = db.get(Routine, routine_id)
    if not routine or routine.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Routine not found")

    association = db.get(RoutineWorkout, association_id)
    if not association or association.routine_id != routine.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workout association not found")

    db.delete(association)
    db.commit()

    flash(request, f"Removed workout from {routine.name}.")
    return RedirectResponse(request.url_for("routine_detail_view", routine_id=routine.id), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/routines/{routine_id}/remix")
async def remix_workout(
    request: Request,
    routine_id: int,
    user: AuthDep,
    db: SessionDep,
    association_id: int = Form(...),
    replacement_workout_id: int = Form(...),
):
    routine = db.get(Routine, routine_id)
    if not routine or routine.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Routine not found")

    association = db.get(RoutineWorkout, association_id)
    if not association or association.routine_id != routine.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workout association not found")

    replacement = db.get(Workout, replacement_workout_id)
    if not replacement:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Replacement workout not found")

    current_ids = [row.workout_id for row in db.exec(select(RoutineWorkout).where(RoutineWorkout.routine_id == routine.id)).all()]
    if replacement.id in current_ids and replacement.id != association.workout_id:
        flash(request, "That workout is already in the routine.", "warning")
        return RedirectResponse(request.url_for("routine_detail_view", routine_id=routine.id), status_code=status.HTTP_303_SEE_OTHER)

    association.workout_id = replacement.id
    db.add(association)
    db.commit()

    flash(request, f"Replaced a workout in {routine.name} with {replacement.title}.")
    return RedirectResponse(request.url_for("routine_detail_view", routine_id=routine.id), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/routines/{routine_id}/update-plan")
async def update_routine_workout_plan(
    request: Request,
    routine_id: int,
    user: AuthDep,
    db: SessionDep,
    association_id: int = Form(...),
    sets: int = Form(3),
    reps: str = Form("10"),
    note: str = Form(""),
):
    routine = db.get(Routine, routine_id)
    if not routine or routine.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Routine not found")

    association = db.get(RoutineWorkout, association_id)
    if not association or association.routine_id != routine.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workout association not found")

    association.sets = max(1, sets)
    association.reps = reps.strip() or "10"
    association.note = note.strip() or None
    db.add(association)
    db.commit()

    flash(request, "Routine item updated.")
    return RedirectResponse(request.url_for("routine_detail_view", routine_id=routine.id), status_code=status.HTTP_303_SEE_OTHER)
