from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import status
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from app.dependencies.session import SessionDep
from app.dependencies.auth import AdminDep
from app.models import Workout
from app.utilities.flash import flash
from . import router, templates


@router.get("/admin", response_class=HTMLResponse)
async def admin_home_view(
    request: Request,
    user: AdminDep,
    db: SessionDep,
):
    workouts = db.exec(select(Workout).order_by(Workout.title)).all()
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "user": user,
            "workouts": workouts,
        },
    )


@router.post("/admin/add-workout")
async def add_workout(
    request: Request,
    user: AdminDep,
    db: SessionDep,
    title: str = Form(...),
    description: Optional[str] = Form(""),
    workout_type: str = Form(...),
    body_part: str = Form(...),
    equipment: str = Form(...),
    level: str = Form(...),
    rating: Optional[float] = Form(None),
    rating_desc: Optional[str] = Form(""),
):
    if not title.strip() or not workout_type.strip() or not body_part.strip() or not equipment.strip() or not level.strip():
        flash(request, "Please fill in all required workout fields.", "danger")
        return RedirectResponse(url=request.url_for("admin_home_view"), status_code=status.HTTP_303_SEE_OTHER)

    workout = Workout(
        title=title.strip(),
        description=description.strip() if description else None,
        type=workout_type.strip(),
        body_part=body_part.strip(),
        equipment=equipment.strip(),
        level=level.strip(),
        rating=rating,
        rating_desc=rating_desc.strip() if rating_desc else None,
    )
    try:
        db.add(workout)
        db.commit()
        flash(request, f"Workout '{workout.title}' added successfully.")
    except IntegrityError:
        db.rollback()
        flash(request, "A workout with that title already exists.", "danger")
    except Exception:
        db.rollback()
        flash(request, "Unable to add workout. Please try again.", "danger")

    return RedirectResponse(url=request.url_for("admin_home_view"), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/admin/delete-workout")
async def delete_workout(
    request: Request,
    user: AdminDep,
    db: SessionDep,
    workout_id: int = Form(...),
):
    workout = db.get(Workout, workout_id)
    if not workout:
        flash(request, "Workout not found.", "danger")
        return RedirectResponse(url=request.url_for("admin_home_view"), status_code=status.HTTP_303_SEE_OTHER)

    db.delete(workout)
    db.commit()
    flash(request, f"Workout '{workout.title}' deleted.")
    return RedirectResponse(url=request.url_for("admin_home_view"), status_code=status.HTTP_303_SEE_OTHER)
