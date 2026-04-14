from typing import Optional
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, Integer


class RoutineBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None, nullable=True)
    user_id: int = Field(index=True)


class Routine(RoutineBase, table=True):
    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=True)
    )


class RoutineWorkoutBase(SQLModel):
    routine_id: int = Field(foreign_key="routine.id")
    workout_id: int = Field(foreign_key="workout.id")
    order: int = Field(default=0)
    sets: int = Field(default=3, ge=1)
    reps: str = Field(default="10")
    note: Optional[str] = Field(default=None, nullable=True)


class RoutineWorkout(RoutineWorkoutBase, table=True):
    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=True)
    )
