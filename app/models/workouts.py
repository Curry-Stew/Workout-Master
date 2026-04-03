from sqlmodel import Field, SQLModel
from typing import Optional

class WorkoutBase(SQLModel):
    name: str = Field(index=True, unique=True)
    level: str
    sets: int
    reps: int
    intensity: str

class Workout(WorkoutBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)   


