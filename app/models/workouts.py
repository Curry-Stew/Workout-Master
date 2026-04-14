from sqlmodel import Field, SQLModel
from typing import Optional
from sqlalchemy import Column, Integer

class WorkoutBase(SQLModel):
    title: str = Field(index=True, unique=True)  # Maps to 'Title' column
    description: str = Field(nullable=True)  # Maps to 'Desc' column (can be empty)
    type: str  # Maps to 'Type' column (Strength, Plyometrics, Cardio, etc.)
    body_part: str  # Maps to 'BodyPart' column (Abdominals, Biceps, etc.)
    equipment: str  # Maps to 'Equipment' column (Bands, Barbell, Dumbbell, etc.)
    level: str  # Maps to 'Level' column (Beginner, Intermediate, Expert)
    rating: float = Field(nullable=True, default=None)  # Maps to 'Rating' column
    rating_desc: str = Field(nullable=True, default=None)  # Maps to 'RatingDesc' column

class Workout(WorkoutBase, table=True):
    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=True)
    )   


