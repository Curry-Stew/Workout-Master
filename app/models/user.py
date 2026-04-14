from sqlmodel import Field, SQLModel
from typing import Optional
from pydantic import EmailStr
from sqlalchemy import Column, Integer


class UserBase(SQLModel,):
    username: str = Field(index=True, unique=True)
    email: EmailStr = Field(index=True, unique=True)
    password: str
    role:str = ""

class User(UserBase, table=True):
    id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, autoincrement=True)
    )


    
