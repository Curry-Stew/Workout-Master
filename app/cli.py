import typer
import csv
import subprocess
import platform
import os
from sqlmodel import select

# Import models FIRST to register with SQLModel
from app.models import *
from app.database import create_db_and_tables, get_cli_session, drop_all
from tabulate import tabulate

from app.models.user import UserBase
from app.utilities.security import encrypt_password

cli = typer.Typer()

@cli.callback()
def main() -> None:
    """Workout Master CLI."""
    return None

@cli.command()
def initialize():
    """Initialize the database with Workout data from Gym.csv"""
    typer.echo("Creating database tables...")
    
    # Drop existing tables and recreate them
    drop_all()
    create_db_and_tables()
    
    typer.echo("Loading Workout data from CSV...")
    
    # Read and parse Gym.csv
    csv_file_path = "Gym.csv"
    
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        with get_cli_session() as db:
            bob = UserBase(username='bob', email='bob@mail.com', password=encrypt_password("bobpass"), role='admin')
            bob_db = User.model_validate(bob)

            db.add(bob_db)
            db.commit()      

            for row in csv_reader:
                try:
                    # Handle empty rating field
                    rating = row.get('Rating', '')
                    if rating and rating.strip():
                        try:
                            rating = float(rating)
                        except ValueError:
                            rating = None
                    else:
                        rating = None
                    
                    # Handle empty rating description field
                    rating_desc = row.get('RatingDesc', '')
                    if not rating_desc or rating_desc.strip() == '':
                        rating_desc = None
                    
                    # Handle empty description field
                    description = row.get('Desc', '')
                    if not description or description.strip() == '':
                        description = None
                    
                    # Create Workout object
                    workout = Workout(
                        title=row['Title'],
                        description=description,
                        type=row['Type'],
                        body_part=row['BodyPart'],
                        equipment=row['Equipment'],
                        level=row['Level'],
                        rating=rating,
                        rating_desc=rating_desc
                    )
                    
                    db.add(workout)
                    
                except (ValueError, KeyError) as e:
                    typer.echo(f"Error processing row for {row.get('Title', 'Unknown')}: {e}")
                    continue
            
            db.commit()
    
    typer.echo("Database initialized successfully!")

if __name__ == "__main__":
    cli()