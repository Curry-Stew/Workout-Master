import typer
import csv
import subprocess
import platform
import os
from sqlmodel import select

# Import models FIRST to register with SQLModel
from app.models import Pokemon
from app.database import create_db_and_tables, get_cli_session, drop_all
from app.auth import encrypt_password
from tabulate import tabulate

cli = typer.Typer()

@cli.command()
def initialize():
    """Initialize the database with Pokemon data from pokemon.csv"""
    typer.echo("Creating database tables...")
    
    # Drop existing tables and recreate them
    drop_all()
    create_db_and_tables()
    
    typer.echo("Loading Workout data from CSV...")
    
    # Read and parse pokemon.csv
    csv_file_path = "pokemon.csv"
    
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        with get_cli_session() as db:
            for row in csv_reader:
                try:
                    # Handle empty type2 field
                    type2 = row.get('type2', '')
                    if not type2 or type2.strip() == '':
                        type2 = None
                    
                    # Handle empty height/weight fields
                    height = row.get('height_m')
                    if height and height.strip():
                        try:
                            height = float(height)
                        except ValueError:
                            height = None
                    else:
                        height = None
                    
                    weight = row.get('weight_kg')
                    if weight and weight.strip():
                        try:
                            weight = float(weight)
                        except ValueError:
                            weight = None
                    else:
                        weight = None
                    
                    # Create Pokemon object
                    pokemon = Pokemon(
                        pokemon_id=int(row['pokedex_number']),
                        name=row['name'],
                        attack=int(row['attack']),
                        defense=int(row['defense']),
                        sp_attack=int(row['sp_attack']),
                        sp_defense=int(row['sp_defense']),
                        speed=int(row['speed']),
                        hp=int(row['hp']),
                        height=height,
                        weight=weight,
                        type1=row['type1'],
                        type2=type2
                    )
                    
                    db.add(pokemon)
                    
                except (ValueError, KeyError) as e:
                    typer.echo(f"Error processing row for {row.get('name', 'Unknown')}: {e}")
                    continue
            
            db.commit()
    
    typer.echo("Database initialized successfully!")

if __name__ == "__main__":
    cli()