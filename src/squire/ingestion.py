import json
import csv
from pathlib import Path
from typing import List, Dict
from squire.models.project import Project
from squire.models.requirement import Requirement
from squire.models.persona import Persona

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

def load_projects() -> List[Project]:
    """
    Loads projects from data/project_summaries.json.
    """
    json_path = DATA_DIR / 'project_summaries.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Project summaries file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [Project(**item) for item in data]

def load_requirements() -> List[Requirement]:
    """
    Loads requirements from data/NICE.csv.
    """
    csv_path = DATA_DIR / 'NICE.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {csv_path}")
    
    requirements = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            req = Requirement(**row)
            req.line_number = i + 2  # header is line 1; first data row is line 2
            requirements.append(req)
    return requirements

def load_personas() -> List[Persona]:
    """
    Loads personas from data/personas.json.
    """
    json_path = DATA_DIR / 'personas.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Personas file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [Persona(**item) for item in data]

def get_requirements_by_project(project_id: str, requirements: List[Requirement]) -> List[Requirement]:
    """
    Filters requirements for a specific project.
    """
    return [req for req in requirements if req.project_id == project_id]
