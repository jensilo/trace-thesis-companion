from pydantic import BaseModel, Field

class Project(BaseModel):
    """
    Represents a software project with its high-level description.
    """
    id: str = Field(..., description="Unique identifier for the project")
    name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Summary or description of the project")
