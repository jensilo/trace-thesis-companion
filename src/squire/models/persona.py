from pydantic import BaseModel, Field

class Persona(BaseModel):
    """
    Represents an agent persona (Interviewer or Interviewee).
    """
    id: str = Field(..., description="Unique identifier for the persona")
    name: str = Field(..., description="Display name of the persona")
    role: str = Field(..., description="Role description (e.g., Requirements Engineer, Stakeholder)")
    system_prompt_template: str = Field(..., description="Template for the system prompt")
