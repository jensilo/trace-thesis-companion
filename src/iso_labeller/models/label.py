from pydantic import BaseModel, field_validator

from iso_labeller.taxonomy import ALL_SUB_CHARS


class ISOLabelResult(BaseModel):
    project_id: int
    requirement_text: str
    nice_labels: list[str]
    iso_sub_chars: list[str]
    raw_response: str | None = None

    @field_validator("iso_sub_chars")
    @classmethod
    def validate_sub_chars(cls, v: list[str]) -> list[str]:
        return [s for s in v if s in ALL_SUB_CHARS]


class RunOutput(BaseModel):
    model: str
    timestamp: str
    temperature: float
    n_requirements: int
    results: list[ISOLabelResult]
