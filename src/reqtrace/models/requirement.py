from typing import Literal

from pydantic import BaseModel


class ExtractedRequirement(BaseModel):
    requirement: str
    source_citation: str
    citation_verified: bool
    citation_warning: str | None
    classification: Literal["Functional", "Non-Functional"]
    quality_attribute: str | None
    confidence: Literal[1, 2, 3, 4, 5]
    rationale: str
    follow_up_question: str


class TraceExtraction(BaseModel):
    transcript_file: str
    model: str
    timestamp: str
    requirements: list[ExtractedRequirement]
