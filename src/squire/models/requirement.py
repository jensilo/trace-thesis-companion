from pydantic import BaseModel, Field
from typing import Optional

class Requirement(BaseModel):
    """
    Represents a single requirement from the NICE dataset.
    """
    project_id: str = Field(..., alias="ProjectID")
    text: str = Field(..., alias="RequirementText")
    line_number: int | None = None  # 1-based CSV line number (set after construction)

    # Classification Flags (0 or 1 in CSV)
    is_functional: int = Field(..., alias="IsFunctional")
    is_quality: int = Field(..., alias="IsQuality")

    # NFR Categories (Binary)
    availability: int = Field(0, alias="Availability (A)")
    fault_tolerance: int = Field(0, alias="Fault Tolerance (FT)")
    legal: int = Field(0, alias="Legal (L)")
    look_and_feel: int = Field(0, alias="Look & Feel (LF)")
    maintainability: int = Field(0, alias="Maintainability (MN)")
    operability: int = Field(0, alias="Operability (O)")
    performance: int = Field(0, alias="Performance (PE)")
    portability: int = Field(0, alias="Portability (PO)")
    scalability: int = Field(0, alias="Scalability (SC)")
    security: int = Field(0, alias="Security (SE)")
    usability: int = Field(0, alias="Usability (US)")
    other: int = Field(0, alias="Other (OT)")

    class Config:
        populate_by_name = True

    @property
    def categories(self) -> list[str]:
        _map = [
            ("quality", self.is_quality),
            ("functional", self.is_functional),
            ("availability", self.availability),
            ("fault_tolerance", self.fault_tolerance),
            ("legal", self.legal),
            ("look_and_feel", self.look_and_feel),
            ("maintainability", self.maintainability),
            ("operability", self.operability),
            ("performance", self.performance),
            ("portability", self.portability),
            ("scalability", self.scalability),
            ("security", self.security),
            ("usability", self.usability),
            ("other", self.other),
        ]
        return [name for name, flag in _map if flag]
