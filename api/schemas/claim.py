from datetime import date

from pydantic import BaseModel, Field


class ClaimIn(BaseModel):
    claim_id: str
    member_id: str
    provider_id: str
    code: str
    claim_amount: float = Field(gt=0)
    claim_date: date
    member_age: int | None = Field(default=None, ge=0, le=120)