from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ApplicantFeatures(BaseModel):
    # Keep it flexible for demo: accept arbitrary fields
    # but you should validate required ones in production.
    payload: Dict[str, Any] = Field(..., description="Applicant features key/value")

class ScoreResponse(BaseModel):
    eligibility: Dict[str, Any]
    default_risk: Dict[str, Any]
    affordability: Dict[str, Any]
    explanation: Dict[str, Any]
    model_version: str
    warnings: List[str] = []
