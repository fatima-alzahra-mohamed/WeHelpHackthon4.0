from fastapi import APIRouter, Depends
from app.api.schemas import ApplicantFeatures, ScoreResponse
from app.services.explain import feature_importance_explanation
from app.main import get_model_service

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/metadata")
def metadata(ms = Depends(get_model_service)):
    return {
        "features": ms.feature_names,
        "model_path": str(ms.model_path),
    }

@router.post("/score", response_model=ScoreResponse)
def score(req: ApplicantFeatures, ms = Depends(get_model_service)):
    warnings = []
    result, X = ms.score(req.payload)

    # explanation based on default_risk model
    dr_model = ms.engine["models"]["default_risk"]
    explanation = feature_importance_explanation(dr_model, X)

    return {
        **result,
        "explanation": explanation,
        "model_version": "credit_engine.pkl",
        "warnings": warnings,
    }
