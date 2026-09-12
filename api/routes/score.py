from fastapi import APIRouter, Depends

from fraud_engine.config import CONFIG
from fraud_engine.features.pipeline import FEATURES, build_features_for_claim
from fraud_engine.scoring.fraud_score import FraudScorer

from api.deps import get_history, get_model
from api.schemas.claim import ClaimIn

router = APIRouter()


@router.post("/v1/score")
def score_claim(claim: ClaimIn, model=Depends(get_model), history=Depends(get_history)):
    feats = build_features_for_claim(claim.model_dump(), history, CONFIG)
    proba = float(model.predict_proba(feats[FEATURES])[0, 1])
    result = FraudScorer(model, CONFIG).score(feats.iloc[0], proba)
    return {"claim_id": claim.claim_id, **result}