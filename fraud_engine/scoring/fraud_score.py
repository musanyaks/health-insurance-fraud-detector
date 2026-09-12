"""Composite fraud risk score (0-100) + TreeSHAP 'Reasons'.

Uses XGBoost's native pred_contribs (TreeSHAP) instead of the `shap` package:
identical values, no matplotlib dependency, faster at serving time.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_engine.config import CONFIG
from fraud_engine.features.pipeline import FEATURES
from fraud_engine.rules import RULES

REASON_TEXT = {
    "r_duplicate": "Duplicate procedure: same patient/provider/procedure within 7 days",
    "r_amount_outlier": "Claim amount far above this provider's historical average",
    "r_member_freq": "Patient has an excessive number of claims in the last 30 days",
    "r_rare_expensive": "Provider rarely bills this high-cost procedure",
    "amount_vs_provider_avg": "Claim amount is an extreme multiple of provider average",
    "member_claims_30d": "Abnormal patient claim frequency",
    "provider_code_prior_share": "Procedure unusual for this provider's history",
    "code_is_expensive": "High-cost procedure code",
    "provider_prior_claims": "Provider claim volume",
    "member_prior_claims": "Patient claim history volume",
    "claim_amount": "Absolute claim amount",
    "provider_prior_avg_amt": "Provider's historical average claim amount",
    "member_age": "Patient age",
    "rule_hits": "Multiple fraud rules triggered",
}


class FraudScorer:
    def __init__(self, model, cfg: dict = CONFIG):
        self.model = model
        self.cfg = cfg
        self.booster = model.get_booster()

    def _top_shap_features(self, x: np.ndarray, k: int = 3) -> np.ndarray:
        # feature_names must match training, or xgboost raises a mismatch error
        dm = xgb.DMatrix(x, feature_names=FEATURES)
        contribs = self.booster.predict(dm, pred_contribs=True)
        sv = contribs[0][:-1]  # last column is the bias term
        return np.abs(sv).argsort()[::-1][:k]

    def score(self, features_row, ml_proba: float) -> dict:
        s = self.cfg["scoring"]
        row = features_row if isinstance(features_row, pd.Series) else pd.Series(features_row)

        rule_hits = [r for r in (rule.name for rule in RULES) if bool(row[r])]
        rule_norm = min(len(rule_hits), 3) / 3
        composite = 100 * (s["ml_weight"] * float(ml_proba) + s["rule_weight"] * rule_norm)

        x = row[FEATURES].astype(float).to_numpy().reshape(1, -1)
        reasons = [REASON_TEXT[r] for r in rule_hits]
        for i in self._top_shap_features(x):
            text = REASON_TEXT.get(FEATURES[i], FEATURES[i].replace("_", " "))
            if text not in reasons:
                reasons.append(text)
        reasons = reasons[:4]

        level = "HIGH" if composite >= s["high_threshold"] else \
                "MEDIUM" if composite >= s["medium_threshold"] else "LOW"
        action = {"HIGH": "Assign to senior investigator",
                  "MEDIUM": "Standard review queue",
                  "LOW": "Auto-approve"}[level]

        return {
            "fraud_risk_score": int(round(composite)),
            "risk_level": level,
            "ml_probability": round(float(ml_proba), 3),
            "rule_hits": rule_hits,
            "reasons": reasons,
            "recommended_action": action,
        }


def build_investigation_queue(test_df, proba, model, cfg: dict = CONFIG) -> pd.DataFrame:
    """Top-N riskiest claims with full explanation, exported for the dashboard."""
    scorer = FraudScorer(model, cfg)
    idx = np.argsort(-np.asarray(proba))[: cfg["queue"]["size"]]
    rows = []
    for i in idx:
        row = test_df.iloc[i]
        out = scorer.score(row, float(proba[i]))
        rows.append({
            "claim_id": row["claim_id"], "member_id": row["member_id"],
            "provider_id": row["provider_id"], "code": row["code"],
            "claim_date": str(row["claim_date"].date()),
            "claim_amount": row["claim_amount"],
            "is_fraud": int(row["is_fraud"]), "fraud_type": row["fraud_type"],
            **out,
            "rule_hits": ", ".join(out["rule_hits"]) or "none",
            "reasons": " | ".join(out["reasons"]),
        })
    return pd.DataFrame(rows)