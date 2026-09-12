"""ONE feature builder used by training, ETL, tests AND the API.

Causal guarantee: every aggregate uses only claims strictly BEFORE the current
row (shift / cumcount / trailing windows). tests/test_no_leakage.py enforces it.
"""
import pandas as pd

from fraud_engine.config import CONFIG
from fraud_engine.rules import apply_rules

FEATURES = [
    "claim_amount", "member_age", "code_is_expensive",
    "provider_prior_claims", "provider_prior_avg_amt", "amount_vs_provider_avg",
    "provider_code_prior_share", "member_prior_claims", "member_claims_30d",
    "rule_hits",
]

RAW_COLS = ["claim_id", "member_id", "provider_id", "code",
            "claim_date", "claim_amount", "member_age"]


def build_features(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    df = df.sort_values(["claim_date", "claim_id"], kind="mergesort").reset_index(drop=True)

    # ---- static ----
    df["code_is_expensive"] = df["code"].isin(cfg["expensive_codes"]).astype(int)

    # ---- provider context (past-only) ----
    df["provider_prior_claims"] = df.groupby("provider_id").cumcount()
    prior_avg = df.groupby("provider_id")["claim_amount"].transform(
        lambda s: s.shift(1).expanding().mean())
    # first claim per provider: neutral ratio of 1.0
    df["provider_prior_avg_amt"] = prior_avg.fillna(df["claim_amount"])
    df["amount_vs_provider_avg"] = df["claim_amount"] / df["provider_prior_avg_amt"]
    df["provider_code_prior_count"] = df.groupby(["provider_id", "code"]).cumcount()
    df["provider_code_prior_share"] = (
        df["provider_code_prior_count"] / (df["provider_prior_claims"] + 1))

    # ---- member context (past-only) ----
    df["member_prior_claims"] = df.groupby("member_id").cumcount()
    msort = df.sort_values(["member_id", "claim_date"], kind="mergesort")
    window = f'{cfg["rules"]["member_freq_window_days"]}D'
    rolled = (msort.set_index("claim_date")
                   .groupby("member_id")["claim_id"]
                   .rolling(window).count().to_numpy() - 1)  # exclude current claim
    df["member_claims_30d"] = (pd.Series(rolled, index=msort.index)
                                 .reindex(df.index).fillna(0).astype(int))

    # ---- rules (also used as model features via rule_hits) ----
    df = apply_rules(df, cfg)
    return df


def build_features_for_claim(claim: dict, history: pd.DataFrame,
                             cfg: dict = CONFIG) -> pd.DataFrame:
    """Score-time path: append the new claim onto history, run the SAME builder,
    return the row for THIS claim_id. Never use tail(1): history may contain
    claims dated after the submitted claim."""
    new = dict(claim)
    new["claim_date"] = pd.to_datetime(new["claim_date"])
    if new.get("member_age") is None:
        new["member_age"] = 35
    # guard: drop any history row sharing the submitted id so selection is unique
    hist = history.loc[history["claim_id"] != new["claim_id"], RAW_COLS]
    combined = pd.concat([hist, pd.DataFrame([new])[RAW_COLS]], ignore_index=True)
    feats = build_features(combined, cfg)
    row = feats.loc[feats["claim_id"] == new["claim_id"]]
    if row.empty:
        raise ValueError(f"claim {new['claim_id']} not found after feature build")
    return row.reset_index(drop=True)