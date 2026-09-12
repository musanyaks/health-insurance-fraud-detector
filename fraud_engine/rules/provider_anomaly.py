"""R2 + R4 - provider-level anomalies, using PRIOR claims only."""
import pandas as pd


def rule_amount_outlier(df: pd.DataFrame, cfg: dict) -> pd.Series:
    r = cfg["rules"]
    return (df["claim_amount"] > r["amount_outlier_multiple"] * df["provider_prior_avg_amt"]) & \
           (df["provider_prior_claims"] >= r["amount_outlier_min_provider_claims"])


def rule_rare_expensive(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Provider bills a high-cost code they have (almost) never billed before."""
    r = cfg["rules"]
    return df["code"].isin(cfg["expensive_codes"]) & \
           (df["provider_code_prior_count"] <= 1) & \
           (df["provider_prior_claims"] >= r["rare_expensive_min_provider_claims"])