"""R3 - member frequency: >= N claims in the trailing 30 days (excluding current)."""
import pandas as pd


def rule_member_frequency(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return df["member_claims_30d"].ge(cfg["rules"]["member_freq_threshold"])