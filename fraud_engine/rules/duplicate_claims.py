"""R1 - duplicate service: same member + provider + procedure within N days."""
import pandas as pd


def rule_duplicate(df: pd.DataFrame, cfg: dict) -> pd.Series:
    win = cfg["rules"]["duplicate_window_days"]
    prev = df.groupby(["member_id", "provider_id", "code"])["claim_date"].shift(1)
    gap_days = (df["claim_date"] - prev).dt.days
    return gap_days.le(win)  # NaN -> False (first occurrence)