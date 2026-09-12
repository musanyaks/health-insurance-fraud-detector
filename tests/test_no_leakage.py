"""Time-travel test: features for PAST claims must be byte-identical whether or
not FUTURE claims exist in the input. Proves no aggregate peeks ahead."""
import pandas as pd

from fraud_engine.config import CONFIG
from fraud_engine.features.pipeline import build_features


def test_features_are_past_only(claims_df):
    cutoff = claims_df["claim_date"].quantile(0.5)

    past_only = build_features(claims_df[claims_df["claim_date"] <= cutoff], CONFIG)
    full = build_features(claims_df, CONFIG)
    full_past = full[full["claim_date"] <= cutoff]

    pd.testing.assert_frame_equal(
        past_only.reset_index(drop=True),
        full_past[past_only.columns].reset_index(drop=True),
    )