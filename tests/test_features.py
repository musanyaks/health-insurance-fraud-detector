import numpy as np

from fraud_engine.features.pipeline import FEATURES


def test_all_features_present(featured_df):
    missing = set(FEATURES) - set(featured_df.columns)
    assert not missing, f"missing: {missing}"


def test_features_numeric_and_finite(featured_df):
    X = featured_df[FEATURES]
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    assert np.isfinite(X.to_numpy()).all()


def test_first_claim_per_provider_has_no_history(featured_df):
    firsts = featured_df.loc[featured_df.groupby("provider_id")["claim_date"].idxmin()]
    assert (firsts["provider_prior_claims"] == 0).all()


def test_member_frequency_non_negative(featured_df):
    assert (featured_df["member_claims_30d"] >= 0).all()

def test_build_features_for_claim_returns_submitted_row(claims_df):
    from fraud_engine.features.pipeline import build_features_for_claim
    from fraud_engine.config import CONFIG

    claim = {
        "claim_id": "CLMNEW0001", "member_id": claims_df["member_id"].iloc[0],
        "provider_id": claims_df["provider_id"].iloc[0], "code": "CONS",
        "claim_amount": 1500.0, "claim_date": "2024-12-05", "member_age": 40,
    }
    out = build_features_for_claim(claim, claims_df, CONFIG)
    assert out.loc[0, "claim_id"] == "CLMNEW0001"   # would fail with tail(1)