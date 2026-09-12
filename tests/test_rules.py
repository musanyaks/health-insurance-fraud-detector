from fraud_engine.config import CONFIG
from fraud_engine.features.pipeline import build_features
from fraud_engine.rules import RULES, RULE_NAMES


def test_apply_rules_adds_all_rule_columns(featured_df):
    for name in RULE_NAMES:
        assert name in featured_df.columns
    assert "rule_hits" in featured_df.columns


def test_duplicate_rule_catches_planted_duplicates(claims_df):
    feats = build_features(claims_df, CONFIG)
    dup = feats[feats["fraud_type"] == "duplicate"]
    assert len(dup) > 0
    assert dup["r_duplicate"].mean() > 0.5


def test_legitimate_claims_are_mostly_unflagged(featured_df):
    legit = featured_df[featured_df["fraud_type"] == "none"]
    assert (legit["rule_hits"] == 0).mean() > 0.80


def test_rule_registry_metadata_complete():
    assert len(RULES) == len(RULE_NAMES)
    assert all(r.reason for r in RULES)