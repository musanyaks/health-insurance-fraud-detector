"""Rule registry: metadata + application.

Contract: rule functions read only PAST-CAUSAL context columns that
fraud_engine.features.pipeline has already computed (provider_prior_*,
member_claims_30d, provider_code_prior_count). Rules never peek at the future.
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from fraud_engine.rules.duplicate_claims import rule_duplicate
from fraud_engine.rules.excessive_claims import rule_member_frequency
from fraud_engine.rules.provider_anomaly import rule_amount_outlier, rule_rare_expensive


@dataclass(frozen=True)
class Rule:
    name: str
    func: Callable[[pd.DataFrame, dict], pd.Series]
    description: str
    reason: str  # human-readable text shown to investigators


RULES: list[Rule] = [
    Rule("r_duplicate", rule_duplicate,
         "Same member+provider+procedure billed within 7 days",
         "Duplicate procedure: same patient/provider/procedure within 7 days"),
    Rule("r_amount_outlier", rule_amount_outlier,
         "Amount > 3x provider's own historical average",
         "Claim amount far above this provider's historical average"),
    Rule("r_member_freq", rule_member_frequency,
         "Member has 5+ claims in the trailing 30 days",
         "Patient has an excessive number of claims in the last 30 days"),
    Rule("r_rare_expensive", rule_rare_expensive,
         "High-cost code the provider has rarely billed before",
         "Provider rarely bills this high-cost procedure"),
]

RULE_NAMES = [r.name for r in RULES]


def apply_rules(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    for rule in RULES:
        df[rule.name] = rule.func(df, cfg).astype(bool)
    df["rule_hits"] = df[RULE_NAMES].sum(axis=1).astype(int)
    return df