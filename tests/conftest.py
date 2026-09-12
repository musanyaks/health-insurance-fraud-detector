import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synthetic.generate_claims import generate_claims  # noqa: E402
from fraud_engine.config import CONFIG  # noqa: E402
from fraud_engine.features.pipeline import FEATURES, build_features  # noqa: E402


@pytest.fixture(scope="session")
def claims_df() -> pd.DataFrame:
    """Small synthetic dataset, generated on the fly -> hermetic tests."""
    return generate_claims(n_members=400, n_providers=30, n_legit=4000, seed=7)


@pytest.fixture(scope="session")
def featured_df(claims_df) -> pd.DataFrame:
    return build_features(claims_df, CONFIG)


@pytest.fixture(scope="session")
def small_model(featured_df):
    cut = featured_df["claim_date"].quantile(0.8)
    tr = featured_df[featured_df["claim_date"] < cut]
    model = XGBClassifier(n_estimators=60, max_depth=4, random_state=0,
                          eval_metric="aucpr")
    model.fit(tr[FEATURES], tr["is_fraud"])
    return model