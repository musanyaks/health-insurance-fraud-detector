"""Startup-time singletons, exposed as FastAPI dependencies so tests can override."""
from functools import lru_cache

import joblib
import pandas as pd

from fraud_engine.config import path


@lru_cache
def get_model():
    return joblib.load(path("models_dir") / "xgb_fraud.joblib")


@lru_cache
def get_history() -> pd.DataFrame:
    """MVP: claim history from the raw parquet. P2: query Postgres instead."""
    return pd.read_parquet(path("raw"))