"""Batch ETL path: raw parquet -> featured parquet. Thin wrapper on purpose —
all logic lives in fraud_engine.features.pipeline (one source of truth)."""
import pandas as pd

from fraud_engine.config import CONFIG, path
from fraud_engine.features.pipeline import build_features


def main():
    df = pd.read_parquet(path("raw"))
    out = build_features(df, CONFIG)
    dest = path("processed_dir") / "claims_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"wrote {len(out):,} featured claims -> {dest}")


if __name__ == "__main__":
    main()