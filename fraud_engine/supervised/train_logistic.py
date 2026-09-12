"""Baseline model - run this FIRST. If XGBoost can't beat it clearly,
the features are the problem, not the model."""
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_engine.config import CONFIG, path
from fraud_engine.evaluation.metrics import per_mechanism_recall, pr_auc, recall_at_k
from fraud_engine.features.pipeline import FEATURES, build_features


def main():
    df = build_features(pd.read_parquet(path("raw")), CONFIG)
    cut = df["claim_date"].quantile(CONFIG["split"]["train_fraction"])
    train, test = df[df["claim_date"] < cut], df[df["claim_date"] >= cut]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(train[FEATURES], train["is_fraud"])
    proba = model.predict_proba(test[FEATURES])[:, 1]

    metrics = {
        "model": "logistic_baseline",
        "pr_auc": round(pr_auc(test["is_fraud"], proba), 4),
        "recall_at_10pct": round(recall_at_k(test["is_fraud"], proba, 0.10), 4),
        "cutoff_date": str(cut.date()),
        "n_train": len(train), "n_test": len(test),
    }
    print(json.dumps(metrics, indent=2))
    print(per_mechanism_recall(test, proba))

    path("models_dir").mkdir(exist_ok=True)
    with open(path("models_dir") / "logistic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()