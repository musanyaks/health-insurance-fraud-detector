"""Primary model: time-split training, honest metrics, investigation-queue export."""
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_engine.config import CONFIG, path
from fraud_engine.evaluation.metrics import per_mechanism_recall, pr_auc, recall_at_k
from fraud_engine.features.pipeline import FEATURES, build_features
from fraud_engine.scoring.fraud_score import FraudScorer, build_investigation_queue


def main():
    df = build_features(pd.read_parquet(path("raw")), CONFIG)
    cut = df["claim_date"].quantile(CONFIG["split"]["train_fraction"])
    train, test = df[df["claim_date"] < cut], df[df["claim_date"] >= cut]

    spw = (train["is_fraud"] == 0).sum() / max(1, (train["is_fraud"] == 1).sum())
    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr", random_state=42,
    )
    model.fit(train[FEATURES], train["is_fraud"], verbose=False)
    proba = model.predict_proba(test[FEATURES])[:, 1]

    metrics = {
        "model": "xgboost",
        "pr_auc": round(pr_auc(test["is_fraud"], proba), 4),
        "recall_at_10pct": round(recall_at_k(test["is_fraud"], proba, 0.10), 4),
        "cutoff_date": str(cut.date()),
        "n_train": len(train), "n_test": len(test),
    }
    print(json.dumps(metrics, indent=2))
    print("\nRecall per planted fraud mechanism (top-10% review):")
    print(per_mechanism_recall(test, proba))

    models_dir = path("models_dir")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "xgb_fraud.joblib")
    with open(models_dir / "xgb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # export the investigation queue for the Shiny dashboard
    queue = build_investigation_queue(test, proba, model, CONFIG)
    processed = path("processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    queue.to_csv(processed / "investigation_queue.csv", index=False)
    print(f"\nqueue -> {processed / 'investigation_queue.csv'} ({len(queue)} rows)")


if __name__ == "__main__":
    main()