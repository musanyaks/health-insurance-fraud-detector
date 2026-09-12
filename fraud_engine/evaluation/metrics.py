"""Honest metrics for ~4% base-rate fraud. Accuracy is never reported."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))


def recall_at_k(y_true, scores, frac: float = 0.10) -> float:
    """Recall if investigators review only the riskiest `frac` of claims."""
    y = np.asarray(y_true)
    k = max(1, int(frac * len(y)))
    top = np.argsort(-np.asarray(scores))[:k]
    return float(y[top].sum() / max(1, y.sum()))


def per_mechanism_recall(eval_df: pd.DataFrame, scores, frac: float = 0.10) -> pd.DataFrame:
    """The headline table: recall per planted fraud mechanism."""
    k = max(1, int(frac * len(eval_df)))
    picked = eval_df.iloc[np.argsort(-np.asarray(scores))[:k]]
    total = eval_df[eval_df["is_fraud"] == 1].groupby("fraud_type").size()
    caught = (picked[picked["is_fraud"] == 1]
              .groupby("fraud_type").size().reindex(total.index).fillna(0))
    out = pd.DataFrame({"total": total, "caught": caught.astype(int)})
    out["recall"] = (out["caught"] / out["total"]).round(3)
    return out.sort_values("recall", ascending=False)