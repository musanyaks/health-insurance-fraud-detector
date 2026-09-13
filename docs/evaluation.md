# Evaluation methodology

## Protocol

1. Generate claims (seeded) → `data/raw/claims.parquet`
2. Build features with the shared builder (past-only — see `test_no_leakage.py`)
3. **Chronological split** at the 80th percentile of `claim_date`
   (train ≤ **2024-10-18**, test = final 2.5 months, n = 12,590)
4. Fit on train, score test, never touch test during development

## Why these metrics

| Metric | Formula / definition | Why |
|---|---|---|
| PR-AUC | area under precision–recall curve | At ~4% base rate it focuses on the rare class; ROC-AUC flatters |
| Recall @ 10% review | fraud caught ÷ total fraud, among the riskiest 10% of test claims | Matches a real review team's capacity constraint |
| Per-mechanism recall | same, computed per `fraud_type` | Prevents aggregate metrics from hiding failure on specific patterns |
| — | accuracy | **Never reported**: "never fraud" scores 96% |

## Results (easy profile)

| Model | PR-AUC | Recall @ 10% |
|---|---|---|
| Logistic baseline | 0.445 | 0.755 |
| XGBoost | **0.711** | **0.880** |

Baseline exists to prove the features (not the model class) carry the signal:
XGBoost's lift over logistic = +0.27 PR-AUC.

Per-mechanism: see `fraud_taxonomy.md` (kept there so generator and recall
stay in one table).

## Threats to validity

- **Label noise**: metrics measured against noisy labels (10% FN, 0.5% FP) —
  true performance is slightly *higher* than reported on clean labels, but
  unknowable from noisy ones.
- **Benchmark concentration**: 6 ring providers → 1.000 recalls measure design,
  not real-world skill (ADR-008 addresses this).
- **Single time split**: one cutoff date; P2 adds rolling-origin evaluation.
- **No calibration analysis yet**: `ml_probability` is used monotonically by
  the composite score, so uncalibrated probabilities don't break rankings.

## Reproduce

```bash
make data && make train      # prints every number above
make test                    # proves the feature causality behind them
```
