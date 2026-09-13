# Architecture

## System context

The platform has two distinct paths — **offline (training)** and **online (serving)** —
that share exactly one component: the feature builder.

```text
OFFLINE (batch, reproducible via `make`)
─────────────────────────────────────────
synthetic/generate_claims.py ──► data/raw/claims.parquet
                                        │
                        fraud_engine.features.build_features   ★ shared
                        fraud_engine.rules.apply_rules         ★ shared
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
            train_logistic.py                        train_xgboost.py
            (baseline, sanity)                       (primary model)
                    │                                       │
                    └───────────► models/*.joblib  ◄────────┘
                                  data/processed/investigation_queue.csv
                                              │
                                     dashboard/shiny/app.R

ONLINE (per-request)
─────────────────────────────────────────
client ──► POST /v1/score ──► api/deps.py (model + history, cached)
                │
                ▼
   build_features_for_claim(claim, history)   ← calls the SAME builder
                │
                ▼
   FraudScorer: P(fraud) + rule hits + TreeSHAP reasons ──► 0–100 score JSON
```

## Components and responsibilities

| Component | Responsibility | Must never do |
|---|---|---|
| `synthetic/` | Seeded data generation, planted fraud | Depend on any other component |
| `fraud_engine/features/` | **The only definition of features** | Import from api/, etl/, tests/ |
| `fraud_engine/rules/` | Declarative rule registry, past-only predicates | Read raw history itself (features provide context) |
| `fraud_engine/supervised/` | Train models, export artifacts + metrics | Define features inline |
| `fraud_engine/scoring/` | Composite score, explanations, policy override | Retrain, recompute features |
| `fraud_engine/evaluation/` | PR-AUC, recall@k, per-mechanism recall | Report accuracy |
| `api/` | HTTP transport, validation, dependency injection | Contain feature or scoring logic |
| `etl/` | Thin batch wrapper | Contain feature logic (wraps the builder) |
| `dashboard/` | Visualize exported artifacts | Compute anything |

## Dependency rule

```text
synthetic ──► (parquet artifacts) ──► fraud_engine ◄── { api, etl, tests, notebooks }
```

Arrows point **toward** `fraud_engine`, never out of it. This is what makes the
package testable in isolation and reusable beyond this repo.

## Artifact lifecycle

| Artifact | Produced by | Committed? |
|---|---|---|
| `data/raw/*.parquet` | `make data` (seeded, deterministic) | No — `data/sample/` is committed instead |
| `models/*.joblib`, `models/*_metrics.json` | `make train` | No |
| `data/processed/investigation_queue.csv` | `make train` | No |

A clean clone + `make install data train` reproduces every number in the README.

## Key invariants

1. **Causal features.** Every aggregate uses `shift(1)`, `cumcount()`, or a trailing
   rolling window — provably past-only (`tests/test_no_leakage.py`).
2. **One feature builder.** Training, ETL, and serving all call
   `build_features()`. There is no second implementation anywhere.
3. **Row identity at serving time.** `build_features_for_claim()` selects the scored
   row by `claim_id`, never by position — history may contain claims dated *after*
   the submitted claim (see ADR-006 in `decisions.md`).
4. **Rules are past-only predicates** over context columns the feature builder
   already produced; they cannot peek at the future by construction.
5. **No plotting/HEAVY deps on the serving path.** SHAP values come from XGBoost's
   native `pred_contribs` (ADR-005).

## P2/P3 evolution (planned, not built)

PostgreSQL-backed history behind `api/deps.py` (same interface, new source) →
`GET /v1/queue` → graph features in `fraud_engine/graph/` → Kafka ingestion →
Airflow orchestration. Each addition slots in behind an existing interface;
none requires changing the feature builder's contract.
