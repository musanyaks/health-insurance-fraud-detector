# Kenya Health Insurance Claims Fraud Detection & Intelligence Platform

Rules + gradient-boosted ML on health claims — with provably leak-free features, an explained 0–100 fraud risk score, and an investigator-ready API.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

**Stack:** FastAPI · XGBoost · pandas · pytest · R Shiny · Docker

> ⚠️ **All data in this project is synthetic.** Fraud is deliberately planted by a seeded generator, with every mechanism documented in `docs/fraud_taxonomy.md`. No real patient, provider, or claim data is used — which is what makes the project publishable, reproducible, and ethical.

---

## Table of Contents
1. [What this is](#1-what-this-is)
2. [Results](#2-results)
3. [Design guarantees](#3-design-guarantees)
4. [Architecture](#4-architecture)
5. [Fraud taxonomy](#5-fraud-taxonomy)
6. [Quickstart](#6-quickstart)
7. [Scoring API](#7-scoring-api)
8. [Investigation dashboard](#8-investigation-dashboard)
9. [Configuration](#9-configuration)
10. [Project structure](#10-project-structure)
11. [Testing](#11-testing)
12. [Docker](#12-docker)
13. [Roadmap](#13-roadmap)
14. [Known limitations](#14-known-limitations)

---

## 1. What this is

Health insurers lose a meaningful share of claims spend to fraud: duplicate billing, phantom procedures, upcoding, excess utilization, and provider–patient collusion. Most fraud projects stop at CSV → ML → prediction. This one is built as a platform:

- **Synthetic claims generator** with five planted fraud mechanisms (plus a "hard" profile with dispersed fraud and legitimate high-utilization members)
- **Rules engine** — 4 investigator-grade rules in a declarative registry
- **XGBoost** with a logistic baseline, trained on a time-based split
- **Composite risk score (0–100)**: ML probability + rule hits + SHAP-style reasons
- **FastAPI scoring endpoint** with investigator explanations
- **R Shiny dashboard**: investigation queue, claim detail, provider profiles, model monitor
- **14 tests**, including a test that proves no feature can see the future

---

## 2. Results

**Protocol:** chronological 80/20 split (train ≤ 2024-10-18, test on the final ~2.5 months) — the model is always evaluated on claims after its training data, as it would be in production. Base fraud rate ≈ 4%, so accuracy is never reported; PR-AUC and recall-at-review-capacity are.

| Model | PR-AUC | Recall @ 10% review |
|---|---|---|
| Logistic regression (baseline) | 0.445 | 0.755 |
| XGBoost | 0.711 | 0.880 |

*Recall @ 10% review = share of fraud caught if investigators review only the riskiest 10% of claims — the metric an actual claims operation cares about.*

**Recall per planted mechanism** (top-10% review, test period):

| Mechanism | Recall | Primary detection driver |
|---|---|---|
| Phantom procedures (ring) | 1.000 | `provider_code_prior_share` |
| Upcoding (ring) | 1.000 | `amount_vs_provider_avg` |
| Provider–patient collusion | 1.000 | provider history features |
| Duplicate claims | 0.987 | `r_duplicate` |
| Excess frequency | 0.917 | `member_claims_30d` |
| Label noise (mislabeled legit) | 0.089 | — the model does not chase noise |

**Read the 1.000s correctly.** Planted fraud is concentrated in 6 of 120 "ring" providers, which makes it statistically easy to separate. This is a property of the benchmark, not a claim about real-world performance. The hard profile (dispersed fraud, chronic-care members) exists precisely to make the problem honest.

---

## 3. Design guarantees

These are enforced by tests, not asserted in comments:

| Guarantee | How it's enforced |
|---|---|
| Features are causally past-only — no aggregate ever sees a later claim | `tests/test_no_leakage.py` rebuilds the past with the future deleted and asserts byte-identical features. The 1.00 recalls are not time travel. |
| One feature builder shared by training, ETL, and the API — no train/serve skew | `fraud_engine/features/pipeline.py` is the only definition; the serving path calls the identical function. A serving-path bug found via behavioral testing is pinned by regression tests. |
| Time-based splits only | Training and evaluation always split on `claim_date`; no random splits anywhere. |
| A fired rule is never auto-approved | Policy override in the scorer: any rule hit forces at minimum standard review, regardless of ML confidence. |
| Determinism | Fixed seeds in the generator and models; `make data && make train` reproduces these numbers. |

---

## 4. Architecture

```text
synthetic/generate_claims.py            (seeded; plants documented fraud)
          │ parquet
          ▼
fraud_engine.features.build_features    ← ONE source of truth
fraud_engine.rules (registry, 4 rules)
          │
   ┌──────┴───────────┐
   ▼                   ▼
Logistic baseline   XGBoost ──► models/*.joblib + metrics + queue CSV
                            │
                            ▼
        fraud_engine.scoring  (0–100 composite + SHAP-style reasons)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     FastAPI POST /v1/score       R Shiny dashboard (4 tabs)
```

**Dependency rule:** everything imports from `fraud_engine`; nothing inside `fraud_engine` imports from `api/`, `etl/`, or `tests/`. Data and models are build artifacts (gitignored) — a clean clone rebuilds everything via `make`.

---

## 5. Fraud taxonomy

Full traceability matrix (mechanism → generator → rule → feature → measured recall) lives in `docs/fraud_taxonomy.md`. Summary:

| Pattern | Generator mechanism | Rule | Difficulty |
|---|---|---|---|
| Duplicate claims | Legit claim re-billed 1–3 days later | `r_duplicate` | Easy |
| Phantom procedures | Ring providers bill big-ticket codes for walk-ins | `r_rare_expensive` | Medium |
| Upcoding | Cheap service billed under expensive code | — (feature-driven) | Hard |
| Excess frequency | 6–10 identical claims in a 2-week window | `r_member_freq` | Easy |
| Collusion | Ring provider + loyal patients, steady billing | — (graph target) | Hardest |
| Legitimate lookalikes | Chronic-care members; follow-up re-bills | must not fire | Trap |

Label noise is applied on top: 10% of fraud mislabeled as legitimate, 0.5% of legitimate claims flagged — because real labels are never clean.

---

## 6. Quickstart

**Prerequisites:** Python 3.10+, Make. Optional: R ≥ 4.2 (dashboard), Docker.

```bash
make install        # pip install -e ".[dev]"
make data           # ~62k synthetic claims -> data/raw/claims.parquet (~4% fraud)
make train          # logistic baseline + XGBoost; prints metrics + per-mechanism table;
                     # writes models/xgb_fraud.joblib and data/processed/investigation_queue.csv
make test            # 14 tests, incl. the leakage proof
```

Score a claim and explore it live:

```bash
make api            # uvicorn on :8000 — interactive docs at /docs
```

Regenerate just the featured dataset for analysis:

```bash
make features        # data/processed/claims_features.parquet
```

**Windows note:** the Makefile requires tab indentation (`.editorconfig` is included); if a paste converts tabs to spaces, run `sed -E -i 's/^[[:space:]]+/\t/' Makefile` in Git Bash.

---

## 7. Scoring API

`POST /v1/score` — score a single claim against full member/provider history.

```bash
curl -s -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"CLMTEST001","member_id":"MBR00123","provider_id":"PRV0042",
       "code":"MRISC","claim_amount":72000,"claim_date":"2024-12-05"}'
```

Example response:

```json
{
  "claim_id": "CLMTEST001",
  "fraud_risk_score": 87,
  "risk_level": "HIGH",
  "ml_probability": 0.93,
  "rule_hits": ["r_amount_outlier"],
  "reasons": [
    "Claim amount far above this provider's historical average",
    "Duplicate procedure: same patient/provider/procedure within 7 days",
    "Claim amount is an extreme multiple of provider average"
  ],
  "recommended_action": "Assign to senior investigator"
}
```

**Score composition:** `100 × (0.65 × P(fraud|model) + 0.35 × min(rule_hits, 3)/3)` — HIGH ≥ 70, MEDIUM ≥ 40, LOW below. `reasons` = triggered-rule explanations first, then the top-3 TreeSHAP feature contributions (computed via XGBoost's native `pred_contribs` — no extra serving dependencies). Policy override: any fired rule prevents auto-approval.

**Behavioral sanity checks:**

| Input change | Expected response |
|---|---|
| Amount 72,000 → 1,500, code CONS | Score drops to LOW → auto-approve |
| Expensive code the provider has never billed | `r_rare_expensive` fires |
| Date within 7 days of same member+provider+procedure | `r_duplicate` fires |

Other endpoints: `GET /health`. Full schema in `docs/api_documentation.md` or the interactive `/docs` page.

---

## 8. Investigation dashboard

```bash
make shiny          # requires: install.packages(c("shiny","DT","plotly","jsonlite"))
```

Four tabs — **Queue** (filterable top-500 riskiest claims with actual-vs-predicted outcomes), **Claim detail** (score gauge + reasons + claim context), **Provider insights** (flagged-volume rankings), **Model** (time-split metrics, honest caveats). Reads `data/processed/investigation_queue.csv` + `models/xgb_metrics.json` — no Python required at runtime.

---

## 9. Configuration

All tunables live in `config/config.yaml` — no magic constants in code:

| Key | Default | Meaning |
|---|---|---|
| `split.train_fraction` | 0.8 | Chronological split point (quantile of `claim_date`) |
| `expensive_codes` | SURG01, SURG02, MRISC, CTSCN | High-cost procedure set |
| `rules.duplicate_window_days` | 7 | Duplicate-claim lookback |
| `rules.amount_outlier_multiple` | 3.0 | Amount vs. provider's own history |
| `rules.member_freq_window_days` / `threshold` | 30 / 5 | Member frequency rule |
| `scoring.ml_weight` / `rule_weight` | 0.65 / 0.35 | Composite score weights |
| `scoring.high_threshold` / `medium_threshold` | 70 / 40 | Risk-level cutoffs |
| `queue.size` | 500 | Rows exported for the dashboard |

---

## 10. Project structure

```text
├── config/config.yaml          # all tunables
├── synthetic/                  # seeded claims generator (easy + hard profiles)
├── fraud_engine/               # ★ core package — everything imports FROM here
│   ├── config.py               # path/config resolution
│   ├── features/pipeline.py    # build_features() — single source of truth
│   ├── rules/                  # registry + 4 rules
│   ├── supervised/             # logistic baseline + XGBoost
│   ├── scoring/fraud_score.py  # 0–100 composite + SHAP-style reasons
│   └── evaluation/metrics.py   # PR-AUC, recall@k, per-mechanism recall
├── etl/                        # thin batch wrapper on the feature builder
├── api/                        # FastAPI: /v1/score, /health
├── dashboard/shiny/            # investigator UI
├── tests/                      # 14 tests incl. leakage proof
├── data/                       # gitignored; data/sample/ committed for instant inspection
├── models/                     # gitignored build artifacts
└── docs/                       # architecture, data dictionary, fraud taxonomy, API
```

---

## 11. Testing

```bash
make test
```

| File | What it proves |
|---|---|
| `test_no_leakage.py` | Flagship. Features for past claims are identical whether or not future claims exist — no time travel, so reported metrics are honest. |
| `test_rules.py` | Rules catch planted duplicates; >80% of legitimate claims unflagged; a fired rule can never yield auto-approval. |
| `test_features.py` | Feature contract (presence, numeric, finite); serving path returns features for the submitted claim, not an arbitrary row. |
| `test_api.py` | Endpoint schema, validation errors, and that structurally different claims receive different scores. |

Tests are hermetic — they generate their own small seeded dataset. No network, no shared state, full suite < 2s.

---

## 12. Docker

```bash
make train          # models/ must exist first (it is volume-mounted, not baked in)
docker compose up --build
```

Brings up the API on `:8000` and PostgreSQL `:5432` (database integration lands with the `/v1/queue` endpoint — see roadmap).

---

## 13. Roadmap

- [ ] Hard-profile benchmark — chronic-care members (legitimate dense utilization), legitimate near-duplicates, dispersed fraud across ~100 non-ring providers
- [ ] Graph features (NetworkX) — bipartite member–provider graph; success measured as recall lift on dispersed/collusion rows without regressing ring rows
- [ ] PostgreSQL-backed history + `GET /v1/queue`
- [ ] MLflow experiment tracking · Evidently drift monitoring
- [ ] Kafka ingestion (replay producer → scoring consumer) · Airflow DAG
- [ ] GitHub Actions CI · AWS deployment

The experiment arc: baseline → harder problem → targeted feature → measured improvement. Results tables above are updated in place as each stage lands.

---

## 14. Known limitations

Stated plainly, because synthetic-data results deserve context:

- Planted fraud is concentrated (6 ring providers out of 120), so per-mechanism recalls of 1.00 reflect benchmark design, not real-world performance. Real fraud is dispersed and adaptive — that's what the hard profile targets.
- Labels carry noise (10% missed fraud, 0.5% false flags), matching reality — but reported metrics are measured against those noisy labels.
- Serving path rebuilds history context per request — correct and fine at MVP scale; a feature store / precomputed context is the production answer.
- Score weights and thresholds are policy choices (0.65/0.35, 70/40), not learned — they should be tuned with investigators against review capacity.
- Synthetic by design — no PII, safe to publish, fully reproducible; conclusions transfer to real data only after validation on a real claims corpus.

Built as a portfolio demonstration of production ML engineering practices: causal feature discipline, honest evaluation, single-source-of-truth pipelines, and tests that prove the guarantees. See `docs/` for details.

---

## License

MIT
