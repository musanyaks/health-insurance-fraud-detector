# Data dictionary

All data is **synthetic**, produced by `synthetic/generate_claims.py` (seeded).
Year of claims: 2024-01-01 → 2024-12-31. Currency: KES.

## Entities

| Entity | Count (default) | Notes |
|---|---|---|
| Members | 5,000 | age 1–89, gender M/F, 7 counties |
| Providers | 120 | tier 1/2/3 → cost multiplier 1.6 / 1.0 / 0.7; **6 are "ring" (fraud) providers** |
| Claims | 62,437 | 3.99% labeled fraud |

## Procedure catalog

| Code | Description | Base cost (KES) | Profile |
|---|---|---|---|
| CONS | Consultation | 1,500 | both |
| PHARM | Pharmacy | 2,500 | both |
| LAB01 | Basic lab | 3,500 | both |
| RAD01 | X-ray | 6,000 | both |
| LAB02 | Advanced lab | 9,000 | both |
| WARD3 | Ward, 3 days | 30,000 | both |
| CTSCN | CT scan | 38,000 | both |
| SURG01 | Minor surgery | 45,000 | both |
| MRISC | MRI | 65,000 | both |
| SURG02 | Major surgery | 180,000 | both |
| DIAL01 | Dialysis session | 12,000 | hard only |
| PHYS01 | Physiotherapy | 3,000 | hard only |

Billed amount = base_cost × tier multiplier × LogNormal(0, 0.25), rounded to 100.

## Raw claim schema (`data/raw/*.parquet`)

| Column | Type | Description |
|---|---|---|
| `claim_id` | str | `CLM#######`, unique |
| `member_id` | str | `MBR#####` |
| `provider_id` | str | `PRV####` |
| `code` | str | procedure code (catalog above) |
| `claim_date` | date | submission date |
| `claim_amount` | float | billed amount, KES |
| `is_fraud` | int | 0/1 **label, with noise** (see below) |
| `fraud_type` | str | generating mechanism (see taxonomy) |
| `member_age` | int | denormalized from member table |

### ⚠️ Label semantics under noise

`is_fraud` and `fraud_type` can deliberately disagree:

- `fraud_type="duplicate"`, `is_fraud=0` → **missed fraud** (10% of all fraud)
- `fraud_type="none"`, `is_fraud=1` → **false flag** (0.5% of legitimate claims)

This mirrors real investigator labels and is why per-mechanism tables
show a `none (label noise)` row.

## Engineered features (`fraud_engine/features/pipeline.py`)

★ = in the model feature list `FEATURES`. All past-only by construction.

| Feature | ★ | Definition | Causal mechanism |
|---|---|---|---|
| `claim_amount` | ★ | raw billed amount | — |
| `member_age` | ★ | from member record | — |
| `code_is_expensive` | ★ | code ∈ {SURG01, SURG02, MRISC, CTSCN} | static set |
| `provider_prior_claims` | ★ | count of provider's earlier claims | `groupby.cumcount()` |
| `provider_prior_avg_amt` | ★ | expanding mean of provider's prior amounts (first claim → own amount) | `shift(1).expanding()` |
| `amount_vs_provider_avg` | ★ | claim_amount / provider_prior_avg_amt | derived |
| `provider_code_prior_count` | | provider's prior uses of this code | `groupby.cumcount()` |
| `provider_code_prior_share` | ★ | prior code count / (prior claims + 1) | derived |
| `member_prior_claims` | ★ | member's earlier claims | `groupby.cumcount()` |
| `member_claims_30d` | ★ | member's claims in trailing 30d, excl. current | rolling `30D` |
| `rule_hits` | ★ | count of fired rules | rules below |
| `r_duplicate` / `r_amount_outlier` / `r_member_freq` / `r_rare_expensive` | | individual rule booleans | see taxonomy |

TBD: hard-profile feature additions (graph features) documented here when built.
