# Fraud taxonomy — traceability matrix

Every planted mechanism maps to: generator code → detection rule → driving feature →
**measured recall**. This matrix is the contract between the data generator and the
fraud engine; if either side changes, this file changes in the same commit.

## Easy profile (measured, time-split test period, top-10% review)

| fraud_type | Generator mechanism | Rule fired | Driving features | Recall |
|---|---|---|---|---|
| `duplicate` | Legit claim re-billed 1–3 days later, same trio | `r_duplicate` | `rule_hits` | **0.987** (76) |
| `phantom` | Ring providers bill big-ticket codes for random walk-ins | `r_rare_expensive` | `provider_code_prior_share` | **1.000** (120) |
| `upcoding` | Cheap service billed under MRISC/SURG01 at ring providers | — | `amount_vs_provider_avg`, `provider_code_prior_share` | **1.000** (93) |
| `excess_frequency` | 6–10 identical claims in a 14-day window | `r_member_freq` | `member_claims_30d` | **0.917** (72) |
| `collusion` | Ring provider + 3–5 loyal patients, steady mid-cost billing | — | provider/member history aggregates | **1.000** (68) |
| `none` (label noise) | 0.5% of legit flagged as fraud | — | — | 0.089 (56) — model correctly ignores noise |

**Reading the 1.000s:** ring fraud is concentrated in 6 of 120 providers, so
provider-identity features separate it trivially. These numbers measure the
benchmark's design as much as the model. The hard profile exists to remove that
shortcut; the graph layer's job is to win there.

## Hard profile (added mechanisms)

| fraud_type | Mechanism | Expected to break | Recall |
|---|---|---|---|
| `none_chronic` | ~200 members on dialysis/physio 2–3×/week — **legitimate** dense utilization | must NOT trip `r_member_freq`-driven fraud calls | TBD |
| `none_near_dup` | Legit follow-up/re-test re-bills within 3 days, labeled 0 | must NOT trip `r_duplicate` | TBD |
| `phantom_dispersed` | 1–3 big-ticket claims each, scattered across ~100 **non-ring** providers | kills `provider_code_prior_share` as a magic feature | TBD |
| `upcoding_dispersed` | 1–2 upcoded claims each, many non-ring providers | same | TBD |

Collusion remains ring-only in the hard profile — deliberately: it is the graph
layer's measurable target.

## Traps (legitimate lookalikes)

| Trap | Looks like | Why it's legitimate |
|---|---|---|
| Chronic care | excess frequency | Medically required cadence (dialysis 2–3×/week) |
| Follow-up re-bill | duplicate | Distinct service event within window |
| Tier-1 hospital claim | phantom/upcoding | High-cost code at 1.6× multiplier is *normal* for tier 1 |

A model that only learned "expensive + frequent = fraud" scores well on easy and
fails every trap. Per-mechanism recall makes that failure visible instead of
hiding it inside aggregate PR-AUC.

## Label noise

Applied after all mechanisms: 10% of fraud rows → `is_fraud=0` (missed),
0.5% of legit rows → `is_fraud=1` (false flags). Realistic; makes metrics honest.
