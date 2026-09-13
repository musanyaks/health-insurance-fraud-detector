# Architecture decision records

Short, dated records of the decisions that shape this system — including the
mistakes that forced them. Newest at the bottom.

---

## ADR-001: Time-based splits only, never random

**Context.** Fraud mechanisms produce near-duplicate rows. A random train/test
split puts twins on both sides and inflates every metric.
**Decision.** Always split on `claim_date` (config: `split.train_fraction`).
**Consequence.** Metrics reflect scoring *future* claims — the production task.
Slightly lower (honest) numbers vs. random splits.

## ADR-002: One feature builder, shared by training and serving

**Context.** Duplicated feature logic causes train/serve skew — the classic
production ML bug.
**Decision.** `fraud_engine/features/pipeline.py::build_features` is the only
definition. ETL and the API call it; nothing re-implements a feature.
**Consequence.** Changing a feature forces retraining awareness; enforced by
import structure, not discipline.

## ADR-003: Metrics = PR-AUC + recall@k, accuracy never reported

**Context.** ~4% fraud base rate. "Never fraud" achieves 96% accuracy.
ROC-AUC also flatters at low base rates.
**Decision.** Report PR-AUC and recall at 10% review capacity, plus
per-mechanism recall. `evaluation/metrics.py` contains no accuracy function.
**Consequence.** Numbers map directly to an investigator team's daily capacity.

## ADR-004: Composite score with policy override

**Context.** Investigators need a score *and* reasons, not a bare classifier
output. Pure ML scores can contradict fired rules (real case: P(fraud)=0.008
with `r_amount_outlier` fired → "Auto-approve" — indefensible).
**Decision.** score = 100 × (0.65·P + 0.35·rule_norm); any fired rule upgrades
"Auto-approve" to "Standard review queue". Score meaning stays stable; only the
action is overridden.
**Consequence.** No rule can be silently ignored; thresholds stay tunable.

## ADR-005: Native TreeSHAP (`pred_contribs`), not the `shap` package

**Context.** `shap` pulls matplotlib into the serving path and failed in a
Windows environment with a broken home dir. Values are identical — XGBoost
computes TreeSHAP natively.
**Decision.** Explanations come from `booster.predict(pred_contribs=True)`;
`shap` remains available for notebooks only.
**Consequence.** Smaller serving footprint, faster startup, fewer env failures.

## ADR-006: Serving-time row selection by `claim_id`, never by position

**Context.** Bug found via behavioral testing: appended claim was scored with
`tail(1)` after a date sort — history contained *later* claims, so the API
scored an arbitrary December claim (three different inputs → identical output).
**Decision.** `build_features_for_claim` selects by `claim_id`; a regression
test pins it, and an API test asserts different claims score differently.
**Consequence.** Correct under out-of-order dates; the bug class is dead.

## ADR-007: Synthetic data with a documented taxonomy

**Context.** Real claims data can't be published. Random synthetic data proves
nothing — the generator defines what is learnable.
**Decision.** Plant specific, documented mechanisms; publish the traceability
matrix; add label noise; report per-mechanism recall; state the concentration
limitation explicitly.
**Consequence.** Reproducible, publishable, and honest about what the metrics
do and don't demonstrate.

## ADR-008: Easy/hard generator profiles

**Context.** Ring-concentrated fraud makes provider-identity features a
shortcut (recalls of 1.000).
**Decision.** Freeze the easy profile as the reproducible benchmark; add a hard
profile (chronic members, legit near-duplicates, dispersed fraud) whose results
become the graph layer's baseline.
**Consequence.** A measurable before/after arc instead of one flattering table.
