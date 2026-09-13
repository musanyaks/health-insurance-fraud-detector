# API documentation

Base URL: `http://localhost:8000` (or `:8010` if 8000 is occupied).
Interactive docs: `/docs` (Swagger UI). Version: **v1** (path-prefixed, no auth — MVP).

## `GET /health`

```json
{ "status": "ok" }
```

## `POST /v1/score`

Scores one claim against the full claim history (member + provider context).

### Request body

| Field | Type | Required | Constraints |
|---|---|---|---|
| `claim_id` | string | ✅ | unique id of the claim being scored |
| `member_id` | string | ✅ | |
| `provider_id` | string | ✅ | |
| `code` | string | ✅ | procedure code (catalog: data_dictionary.md) |
| `claim_amount` | number | ✅ | > 0 |
| `claim_date` | date | ✅ | ISO `YYYY-MM-DD` |
| `member_age` | int | ❌ | 0–120; defaults to 35 if omitted |

Unknown `member_id`/`provider_id` are **valid**: context features fall back to
first-claim neutral values (prior-avg = own amount → ratio 1.0; history-gated
rules won't fire). This mirrors a genuinely new patient/provider.

### Response 200

```json
{
  "claim_id": "CLMTEST001",
  "fraud_risk_score": 12,
  "risk_level": "LOW",
  "ml_probability": 0.008,
  "rule_hits": ["r_amount_outlier"],
  "reasons": [
    "Claim amount far above this provider's historical average",
    "Provider's historical average claim amount",
    "Procedure unusual for this provider's history",
    "Claim amount is an extreme multiple of provider average"
  ],
  "recommended_action": "Standard review queue"
}
```

### Score composition

```
score = 100 × (0.65 × P(fraud|model) + 0.35 × min(rule_hits, 3) / 3)
```

| Level | Range | Default action |
|---|---|---|
| HIGH | ≥ 70 | Assign to senior investigator |
| MEDIUM | ≥ 40 | Standard review queue |
| LOW | < 40 | Auto-approve |

**Policy override:** if any rule fired, `Auto-approve` is upgraded to
`Standard review queue` regardless of score — a fired rule is never ignored.

`reasons` = fired-rule explanations first, then top-3 TreeSHAP feature
contributions (XGBoost native `pred_contribs`), deduplicated, max 4.

### Rule catalog (what `rule_hits` can contain)

| Key | Investigator text |
|---|---|
| `r_duplicate` | Duplicate procedure: same patient/provider/procedure within 7 days |
| `r_amount_outlier` | Claim amount far above this provider's historical average |
| `r_member_freq` | Patient has an excessive number of claims in the last 30 days |
| `r_rare_expensive` | Provider rarely bills this high-cost procedure |

### Errors

| Status | Cause |
|---|---|
| 422 | Validation: missing field, `claim_amount ≤ 0`, bad date format, age out of range |
| 500 | Model/feature failure — traceback in server logs |

### Operational notes

- History and model are loaded **once** at startup (`lru_cache`) — restart the
  server after regenerating data or retraining.
- History context is rebuilt per request. Correct at MVP scale; a precomputed
  feature store is the production answer (README → limitations).

### Example (curl)

```bash
curl -s -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"CLMTEST001","member_id":"MBR00123","provider_id":"PRV0042",
       "code":"MRISC","claim_amount":72000,"claim_date":"2024-12-05"}'
```
