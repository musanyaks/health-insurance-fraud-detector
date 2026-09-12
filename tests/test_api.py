from fastapi.testclient import TestClient

from api.deps import get_history, get_model
from api.main import app


def make_client(claims_df, small_model) -> TestClient:
    app.dependency_overrides[get_model] = lambda: small_model
    app.dependency_overrides[get_history] = lambda: claims_df
    return TestClient(app)


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_score_endpoint(claims_df, small_model):
    client = make_client(claims_df, small_model)
    row = claims_df.iloc[100]
    payload = {
        "claim_id": "CLMTEST001", "member_id": row["member_id"],
        "provider_id": row["provider_id"], "code": row["code"],
        "claim_amount": float(row["claim_amount"]),
        "claim_date": str(row["claim_date"].date()),
        "member_age": int(row["member_age"]),
    }
    r = client.post("/v1/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["claim_id"] == "CLMTEST001"
    assert 0 <= body["fraud_risk_score"] <= 100
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["reasons"], list) and body["reasons"]


def test_validation_error(claims_df, small_model):
    client = make_client(claims_df, small_model)
    r = client.post("/v1/score", json={"claim_id": "X"})  # missing required fields
    assert r.status_code == 422

def test_score_distinguishes_claims(claims_df, small_model):
    client = make_client(claims_df, small_model)
    row = claims_df.iloc[100]
    base = {"member_id": row["member_id"], "provider_id": row["provider_id"],
            "claim_date": "2024-12-05"}
    cheap = client.post("/v1/score", json={**base, "claim_id": "A1", "code": "CONS",
                                           "claim_amount": 1500, "member_age": 40}).json()
    big = client.post("/v1/score", json={**base, "claim_id": "A2", "code": "MRISC",
                                         "claim_amount": 200000, "member_age": 40}).json()
    assert cheap["fraud_risk_score"] != big["fraud_risk_score"]