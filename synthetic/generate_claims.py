"""Synthetic Kenyan health-claims generator with planted, documented fraud mechanisms.

Each fraud_type below corresponds to one row in docs/fraud_taxonomy.md.
Deterministic given --seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COUNTIES = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Nyeri", "Machakos"]

PROCEDURES = pd.DataFrame({
    "code": ["CONS", "LAB01", "LAB02", "RAD01", "PHARM",
             "SURG01", "SURG02", "MRISC", "CTSCN", "WARD3"],
    "desc": ["Consultation", "Basic lab", "Advanced lab", "X-ray", "Pharmacy",
             "Minor surgery", "Major surgery", "MRI", "CT scan", "Ward, 3 days"],
    "base_cost": [1500, 3500, 9000, 6000, 2500, 45000, 180000, 65000, 38000, 30000],
})
CODE_P = [0.30, 0.15, 0.05, 0.08, 0.20, 0.07, 0.02, 0.02, 0.03, 0.08]


def generate_claims(n_members=5000, n_providers=120, n_legit=60000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    START = pd.Timestamp("2024-01-01")

    providers = pd.DataFrame({
        "provider_id": [f"PRV{i:04d}" for i in range(n_providers)],
        "county": rng.choice(COUNTIES, n_providers),
        "tier": rng.choice([1, 2, 3], n_providers, p=[0.25, 0.50, 0.25]),
    })
    providers["cost_mult"] = providers["tier"].map({1: 1.6, 2: 1.0, 3: 0.7})
    fraud_providers = rng.choice(providers["provider_id"], max(6, n_providers // 20), replace=False)

    members = pd.DataFrame({
        "member_id": [f"MBR{i:05d}" for i in range(n_members)],
        "member_age": rng.integers(1, 90, n_members),
        "gender": rng.choice(["M", "F"], n_members),
        "county": rng.choice(COUNTIES, n_members),
    })

    _id = iter(range(10 ** 7))
    def new_ids(n): return [f"CLM{next(_id):07d}" for _ in range(n)]
    def random_dates(n): return START + pd.to_timedelta(rng.integers(0, 365, n), "D")

    base = PROCEDURES.set_index("code")["base_cost"]
    def price(codes, provs):
        b = base.loc[list(codes)].to_numpy()
        m = providers.set_index("provider_id").loc[list(provs), "cost_mult"].to_numpy()
        return (b * m * rng.lognormal(0, 0.25, len(codes))).round(-2)

    def assemble(n, codes, mems, provs, dates, fraud, ftype):
        return pd.DataFrame({
            "claim_id": new_ids(n),
            "member_id": list(mems), "provider_id": list(provs), "code": list(codes),
            "claim_date": pd.to_datetime(dates),
            "claim_amount": price(codes, provs),
            "is_fraud": fraud, "fraud_type": ftype,
        })

    blocks = []

    # ---- legitimate claims ----
    n = n_legit
    blocks.append(assemble(n, rng.choice(PROCEDURES["code"], n, p=CODE_P),
                           rng.choice(members["member_id"], n),
                           rng.choice(providers["provider_id"], n),
                           random_dates(n), 0, "none"))
    legit = blocks[0]

    # ---- mechanism 1: duplicates (re-billed 1-3 days later) ----
    src = legit.sample(400, random_state=1)
    blocks.append(assemble(400, src["code"], src["member_id"], src["provider_id"],
                           src["claim_date"] + pd.to_timedelta(rng.integers(1, 4, 400), "D"),
                           1, "duplicate"))

    # ---- mechanism 2: phantom procedures at ring providers ----
    n = 700
    blocks.append(assemble(n, rng.choice(["SURG02", "MRISC", "WARD3"], n),
                           rng.choice(members["member_id"], n),
                           rng.choice(fraud_providers, n), random_dates(n), 1, "phantom"))

    # ---- mechanism 3: upcoding (cheap service billed as expensive code) ----
    src = legit[legit["provider_id"].isin(fraud_providers)
                & legit["code"].isin(["CONS", "PHARM", "LAB01"])].sample(500, random_state=2)
    blocks.append(assemble(500, rng.choice(["MRISC", "SURG01"], 500, p=[0.6, 0.4]),
                           src["member_id"], src["provider_id"], src["claim_date"], 1, "upcoding"))

    # ---- mechanism 4: excess frequency (6-10 identical claims in 2 weeks) ----
    rows = []
    for _ in range(60):
        m, p = rng.choice(members["member_id"]), rng.choice(providers["provider_id"])
        c = rng.choice(["PHARM", "LAB01", "CONS"])
        d0 = int(rng.integers(0, 350))
        for _ in range(int(rng.integers(6, 11))):
            rows.append((m, p, c, START + pd.Timedelta(days=d0 + int(rng.integers(0, 15)))))
    freq = pd.DataFrame(rows, columns=["member_id", "provider_id", "code", "claim_date"])
    blocks.append(assemble(len(freq), freq["code"], freq["member_id"], freq["provider_id"],
                           freq["claim_date"], 1, "excess_frequency"))

    # ---- mechanism 5: collusion (ring provider + loyal patients) ----
    rows = []
    for _ in range(8):
        p = rng.choice(fraud_providers)
        for m in rng.choice(members["member_id"], int(rng.integers(3, 6))):
            for _ in range(int(rng.integers(8, 15))):
                rows.append((m, p, rng.choice(["PHARM", "LAB02", "RAD01"]),
                             START + pd.Timedelta(days=int(rng.integers(0, 365)))))
    coll = pd.DataFrame(rows, columns=["member_id", "provider_id", "code", "claim_date"])
    blocks.append(assemble(len(coll), coll["code"], coll["member_id"], coll["provider_id"],
                           coll["claim_date"], 1, "collusion"))

    df = pd.concat(blocks, ignore_index=True)

    # ---- label noise: 10% missed fraud, 0.5% false flags ----
    fraud_idx = df.index[df["is_fraud"] == 1].to_numpy()
    neg_idx = df.index[df["is_fraud"] == 0].to_numpy()
    df.loc[rng.choice(fraud_idx, int(0.10 * len(fraud_idx)), replace=False), "is_fraud"] = 0
    df.loc[rng.choice(neg_idx, int(0.005 * len(neg_idx)), replace=False), "is_fraud"] = 1

    df = df.merge(members[["member_id", "member_age"]], on="member_id", how="left")
    return df.sort_values("claim_date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5000)
    ap.add_argument("--providers", type=int, default=120)
    ap.add_argument("--legit", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-out", default="data/raw/claims.parquet")
    ap.add_argument("--sample-out", default="data/sample/sample_claims.parquet")
    ap.add_argument("--sample-rows", type=int, default=5000)
    a = ap.parse_args()

    df = generate_claims(a.members, a.providers, a.legit, a.seed)

    for out in (a.raw_out, a.sample_out):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.raw_out, index=False)
    df.sample(n=min(a.sample_rows, len(df)), random_state=0).to_parquet(a.sample_out, index=False)

    print(f"wrote {len(df):,} claims -> {a.raw_out}")
    print(df["is_fraud"].value_counts(normalize=True).round(4))
    print(df.groupby("fraud_type").size())


if __name__ == "__main__":
    main()