"""Generate a deterministic synthetic e-commerce dataset with a planted story.

Ground truth baked into the data (used by the evaluation suite):
  * Revenue in 2025-Q4 declines versus 2025-Q3.
  * The decline is concentrated in the SMB segment in the West region, whose
    order volume drops by ~55% in Q4 2025.
  * Electronics discounts jump from ~8% to ~22% in Q4 2025, lowering revenue
    per order in that category.
  * ``days_to_ship`` has ~3% missing values; a few extreme ``quantity`` outliers.

Usage:  python sample_data/generate_ecommerce.py [output.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = {"Enterprise": 0.2, "SMB": 0.35, "Consumer": 0.45}
REGIONS = {"North": 0.25, "South": 0.2, "East": 0.25, "West": 0.3}
CHANNELS = {"Web": 0.5, "Mobile": 0.35, "Partner": 0.15}
CATEGORIES = {
    "Electronics": (0.25, 320.0),
    "Home": (0.22, 95.0),
    "Apparel": (0.23, 55.0),
    "Sports": (0.15, 80.0),
    "Beauty": (0.15, 35.0),
}
SEGMENT_QTY = {"Enterprise": 6.0, "SMB": 3.0, "Consumer": 1.6}


def generate(seed: int = 42, n_customers: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "customer_segment": rng.choice(list(SEGMENTS), n_customers, p=list(SEGMENTS.values())),
        "region": rng.choice(list(REGIONS), n_customers, p=list(REGIONS.values())),
    })
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    records = []
    order_id = 100000
    cat_names = list(CATEGORIES)
    cat_p = np.array([v[0] for v in CATEGORIES.values()])
    for day in days:
        q4_2025 = day >= pd.Timestamp("2025-10-01")
        base = 26 + 4 * np.sin(2 * np.pi * day.dayofyear / 365.0) + (day.year - 2024) * 2
        if day.dayofweek >= 5:
            base *= 0.85
        n = rng.poisson(base)
        if n == 0:
            continue
        cust = customers.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        if q4_2025:
            smb_west = (cust["customer_segment"] == "SMB") & (cust["region"] == "West")
            drop = smb_west & (rng.random(n) < 0.55)
            cust = cust[~drop]
        for _, c in cust.iterrows():
            cat = cat_names[rng.choice(len(cat_names), p=cat_p)]
            base_price = CATEGORIES[cat][1]
            unit_price = round(float(base_price * rng.lognormal(0, 0.35)), 2)
            qty = int(max(1, rng.poisson(SEGMENT_QTY[c["customer_segment"]])))
            if rng.random() < 0.002:
                qty *= 8  # bulk-order outliers
            disc_mean = 0.08
            if cat == "Electronics" and q4_2025:
                disc_mean = 0.22
            discount = float(np.clip(rng.normal(disc_mean, 0.03), 0, 0.6))
            revenue = round(unit_price * qty * (1 - discount), 2)
            returned = bool(rng.random() < (0.09 if cat == "Apparel" else 0.04))
            ship = float(max(1, rng.normal(3.5 if c["region"] != "West" else 4.2, 1.2)))
            records.append({
                "order_id": order_id,
                "order_date": day.date().isoformat(),
                "customer_id": int(c["customer_id"]),
                "customer_segment": c["customer_segment"],
                "region": c["region"],
                "channel": rng.choice(list(CHANNELS), p=list(CHANNELS.values())),
                "product_category": cat,
                "quantity": qty,
                "unit_price": unit_price,
                "discount_pct": round(discount, 4),
                "revenue": revenue,
                "is_returned": returned,
                "days_to_ship": round(ship, 1),
            })
            order_id += 1
    df = pd.DataFrame.from_records(records)
    missing = rng.random(len(df)) < 0.03
    df.loc[missing, "days_to_ship"] = np.nan
    return df


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("ecommerce_orders.csv")
    df = generate()
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
