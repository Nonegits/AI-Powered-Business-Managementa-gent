"""Create reproducible fictional OrbitCart order and customer data."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RNG = np.random.default_rng(42)

PRODUCTS = [
    ("AstraBook 14", "Electronics", 62000, 0.16),
    ("OrbitPhone X", "Electronics", 28500, 0.18),
    ("Nova Earbuds", "Electronics", 3200, 0.24),
    ("TrailRunner Shoes", "Fashion", 3100, 0.09),
    ("Linen Everyday Shirt", "Fashion", 1450, 0.07),
    ("CloudFit Jacket", "Fashion", 4200, 0.10),
    ("ErgoTask Chair", "Home & Living", 8900, 0.08),
    ("PureSteel Bottle", "Home & Living", 800, 0.04),
    ("GlowDesk Lamp", "Home & Living", 1850, 0.06),
    ("Kitchen Pro Set", "Home & Living", 3600, 0.05),
]
CITIES = [("Mumbai", "West"), ("Pune", "West"), ("Delhi", "North"), ("Jaipur", "North"),
          ("Bengaluru", "South"), ("Chennai", "South"), ("Kolkata", "East"), ("Bhubaneswar", "East")]

def create_data():
    DATA.mkdir(exist_ok=True)
    n_customers, n_orders = 850, 5200
    customer_ids = [f"C{i:04d}" for i in range(1001, 1001 + n_customers)]
    cities = RNG.choice(len(CITIES), n_customers)
    signup_days = RNG.integers(60, 900, n_customers)
    signup_dates = pd.Timestamp("2025-12-31") - pd.to_timedelta(signup_days, unit="D")
    ages = RNG.integers(20, 66, n_customers)
    genders = RNG.choice(["Female", "Male", "Non-binary"], n_customers, p=[.47, .48, .05])

    # Customers with an older last purchase and lower engagement receive a higher churn probability.
    engagement = RNG.beta(2.3, 2.0, n_customers)
    customer_lookup = pd.DataFrame({
        "customer_id": customer_ids, "age": ages, "gender": genders,
        "city": [CITIES[i][0] for i in cities], "signup_date": signup_dates,
        "engagement": engagement,
    })

    chosen_customers = RNG.choice(n_customers, n_orders, p=(engagement + .15) / (engagement + .15).sum())
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(RNG.integers(0, 731, n_orders), unit="D")
    chosen_products = RNG.choice(len(PRODUCTS), n_orders, p=[.08,.12,.16,.13,.12,.07,.08,.10,.08,.06])
    rows = []
    for i, (c_idx, date, product_idx) in enumerate(zip(chosen_customers, dates, chosen_products), start=10001):
        product, category, base_price, base_return = PRODUCTS[product_idx]
        quantity = int(RNG.choice([1,2,3], p=[.72,.22,.06]))
        discount = float(RNG.choice([0, .05, .10, .15, .20], p=[.30,.27,.23,.13,.07]))
        # Electronics return pressure rises in 2025, giving the agent an explainable pattern to find.
        electronics_uplift = .075 if category == "Electronics" and date >= pd.Timestamp("2025-07-01") else 0
        returned = RNG.random() < base_return + electronics_uplift + (0.025 if discount >= .15 else 0)
        city, region = CITIES[cities[c_idx]]
        unit_price = round(base_price * RNG.uniform(.91, 1.09), 2)
        cost = round(unit_price * RNG.uniform(.57, .76), 2)
        rows.append([f"O{i}", date.date(), customer_ids[c_idx], product, category, quantity,
                     unit_price, discount, city, region, RNG.choice(["UPI","Card","COD","Wallet"], p=[.39,.37,.15,.09]),
                     "Yes" if returned else "No", cost])
    orders = pd.DataFrame(rows, columns=["order_id","date","customer_id","product","category","quantity",
        "price","discount","city","region","payment","returned","unit_cost"])
    orders["date"] = pd.to_datetime(orders["date"])
    orders["revenue"] = (orders.quantity * orders.price * (1 - orders.discount)).round(2)
    orders["profit"] = (orders.revenue - orders.quantity * orders.unit_cost).round(2)
    orders.to_csv(DATA / "orders.csv", index=False)

    summary = orders.groupby("customer_id").agg(
        total_orders=("order_id", "count"), avg_order_value=("revenue", "mean"),
        last_purchase=("date", "max"), return_frequency=("returned", lambda x: (x == "Yes").mean()),
        discount_usage=("discount", "mean")
    ).reindex(customer_ids).fillna({"total_orders":0,"avg_order_value":0,"return_frequency":0,"discount_usage":0}).reset_index()
    customers = customer_lookup.drop(columns="engagement").merge(summary, on="customer_id", how="left")
    customers["last_purchase"] = pd.to_datetime(customers["last_purchase"]).fillna(pd.Timestamp("2024-01-01"))
    customers["days_since_last_purchase"] = (pd.Timestamp("2025-12-31") - customers.last_purchase).dt.days
    customers["tenure_days"] = (pd.Timestamp("2025-12-31") - customers.signup_date).dt.days
    raw_risk = (-2.2 + .018 * customers.days_since_last_purchase - .12 * customers.total_orders
                + 1.8 * customers.return_frequency - .8 * customers.discount_usage)
    probability = 1 / (1 + np.exp(-raw_risk))
    customers["churned"] = (RNG.random(len(customers)) < probability).astype(int)
    customers["customer_segment"] = pd.cut(customers.avg_order_value, [-1, 1500, 5000, np.inf], labels=["Value", "Core", "Premium"]).astype(str)
    customers.drop(columns="last_purchase").to_csv(DATA / "customers.csv", index=False)
    print(f"Created {len(orders):,} orders and {len(customers):,} customers in {DATA}")

if __name__ == "__main__":
    create_data()

