from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"

def load_orders():
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["date"])
    orders["is_returned"] = orders.returned.eq("Yes")
    return orders

def money(amount):
    return f"INR {amount:,.0f}"

def overview(orders):
    return {
        "revenue": orders.revenue.sum(), "profit": orders.profit.sum(),
        "orders": len(orders), "return_rate": orders.is_returned.mean(),
    }

def revenue_by_region(orders):
    return orders.groupby("region", as_index=False).agg(revenue=("revenue","sum"), profit=("profit","sum"), orders=("order_id","count")).sort_values("revenue", ascending=False)

def product_performance(orders):
    return orders.groupby(["category","product"], as_index=False).agg(revenue=("revenue","sum"), profit=("profit","sum"), return_rate=("is_returned","mean"), orders=("order_id","count")).sort_values("revenue", ascending=False)

def return_rates(orders, period="all"):
    scoped = orders.copy()
    if period == "last_quarter":
        end = scoped.date.max()
        scoped = scoped[scoped.date >= end - pd.DateOffset(months=3)]
    return scoped.groupby("category", as_index=False).agg(orders=("order_id","count"), return_rate=("is_returned","mean")).sort_values("return_rate", ascending=False)

def electronics_trend(orders):
    scoped = orders[orders.category.eq("Electronics")].copy()
    scoped["quarter"] = scoped.date.dt.to_period("Q").astype(str)
    return scoped.groupby("quarter", as_index=False).agg(orders=("order_id","count"), return_rate=("is_returned","mean")).sort_values("quarter")

