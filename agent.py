from pathlib import Path
import joblib
import pandas as pd
from analytics import load_orders, money, overview, revenue_by_region, product_performance, return_rates, electronics_trend
from rag import PolicyRetriever

ROOT = Path(__file__).parent

class OperationsAgent:
    def __init__(self):
        self.orders = load_orders()
        self.customers = pd.read_csv(ROOT / "data" / "customers.csv")
        saved = joblib.load(ROOT / "models" / "churn_model.joblib")
        self.model, self.model_auc = saved["model"], saved["auc"]
        self.retriever = PolicyRetriever()
    def answer(self, question):
        q = question.lower()
        if any(word in q for word in ["churn", "leaving", "at risk"]):
            return self._churn()
        if "return policy" in q or ("policy" in q and "return" in q):
            return self._policy(question)
        if "increasing" in q and ("electronics" in q or "return" in q):
            return self._returns_with_policy(question)
        if "highest return" in q or "return rate" in q:
            return self._return_rates()
        if "region" in q:
            return self._region()
        if "best" in q or "worst" in q or "product" in q:
            return self._products()
        return self._overview()

    def _overview(self):
        k = overview(self.orders)
        return {"answer": f"OrbitCart recorded {money(k['revenue'])} revenue and {money(k['profit'])} profit across {k['orders']:,} orders. Overall return rate is {k['return_rate']:.1%}.", "tables": {"Business snapshot": pd.DataFrame([k])}, "sources": []}

    def _region(self):
        table = revenue_by_region(self.orders)
        leader = table.iloc[0]
        return {"answer": f"{leader.region} leads revenue at {money(leader.revenue)} from {leader.orders:,} orders.", "tables": {"Revenue by region": table}, "sources": []}

    def _products(self):
        table = product_performance(self.orders)
        leader = table.iloc[0]
        return {"answer": f"{leader['product']} is the top product by revenue at {money(leader['revenue'])}. The table also shows return rates and profit for every product.", "tables": {"Product performance": table}, "sources": []}

    def _return_rates(self):
        table = return_rates(self.orders, "last_quarter")
        leader = table.iloc[0]
        return {"answer": f"In the latest quarter, {leader.category} had the highest return rate at {leader.return_rate:.1%} across {leader.orders:,} orders.", "tables": {"Latest-quarter return rate": table}, "sources": []}

    def _policy(self, question):
        sources = self.retriever.search(question)
        text = sources[0]["text"] if sources else "No matching policy section was found."
        return {"answer": text, "tables": {}, "sources": sources}

    def _returns_with_policy(self, question):
        trend, rates = electronics_trend(self.orders), return_rates(self.orders, "last_quarter")
        evidence = self.retriever.search("electronics return policy damaged item opened packaging")
        first, last = trend.iloc[0], trend.iloc[-1]
        answer = (f"Electronics has the latest-quarter's highest return rate ({rates.iloc[0].return_rate:.1%}). "
                  f"Its return rate moved from {first.return_rate:.1%} in {first.quarter} to {last.return_rate:.1%} in {last.quarter}. "
                  "The policy evidence shows that electronics are eligible for a short inspection window, so review defect codes, listing accuracy, and fulfillment damage before changing the policy.")
        return {"answer": answer, "tables": {"Electronics return trend": trend, "Category return rates": rates}, "sources": evidence}

    def _churn(self):
        features = ["age","city","tenure_days","total_orders","avg_order_value","days_since_last_purchase","discount_usage","return_frequency"]
        scored = self.customers.copy()
        scored["churn_risk"] = self.model.predict_proba(scored[features])[:, 1]
        high = scored.sort_values("churn_risk", ascending=False).head(20)
        return {"answer": f"The model identified {int((scored.churn_risk >= .60).sum()):,} high-risk customers (risk >= 60%). The list below prioritizes the top 20. Holdout ROC-AUC: {self.model_auc:.2f}.", "tables": {"Highest churn-risk customers": high[["customer_id","city","customer_segment","total_orders","avg_order_value","days_since_last_purchase","return_frequency","churn_risk"]]}, "sources": []}

