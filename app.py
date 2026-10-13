from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent


def ensure_assets():
    if not (ROOT / "data" / "orders.csv").exists():
        subprocess.run([sys.executable, "generate_data.py"], cwd=ROOT, check=True)
    if not (ROOT / "models" / "churn_model.joblib").exists():
        subprocess.run([sys.executable, "train_model.py"], cwd=ROOT, check=True)


ensure_assets()
from agent import OperationsAgent
from analytics import money, product_performance, revenue_by_region, return_rates

st.set_page_config(page_title="OrbitCart Operations", page_icon="Cart", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e8edf3; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(25,50,77,.05);}
.eyebrow {color: #637083; font-size: .82rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .3rem;}
h1 {letter-spacing: -.035em;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return OperationsAgent()


agent = get_agent()
orders = agent.orders.copy()
customers = agent.customers.copy()

with st.sidebar:
    st.title("OrbitCart")
    st.caption("Business operations workspace")
    st.divider()
    st.subheader("Dashboard filters")
    min_date, max_date = orders.date.min().date(), orders.date.max().date()
    selected_dates = st.date_input("Order period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    regions = st.multiselect("Regions", sorted(orders.region.unique()), default=sorted(orders.region.unique()))
    categories = st.multiselect("Categories", sorted(orders.category.unique()), default=sorted(orders.category.unique()))
    st.divider()
    st.caption("All data is fictional and generated locally.")

if len(selected_dates) == 2:
    start_date, end_date = map(pd.Timestamp, selected_dates)
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
filtered = orders[orders.date.between(start_date, end_date) & orders.region.isin(regions) & orders.category.isin(categories)].copy()

st.markdown('<p class="eyebrow">Operations intelligence</p>', unsafe_allow_html=True)
st.title("AI Business Operations Agent")
st.caption("Explore sales performance, retrieve company policies, and prioritize retention work from a single interface.")

dashboard, analytics_tab, assistant_tab, policies_tab, retention_tab = st.tabs(["Dashboard", "Sales analytics", "Ask the agent", "Policy search", "Customer retention"])

with dashboard:
    if filtered.empty:
        st.warning("No orders match these filters. Adjust the date, region, or category selection.")
    else:
        total_revenue, total_profit = filtered.revenue.sum(), filtered.profit.sum()
        metrics = st.columns(4)
        metrics[0].metric("Revenue", money(total_revenue))
        metrics[1].metric("Profit", money(total_profit), f"{total_profit / total_revenue:.1%} margin")
        metrics[2].metric("Orders", f"{len(filtered):,}")
        metrics[3].metric("Return rate", f"{filtered.returned.eq('Yes').mean():.1%}", f"Avg order {money(filtered.revenue.mean())}")
        left, right = st.columns([1.35, 1])
        monthly = filtered.assign(month=filtered.date.dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False).agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        with left:
            st.subheader("Revenue and profit trend")
            st.line_chart(monthly.set_index("month")[["revenue", "profit"]], use_container_width=True)
        with right:
            st.subheader("Revenue by category")
            st.bar_chart(filtered.groupby("category").revenue.sum().sort_values(ascending=False), horizontal=True, use_container_width=True)
        st.subheader("Recent order activity")
        st.dataframe(filtered.sort_values("date", ascending=False)[["order_id", "date", "product", "category", "region", "revenue", "profit", "returned"]].head(15), use_container_width=True, hide_index=True)
        st.download_button("Download filtered orders (CSV)", filtered.to_csv(index=False).encode("utf-8"), "orbitcart_filtered_orders.csv", "text/csv")

with analytics_tab:
    st.subheader("Product and regional performance")
    if filtered.empty:
        st.info("Choose at least one region and category to view analytics.")
    else:
        products, regions_table, returns = product_performance(filtered), revenue_by_region(filtered), return_rates(filtered)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Products")
            st.dataframe(products.style.format({"revenue": "INR {:,.0f}", "profit": "INR {:,.0f}", "return_rate": "{:.1%}"}), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("#### Regions")
            st.dataframe(regions_table.style.format({"revenue": "INR {:,.0f}", "profit": "INR {:,.0f}"}), use_container_width=True, hide_index=True)
        st.markdown("#### Return rate by category")
        st.bar_chart(returns.set_index("category")["return_rate"], horizontal=True, use_container_width=True)

with assistant_tab:
    st.subheader("Ask OrbitCart")
    st.caption("The agent chooses analytics, policy retrieval, or churn prediction based on your question.")
    examples = ["Which category had the highest return rate in the last quarter?", "What is the return policy for electronics?", "Why might electronics returns be increasing?", "Which customers are at high risk of leaving?", "Show revenue by region", "What are the best and worst products?"]
    question = st.text_input("Ask a business question", placeholder="Example: Why might electronics returns be increasing?")
    example = st.selectbox("Or select a starter question", ["Choose an example"] + examples)
    active_question = question.strip() or (example if example != "Choose an example" else "")
    if active_question:
        result = agent.answer(active_question)
        st.markdown("#### Answer")
        st.write(result["answer"])
        for name, table in result["tables"].items():
            st.markdown(f"#### {name}")
            st.dataframe(table, use_container_width=True, hide_index=True)
        if result["sources"]:
            st.markdown("#### Policy evidence")
            for source in result["sources"]:
                with st.expander(f"{source['source']} · relevance {source['score']:.0%}"):
                    st.write(source["text"])

with policies_tab:
    st.subheader("Company policy search")
    policy_query = st.text_input("Search policies", placeholder="electronics returns, shipping damage, discounts...")
    if policy_query:
        matches = agent.retriever.search(policy_query)
        if not matches:
            st.info("No policy section matched that search. Try different wording.")
        for item in matches:
            with st.expander(f"{item['source']} · relevance {item['score']:.0%}", expanded=True):
                st.write(item["text"])
    else:
        st.info("Search the fictional return, shipping, discount, support, and FAQ documents.")

with retention_tab:
    st.subheader("Customer churn worklist")
    st.caption(f"The Random Forest model achieved {agent.model_auc:.2f} ROC-AUC on a held-out sample of the fictional customer data.")
    threshold = st.slider("Minimum churn-risk score", min_value=0.30, max_value=0.90, value=0.60, step=0.05)
    city_filter = st.multiselect("Customer cities", sorted(customers.city.unique()), default=sorted(customers.city.unique()))
    features = ["age", "city", "tenure_days", "total_orders", "avg_order_value", "days_since_last_purchase", "discount_usage", "return_frequency"]
    scored = customers.copy()
    scored["churn_risk"] = agent.model.predict_proba(scored[features])[:, 1]
    worklist = scored[(scored.churn_risk >= threshold) & scored.city.isin(city_filter)].sort_values("churn_risk", ascending=False)
    one, two, three = st.columns(3)
    one.metric("Customers in worklist", f"{len(worklist):,}")
    two.metric("Average risk", f"{worklist.churn_risk.mean():.1%}" if len(worklist) else "—")
    three.metric("Highest risk", f"{worklist.churn_risk.max():.1%}" if len(worklist) else "—")
    st.dataframe(worklist[["customer_id", "city", "customer_segment", "total_orders", "avg_order_value", "days_since_last_purchase", "return_frequency", "churn_risk"]].style.format({"avg_order_value": "INR {:,.0f}", "return_frequency": "{:.1%}", "churn_risk": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.download_button("Download churn worklist (CSV)", worklist.to_csv(index=False).encode("utf-8"), "orbitcart_churn_worklist.csv", "text/csv")
