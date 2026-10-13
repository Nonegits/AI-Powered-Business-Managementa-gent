# OrbitCart Business Operations Agent

OrbitCart is a complete portfolio project for a fictional Indian e-commerce company. It combines three sources of evidence in one application:

- **Structured operations data** for revenue, profits, returns, products, regions, and trends.
- **Company policy documents** retrieved with a local TF-IDF RAG index.
- **A churn model** trained on the same customer base to identify customers who need attention.

## Quick start

```powershell
python generate_data.py
python train_model.py
streamlit run app.py
```

The first command creates reproducible synthetic data in `data/`. The second trains `models/churn_model.joblib`. `app.py` will run both automatically when their output files are missing.

## Interactive interface

The Streamlit interface includes a filtered executive dashboard, sales analytics tables and charts, an agent chat area, direct policy search with evidence, and a downloadable churn-risk worklist. Use the left sidebar to filter dashboard results by date, region, and product category.

## Docker

Build and run the whole project in a container. The container creates the synthetic dataset and model automatically on its first request.

```powershell
docker build -t orbitcart-operations .
docker run --rm -p 8501:8501 orbitcart-operations
```

Then open `http://localhost:8501`.

## What to ask

- `Which category had the highest return rate in the last quarter?`
- `What is the return policy for electronics?`
- `Why might electronics returns be increasing?`
- `Which customers are at high risk of churn?`
- `Show revenue by region`
- `What are the best and worst products?`

## Project layout

```
app.py                 Streamlit interface
agent.py               Intent router that combines analytics, RAG, and ML
analytics.py           Business metric queries
rag.py                 Local document retrieval with cited sources
train_model.py         Customer churn training pipeline
generate_data.py       Reproducible synthetic source data
data/                  Generated CSV data
company_documents/     Fictional company policies for RAG
models/                Generated trained model
```

## Design notes

The project deliberately keeps its datasets fictional and reproducible so it can be run without downloading data or exposing customer information. The RAG system is local and transparent: answers include the policy document that supplied the evidence. The churn score is an operational prioritization aid, not an automatic decision.

