from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
FEATURES = ["age","city","tenure_days","total_orders","avg_order_value","days_since_last_purchase","discount_usage","return_frequency"]

def train():
    customers = pd.read_csv(ROOT / "data" / "customers.csv")
    X, y = customers[FEATURES], customers.churned
    categorical = ["city"]
    preprocessor = ColumnTransformer([("city", OneHotEncoder(handle_unknown="ignore"), categorical)], remainder="passthrough")
    model = Pipeline([("prepare", preprocessor), ("forest", RandomForestClassifier(n_estimators=250, min_samples_leaf=4, random_state=42, class_weight="balanced"))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "auc": auc}, ROOT / "models" / "churn_model.joblib")
    print(f"Model trained. Holdout ROC-AUC: {auc:.3f}")

if __name__ == "__main__":
    train()

