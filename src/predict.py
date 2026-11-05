"""Load the saved pipeline and score new (raw) customers.

    python -m src.predict
"""
from __future__ import annotations

import joblib
import pandas as pd

from .config import MODELS_DIR


def load_model(path=MODELS_DIR / "best_model.joblib"):
    return joblib.load(path)


def predict_churn(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Return churn probability and label for raw rows (no manual preprocessing)."""
    model = model or load_model()
    proba = model.predict_proba(df)[:, 1]
    return pd.DataFrame({"churn_probability": proba.round(4), "churn_prediction": (proba >= 0.5).astype(int)},
                        index=df.index)


if __name__ == "__main__":
    sample = pd.DataFrame([
        {  # high-risk profile: new, month-to-month, fibre, electronic check
            "customerID": "demo-1", "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": "160.0",
        },
        {  # low-risk profile: long tenure, two-year contract
            "customerID": "demo-2", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0, "TotalCharges": "3900.0",
        },
    ])
    print(predict_churn(sample).assign(customer=sample["customerID"]).to_string(index=False))
