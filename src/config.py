"""Central configuration: paths, seeds, column groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"

ID_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
