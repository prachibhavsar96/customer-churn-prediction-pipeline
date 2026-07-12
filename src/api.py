from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

ONE_HOT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
INTERNET_DEPENDENT_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Populated on startup from the saved artifacts (see lifespan below).
artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts["model"] = joblib.load(MODEL_PATH)
    artifacts["feature_columns"] = joblib.load(FEATURE_COLUMNS_PATH)
    artifacts["scaler"] = joblib.load(SCALER_PATH)
    yield
    artifacts.clear()


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


class CustomerInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.70,
                "TotalCharges": 848.40,
            }
        }
    )

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]


def preprocess(customer: CustomerInput) -> pd.DataFrame:
    df = pd.DataFrame([customer.model_dump()])

    # Same collapsing feature_engineering.py applies: "No phone service"
    # and "No internet service" are fully determined by PhoneService/
    # InternetService already being "No", so they're redundant categories.
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    for col in INTERNET_DEPENDENT_COLS:
        df[col] = df[col].replace("No internet service", "No")

    # IMPORTANT: do not pass drop_first=True here. get_dummies() only
    # drops a category when multiple categories are present in the data
    # being encoded. A single incoming row has exactly one category per
    # column, so drop_first=True would drop it unconditionally and
    # zero out every dummy column regardless of the real value. Instead,
    # encode without dropping, then reindex to the exact training column
    # set: if this row's category is the one training dropped as the
    # reference level, its dummy column won't exist in feature_columns
    # and is correctly discarded here; every other category's dummy
    # column matches a training column and survives with value 1.
    df_encoded = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=False)

    feature_columns = artifacts["feature_columns"]
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    scaler = artifacts["scaler"]
    df_encoded[NUMERIC_COLS] = scaler.transform(df_encoded[NUMERIC_COLS])

    return df_encoded


def classify_risk(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.6:
        return "Medium"
    return "High"


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API. See /docs for interactive API documentation."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    model = artifacts["model"]
    X = preprocess(customer)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=probability,
        risk_level=classify_risk(probability),
    )
