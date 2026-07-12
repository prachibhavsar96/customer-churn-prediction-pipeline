# Customer Churn Prediction

Predicts which telecom customers are likely to churn, using a scikit-learn model served behind a FastAPI endpoint.

[![CI](https://github.com/prachibhavsar96/customer-churn-prediction-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/prachibhavsar96/customer-churn-prediction-pipeline/actions)

## Problem Statement

Customer churn — customers cancelling their service — is a direct revenue loss for telecom providers, and it's far cheaper to retain an at-risk customer than to acquire a new one. This project builds a model that predicts, from a customer's account and service attributes, whether they are likely to churn, so retention efforts can be targeted at the customers who need them most.

## Dataset

- **Source**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle / IBM sample dataset)
- **Size**: 7,043 customer records, 21 columns
- **Class balance**: 26.5% churn ("Yes") vs. 73.5% no-churn ("No") — moderately imbalanced
- **Known data quirk**: `TotalCharges` contains 11 blank strings (not `NaN`) for customers with `tenure == 0`, i.e. brand-new customers with no billing history yet

## Architecture Overview

```
data/raw (CSV)
    -> src/eda.py                  exploratory analysis, no side effects
    -> src/feature_engineering.py  cleaning, category collapsing, one-hot encoding,
                                    train/test split, scaling -> data/processed/
    -> src/train_baseline.py       Logistic Regression, logged to MLflow
    -> src/train_compare.py        Random Forest + XGBoost, logged to MLflow
    -> src/train_tuned.py          RandomizedSearchCV tuning of RF/XGBoost, logged to MLflow
    -> src/save_final_model.py     retrains + persists the selected model to models/
    -> src/api.py                  FastAPI serving layer, loads models/*.joblib
    -> Dockerfile                  containerizes the API
    -> .github/workflows/ci.yml    runs tests + validates the Docker build on every push/PR
```

All experiment runs are tracked in MLflow (`sqlite:///mlflow.db`, experiment `customer_churn_prediction`).

## Model Comparison

Metrics evaluated on the held-out test set (20% split, stratified), as logged in MLflow:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.8055 | 0.6582 | 0.5561 | **0.6029** | 0.8422 |
| RandomForest | 0.7935 | 0.6436 | 0.4973 | 0.5611 | 0.8287 |
| XGBoost | 0.7764 | 0.5891 | 0.5214 | 0.5532 | 0.8183 |
| RandomForest_Tuned | **0.8062** | **0.6747** | 0.5214 | 0.5882 | **0.8435** |
| XGBoost_Tuned | 0.7984 | 0.6471 | 0.5294 | 0.5824 | 0.8356 |

## Final Model Selection

**Logistic Regression** is the model served in production, despite `RandomForest_Tuned` edging it out on accuracy and ROC AUC.

The reason is F1 and recall on the churn class. In this problem, a **false negative** (a churner the model misses) is more costly than a **false positive** (a customer flagged for retention outreach who wasn't actually going to leave) — a missed churner is lost revenue, while a false alarm just costs a bit of unnecessary retention effort. Logistic Regression has the best F1 (0.6029) of all five models and the best recall (0.5561) among the tuned candidates, meaning it catches more actual churners than the tree-based alternatives, which is what matters most for this business problem.

## Running Locally

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Run the pipeline in order
python src/eda.py
python src/feature_engineering.py
python src/train_baseline.py
python src/train_compare.py
python src/train_tuned.py
python src/save_final_model.py

# Start the API
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

## Running with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **pandas / numpy** — data cleaning and manipulation
- **scikit-learn** — Logistic Regression, Random Forest, preprocessing, model selection
- **XGBoost** — gradient-boosted tree model
- **MLflow** — experiment tracking and model logging
- **FastAPI / uvicorn** — model serving API
- **Docker** — containerized deployment
- **GitHub Actions** — CI/CD (tests + Docker build validation)

## Project Structure

```
customer-churn-prediction-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/                  # original CSV (gitignored)
│   └── processed/            # cleaned + train/test splits (gitignored, regenerable)
├── models/                   # trained model artifacts (tracked in git)
│   ├── churn_model.joblib
│   ├── feature_columns.joblib
│   └── scaler.joblib
├── notebooks/
├── src/
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_baseline.py
│   ├── train_compare.py
│   ├── train_tuned.py
│   ├── save_final_model.py
│   ├── api.py
│   └── verify_api.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt           # full dev/training environment
├── requirements-api.txt       # minimal runtime deps for serving
└── README.md
```
