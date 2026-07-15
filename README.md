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
    -> src/train_improved.py       feature engineering (tenure buckets, service counts,
                                    avg monthly spend) + class imbalance experiments
                                    (class_weight='balanced', SMOTE), logged to MLflow
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
| LogisticRegression | 0.8055 | 0.6582 | 0.5561 | 0.6029 | 0.8422 |
| RandomForest | 0.7935 | 0.6436 | 0.4973 | 0.5611 | 0.8287 |
| XGBoost | 0.7764 | 0.5891 | 0.5214 | 0.5532 | 0.8183 |
| RandomForest_Tuned | 0.8062 | 0.6747 | 0.5214 | 0.5882 | **0.8435** |
| XGBoost_Tuned | 0.7984 | 0.6471 | 0.5294 | 0.5824 | 0.8356 |
| LogisticRegression_Engineered | 0.8006 | 0.6535 | 0.5294 | 0.5849 | 0.8421 |
| **LogisticRegression_Balanced** | 0.7360 | 0.5017 | **0.7941** | **0.6149** | 0.8420 |
| LogisticRegression_SMOTE | 0.7473 | 0.5183 | 0.6818 | 0.5889 | 0.8267 |

## Final Model Selection

The plain **Logistic Regression** model was initially selected as the best balance of metrics among the first 5 models trained (Logistic Regression, Random Forest, XGBoost, and their tuned variants) — it had the best F1 (0.6029) and best recall (0.5561) of that group, which matters more than raw accuracy for an imbalanced churn problem where missing a churner is costly.

A follow-up round of experiments then tested whether feature engineering (tenure buckets, a total-services count, average monthly spend) and class imbalance handling (`class_weight='balanced'`, SMOTE oversampling) could do better. **`LogisticRegression_Balanced`** emerged as the strongest model of all 8 tested:

- **F1 of 0.6149** — the best of every model tried
- **Recall of 0.7941** — it catches ~79% of actual churners, versus ~56% for the original model

The tradeoff is real and worth stating explicitly: precision drops to 0.50 and accuracy to 0.7360, meaning roughly half of the customers flagged as "will churn" are false alarms. This is an acceptable tradeoff **if** the cost of a retention outreach (an email, a discount offer) is much lower than the cost of a missed churner — which is the typical case in telecom churn prevention, where losing a customer's ongoing revenue far outweighs the cost of an unnecessary discount.

**`LogisticRegression_Balanced` is the model now served by `models/churn_model.joblib` and the FastAPI `/predict` endpoint.** The API's preprocessing was updated to compute the same engineered features and 27-column encoding this model expects, and this was verified end-to-end: all pytest tests pass, and live `/predict` calls against the held-out test set correctly identified churners the previous model had missed (confirmed via `src/verify_api.py`).

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

Note: the Docker image may need to be rebuilt (`docker build -t churn-api .`) to reflect the latest model artifacts in `models/`, since Docker images are not automatically kept in sync with local file changes.

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
│   ├── train_improved.py
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
