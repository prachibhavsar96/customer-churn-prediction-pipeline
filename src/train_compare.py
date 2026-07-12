from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ROOT_DIR = Path(__file__).resolve().parent.parent

# Verified numbers from train_baseline.py's logged run (run ID
# cb9dabf59cc84ce79c7ecf8200440164), included here so the comparison
# table can show all 3 models without retraining Logistic Regression.
LOGISTIC_REGRESSION_METRICS = {
    "accuracy": 0.8055,
    "precision": 0.6582,
    "recall": 0.5561,
    "f1": 0.6029,
    "roc_auc": 0.8422,
}


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    return fig


def train_and_log(model, run_name, params, log_model_fn, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        cm = confusion_matrix(y_test, y_pred)

        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        print("=" * 60)
        print(f"{run_name.upper()} METRICS")
        print("=" * 60)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print(f"\n{run_name} confusion matrix:")
        print(cm)

        fig = plot_confusion_matrix(cm, f"{run_name} Confusion Matrix")
        cm_path = ROOT_DIR / f"confusion_matrix_{run_name.lower()}.png"
        fig.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink()

        # Pass name= as a keyword, not positionally: the 2nd positional
        # parameter is the deprecated artifact_path arg, which routes
        # through MLflow 3.x's older logging path and does not show up
        # under the run's Artifacts tab the way name= does.
        model_info = log_model_fn(model, name="model")
        print(f"\nModel artifact logged at: {model_info.model_uri}")

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}\n")

        return metrics


def main():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("customer_churn_prediction")

    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_metrics = train_and_log(
        rf_model,
        "RandomForest",
        {"model_type": "RandomForestClassifier", "n_estimators": 100},
        mlflow.sklearn.log_model,
        X_train, X_test, y_train, y_test,
    )

    xgb_model = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb_metrics = train_and_log(
        xgb_model,
        "XGBoost",
        {"model_type": "XGBClassifier", "eval_metric": "logloss"},
        mlflow.xgboost.log_model,
        X_train, X_test, y_train, y_test,
    )

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    rows = [
        ("LogisticRegression", LOGISTIC_REGRESSION_METRICS),
        ("RandomForest", rf_metrics),
        ("XGBoost", xgb_metrics),
    ]
    header = f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'ROC AUC':>10}"
    print(header)
    for name, m in rows:
        print(
            f"{name:<20}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
            f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['roc_auc']:>10.4f}"
        )


if __name__ == "__main__":
    main()
