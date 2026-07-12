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
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ROOT_DIR = Path(__file__).resolve().parent.parent

# Verified numbers from earlier logged runs, included here so the final
# comparison table can show all 5 models without retraining the untuned ones.
PRIOR_METRICS = {
    "LogisticRegression": {
        "accuracy": 0.8055,
        "precision": 0.6582,
        "recall": 0.5561,
        "f1": 0.6029,
        "roc_auc": 0.8422,
    },
    "RandomForest": {
        "accuracy": 0.7935,
        "precision": 0.6436,
        "recall": 0.4973,
        "f1": 0.5611,
        "roc_auc": 0.8287,
    },
    "XGBoost": {
        "accuracy": 0.7764,
        "precision": 0.5891,
        "recall": 0.5214,
        "f1": 0.5532,
        "roc_auc": 0.8183,
    },
}

RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.8, 0.9, 1.0],
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


def tune_and_log(estimator, param_dist, run_name, log_model_fn, X_train, X_test, y_train, y_test):
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("=" * 60)
    print(f"{run_name.upper()} BEST HYPERPARAMETERS")
    print("=" * 60)
    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    with mlflow.start_run(run_name=run_name):
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_params(search.best_params_)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        print(f"\n{run_name.upper()} TEST METRICS")
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

        model_info = log_model_fn(best_model, name="model")
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

    rf_metrics = tune_and_log(
        RandomForestClassifier(random_state=42),
        RF_PARAM_DIST,
        "RandomForest_Tuned",
        mlflow.sklearn.log_model,
        X_train, X_test, y_train, y_test,
    )

    xgb_metrics = tune_and_log(
        XGBClassifier(random_state=42, eval_metric="logloss"),
        XGB_PARAM_DIST,
        "XGBoost_Tuned",
        mlflow.xgboost.log_model,
        X_train, X_test, y_train, y_test,
    )

    print("=" * 60)
    print("MODEL COMPARISON (ALL 5 MODELS)")
    print("=" * 60)
    rows = [
        ("LogisticRegression", PRIOR_METRICS["LogisticRegression"]),
        ("RandomForest", PRIOR_METRICS["RandomForest"]),
        ("XGBoost", PRIOR_METRICS["XGBoost"]),
        ("RandomForest_Tuned", rf_metrics),
        ("XGBoost_Tuned", xgb_metrics),
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
