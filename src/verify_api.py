from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CLEANED_PATH = PROCESSED_DIR / "cleaned.csv"
API_URL = "http://127.0.0.1:8000/predict"
N_SAMPLES = 20


def reconstruct_raw_test_rows():
    # X_test.csv/y_test.csv were saved with index=False, so the row
    # identities linking them back to cleaned.csv are gone. To recover the
    # RAW (pre-encoding) field values for each X_test row, reproduce
    # feature_engineering.py's exact train_test_split (same test_size,
    # stratify=Churn, random_state=42) directly on cleaned.csv. Given the
    # same row order/count and the same stratify values, this is
    # deterministic and yields the identical test-set rows in the
    # identical order -- verified below against the saved y_test.
    cleaned = pd.read_csv(CLEANED_PATH)
    X_raw = cleaned.drop(columns=["Churn"])
    y_raw = cleaned["Churn"]

    _, X_test_raw, _, y_test_check = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )
    return X_test_raw.reset_index(drop=True), y_test_check.reset_index(drop=True)


def main():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    X_test_raw, y_test_check = reconstruct_raw_test_rows()

    assert len(X_test_raw) == len(X_test), "Row count mismatch vs X_test.csv!"
    assert (y_test_check.values == y_test.values).all(), (
        "Reconstructed split does not match saved y_test -- row alignment broken!"
    )

    print("=" * 60)
    print(f"TESTING FIRST {N_SAMPLES} X_test ROWS AGAINST LIVE API")
    print("=" * 60)

    correct = 0
    for i in range(N_SAMPLES):
        payload = X_test_raw.iloc[i].to_dict()
        actual = int(y_test.iloc[i])

        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        predicted = response.json()["churn_prediction"]

        if predicted == actual:
            correct += 1

        print(
            f"Row {i:>2}: predicted={predicted} actual={actual}  "
            f"{'correct' if predicted == actual else 'wrong':<7} "
            f"running: {correct}/{i + 1}"
        )

    accuracy = correct / N_SAMPLES * 100
    print("\n" + "=" * 60)
    print(f"FINAL: {correct}/{N_SAMPLES} correct -> {accuracy:.1f}% accuracy")
    print("=" * 60)


if __name__ == "__main__":
    main()
