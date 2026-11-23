"""
Main sentiment classification prototype
Input: final_preprocessed_dataset.csv
Features: preprocessed_text
Target: sentiment (negative, neutral, positive)

Run:
    pip install -r requirements.txt
    python main.py
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------
DATA_PATH = Path("final_preprocessed_dataset.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.20
LABELS = ["negative", "neutral", "positive"]


# ---------------------------------------------------------
# 2. LOAD AND INSPECT DATASET
# ---------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Put the CSV file in the same "
            "folder as main.py and name it final_preprocessed_dataset.csv."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET INSPECTION ==========")
    print("Dataset shape (rows, columns):", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData information:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nFull-row duplicate count:", df.duplicated().sum())
    print(
        "Duplicate preprocessed_text count:",
        df["preprocessed_text"].duplicated().sum()
    )
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print("\nSentiment percentages:")
    print((df["sentiment"].value_counts(normalize=True) * 100).round(2))

    # Only the two columns needed for supervised classification.
    required_columns = ["preprocessed_text", "sentiment"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column is missing: {column}")

    # The uploaded dataset is already preprocessed.
    # We only remove rows with missing/empty required values.
    df = df[required_columns].dropna().copy()
    df["preprocessed_text"] = df["preprocessed_text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["preprocessed_text"] != ""]

    unexpected = set(df["sentiment"].unique()) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels found: {unexpected}")

    print("\nRecords used for modelling:", len(df))

    # ---------------------------------------------------------
    # 3. DEFINE INPUT (X) AND TARGET (y)
    # ---------------------------------------------------------
    X = df["preprocessed_text"]
    y = df["sentiment"]

    # ---------------------------------------------------------
    # 4. TRAIN-TEST SPLIT
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n========== TRAIN-TEST SPLIT ==========")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("\nTraining label distribution:")
    print(y_train.value_counts())
    print("\nTesting label distribution:")
    print(y_test.value_counts())

    # ---------------------------------------------------------
    # 5. TF-IDF + TWO CLASSIFICATION MODELS
    # ---------------------------------------------------------
    # Keeping TF-IDF inside Pipeline ensures that the vocabulary
    # and IDF weights are learned from training data only.
    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                sublinear_tf=True
            )),
            ("classifier", MultinomialNB())
        ]),

        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=10000,
                sublinear_tf=True
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    }

    comparison_results = []
    all_reports = {}

    # ---------------------------------------------------------
    # 6. TRAIN, PREDICT AND EVALUATE EACH MODEL
    # ---------------------------------------------------------
    for model_name, model in models.items():
        print(f"\n========== {model_name.upper()} ==========")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        f1_macro = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        f1_weighted = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Macro Precision:   {precision_macro:.4f}")
        print(f"Macro Recall:      {recall_macro:.4f}")
        print(f"Macro F1-score:    {f1_macro:.4f}")
        print(f"Weighted F1-score: {f1_weighted:.4f}")

        print("\nClassification report:")
        report = classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0
        )
        print(classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            zero_division=0
        ))
        all_reports[model_name] = report

        # Confusion matrix: rows = actual, columns = predicted.
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        print("Confusion matrix (actual rows x predicted columns):")
        print(cm)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS
        )
        display.plot(values_format="d", xticks_rotation=30)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(OUTPUT_DIR / f"confusion_matrix_{safe_name}.png", dpi=300)
        plt.close()

        # Save actual vs predicted labels for error inspection.
        prediction_df = pd.DataFrame({
            "preprocessed_text": X_test.reset_index(drop=True),
            "actual_sentiment": y_test.reset_index(drop=True),
            "predicted_sentiment": pd.Series(y_pred)
        })
        prediction_df.to_csv(
            OUTPUT_DIR / f"predictions_{safe_name}.csv",
            index=False
        )

        comparison_results.append({
            "model": model_name,
            "accuracy": accuracy,
            "macro_precision": precision_macro,
            "macro_recall": recall_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted
        })

        # Save the complete pipeline (TF-IDF + trained classifier).
        joblib.dump(model, OUTPUT_DIR / f"{safe_name}_model.joblib")

    # ---------------------------------------------------------
    # 7. SAVE COMPARISON AND REPORTS
    # ---------------------------------------------------------
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "classification_reports.json", "w", encoding="utf-8") as file:
        json.dump(all_reports, file, indent=4)

    print("\n========== MODEL COMPARISON ==========")
    print(comparison_df.to_string(index=False))
    
"""
    print("\nFinished. Check the 'outputs' folder for:")
    print("- model_comparison.csv")
    print("- classification_reports.json")
    print("- confusion_matrix_naive_bayes.png")
    print("- confusion_matrix_logistic_regression.png")
    print("- predictions_*.csv")
    print("- *_model.joblib")
"""

if __name__ == "__main__":
    main()
