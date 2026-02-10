import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "support_tickets.csv"
    model_path = project_root / "models" / "ticket_classifier.joblib"

    df = pd.read_csv(data_path)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=300))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Saving model...")
    joblib.dump(pipeline, model_path)

    print(f"Model saved at: {model_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
