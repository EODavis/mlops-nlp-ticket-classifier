import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.utils.config_loader import load_config


def train_model():
    config = load_config()

    raw_path = config["data"]["raw_path"]
    model_path = config["model"]["model_path"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]
    report_path = config["reports"]["evaluation_report"]

    print(f"Loading dataset from: {raw_path}")
    df = pd.read_csv(raw_path)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred)

    # Ensure folders exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # Save report
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Evaluation report saved to: {report_path}")
    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    train_model()
