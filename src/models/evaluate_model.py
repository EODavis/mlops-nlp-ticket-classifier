import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "support_tickets.csv"
    model_path = project_root / "models" / "ticket_classifier.joblib"
    report_path = project_root / "reports" / "evaluation_report.txt"

    # Load data and model
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # Split data
    X = df["text"]
    y = df["label"]
    test_size = 0.2
    random_state = 42
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save report
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("====================\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix\n")
        f.write(str(cm))

    print("Evaluation complete. Report saved to:", report_path)
    print(report)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
