import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.config import MODEL_PATH, METRICS_PATH
import os

def evaluate_model(X_test, y_test):
    model, vectorizer = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"\nClassification Report:\n{report}")

    print("Accuracy:", acc)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)