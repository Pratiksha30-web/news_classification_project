from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import joblib
import os

from src.feature_engineering import load_data, create_vectorizer
from src.config import MODEL_PATH, TEST_SIZE, RANDOM_STATE

def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    vectorizer = create_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_vec, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, vectorizer), MODEL_PATH)

    print("Model saved.")
    return X_test_vec, y_test