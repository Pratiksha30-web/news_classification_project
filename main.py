from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    preprocess_data()
    X_test, y_test = train_model()
    evaluate_model(X_test, y_test)

if __name__ == "__main__":
    main()