import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "BBCNews.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "news_processed.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "news_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.txt")

TEST_SIZE = 0.2
RANDOM_STATE = 42