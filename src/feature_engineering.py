import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import PROCESSED_DATA_PATH

def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df['text'], df['category']

def create_vectorizer():
    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True
    )