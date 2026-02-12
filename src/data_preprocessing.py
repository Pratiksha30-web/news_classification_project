import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return ' '.join(words)

def preprocess_data():
    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop junk column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    df.dropna(inplace=True)

    # Clean text
    df['text'] = df['descr'].apply(clean_text)

    # Use FIRST tag only
    df['category'] = df['tags'].apply(lambda x: x.split(',')[0].strip())

    # 🔥 REMOVE RARE TAGS (KEY FOR ACCURACY)
    tag_counts = df['category'].value_counts()
    valid_tags = tag_counts[tag_counts >= 20].index
    df = df[df['category'].isin(valid_tags)]

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df[['text', 'category']].to_csv(PROCESSED_DATA_PATH, index=False)

    print("Classes after filtering:", df['category'].nunique())
    print("Preprocessed data saved.")