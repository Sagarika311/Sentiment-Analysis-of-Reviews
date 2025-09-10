# train.py
import os
import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from preprocess import tokenize  # must be importable for pickling/unpickling

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")

def load_data(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must have 'review' and 'sentiment' columns")
        return df[['review', 'sentiment']].dropna()
    # fallback tiny dataset (replace with real data)
    data = [
        ("This product is great!", "positive"),
        ("I don't like this product.", "negative"),
        ("Amazing quality and good service!", "positive"),
        ("Worst purchase ever.", "negative"),
        ("The item was damaged upon arrival.", "negative"),
        ("Excellent customer support!", "positive"),
        ("The product broke after a week.", "negative"),
        ("I'm very satisfied with my purchase.", "positive"),
        ("Terrible experience, would not recommend.", "negative"),
        ("This exceeded my expectations!", "positive")
    ]
    return pd.DataFrame(data, columns=['review', 'sentiment'])

def build_and_train(df, random_state=42):
    X = df['review'].values
    y = df['sentiment'].values

    # encode labels for consistency
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # pipeline - tokenizer is our tokenize function defined in preprocess.py
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,2), max_features=20000)),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000, class_weight=None, random_state=random_state))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Evaluation on held-out test set:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Attach label encoder classes to pipeline for convenience
    pipeline.label_classes_ = le.classes_
    return pipeline

def save_pipeline(pipeline, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved pipeline to {path}")

def main(args):
    df = load_data(args.data) if args.data else load_data(None)
    pipeline = build_and_train(df)
    save_pipeline(pipeline, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to CSV containing review,sentiment columns", default=None)
    parser.add_argument("--output", help="Where to save the pipeline", default=MODEL_PATH)
    args = parser.parse_args()
    main(args)
