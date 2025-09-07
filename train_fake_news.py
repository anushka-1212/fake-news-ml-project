#!/usr/bin/env python3
"""
Fake News Detection (Machine Learning) - Training Script

This script trains two models (Naive Bayes and Logistic Regression) to detect fake vs real news
using a Kaggle CSV dataset with text and label columns. It includes:

1) Import libraries
2) Load dataset
3) Preprocess text (lowercase, remove punctuation, stopwords, lemmatization)
4) Convert text to numerical features (TF-IDF)
5) Train/test split
6) Train at least 2 models (Naive Bayes & Logistic Regression)
7) Evaluate with accuracy, confusion matrix, precision, recall, F1-score
8) Save the best trained model using pickle (as a full pipeline ready for inference)

Beginner-friendly: everything is commented step by step.

USAGE (examples):
-----------------
# Basic run with default column names: text, label
python train_fake_news.py --data_path data/fake_news.csv

# If your dataset has different column names (e.g., 'content' and 'label'):
python train_fake_news.py --data_path data/your_file.csv --text_column content --label_column label

# If your dataset has both 'title' and 'text' and you want to combine them:
python train_fake_news.py --data_path data/your_file.csv --text_column text --title_column title

The script will save the best model pipeline to: models/best_model.pkl
and a metrics report to: reports/metrics.txt
"""

import argparse
import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")  # Keep console output beginner-friendly

# 1) Import libraries
import numpy as np
import pandas as pd

# We'll use NLTK for stopwords and lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessor import TextPreprocessor

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ---------------------------
# Helper: Make sure NLTK data is available
# ---------------------------
def ensure_nltk_data():
    """
    Downloads NLTK resources if they are not present.
    This is safe to run multiple times.
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

# ---------------------------
# Text Preprocessor (Lowercase, remove punctuation, stopwords, lemmatization)
# ---------------------------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that:
      - lowercases text
      - removes punctuation and non-letters
      - removes English stopwords
      - lemmatizes tokens (reduces words to their base form)
    Output: cleaned string, tokens joined by spaces
    """
    def __init__(self, remove_stopwords=True, do_lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.do_lemmatize = do_lemmatize
        self._stopwords = None
        self._lemmatizer = None
        # Precompile a simple regex to keep only letters and spaces
        self._only_letters = re.compile(r'[^a-zA-Z\s]+')

    def fit(self, X, y=None):
        ensure_nltk_data()
        if self.remove_stopwords:
            self._stopwords = set(stopwords.words('english'))
        else:
            self._stopwords = set()

        if self.do_lemmatize:
            self._lemmatizer = WordNetLemmatizer()
        return self

    def _process_one(self, text):
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        # Lowercase
        text = text.lower()

        # Remove punctuation and non-letters
        text = self._only_letters.sub(' ', text)

        # Tokenize on whitespace
        tokens = [tok for tok in text.split() if tok.strip() != ""]

        # Remove stopwords
        if self.remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        # Lemmatize (lightweight – no POS tagging for simplicity)
        if self.do_lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        # Join back to a single string for the vectorizer
        return " ".join(tokens)

    def transform(self, X):
        return [self._process_one(x) for x in X]

# ---------------------------
# Load and normalize dataset
# ---------------------------
def load_dataset(path, text_column="text", label_column="label", title_column=None):
    """
    Loads a CSV and returns a DataFrame with exactly two columns:
      - 'text' : the news content as string
      - 'label': the label (string or int)
    If title_column is provided and exists, we concatenate: "[title]. [text]"
    """
    df = pd.read_csv(path)


    # Try to be flexible with typical Kaggle schemas
    # If the requested columns don't exist, try common alternatives
    available_cols = set(df.columns.str.lower())
    col_map = {c.lower(): c for c in df.columns}  # lower->original

    # Heuristics for text column
    candidates_text = [text_column.lower(), 'text', 'content', 'article', 'body']
    text_col = next((col_map[c] for c in candidates_text if c in available_cols), None)

    # Heuristics for label column
    candidates_label = [label_column.lower(), 'label', 'target', 'class']
    label_col = next((col_map[c] for c in candidates_label if c in available_cols), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns. Found columns: {list(df.columns)}")

    # If a title column is provided and exists, combine it
    if title_column:
        if title_column.lower() in available_cols:
            title_col = col_map[title_column.lower()]
            df['__combined_text__'] = df[title_col].astype(str).fillna('') + '. ' + df[text_col].astype(str).fillna('')
            text_col = '__combined_text__'
        else:
            print(f"[INFO] Provided title_column='{title_column}' not found. Using only '{text_col}'.")

    # Keep only two columns
    df = df[[text_col, label_col]].copy()
    df.columns = ['text', 'label']  # rename for consistency

    # Drop rows with missing text/label
    before = len(df)
    df.dropna(subset=['text', 'label'], inplace=True)
    after = len(df)
    if after < before:
        print(f"[INFO] Dropped {before - after} rows with missing text/label.")

    # Strip whitespace
    df['text'] = df['text'].astype(str).str.strip()
    return df

# ---------------------------
# Prepare labels with LabelEncoder
# ---------------------------
def encode_labels(y_raw):
    """
    Use sklearn's LabelEncoder so we can robustly handle string or numeric labels.
    Returns encoded y, and the fitted encoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return y, le

# ---------------------------
# Build pipelines (TF-IDF + Classifier)
# ---------------------------
def build_pipelines():
    """
    Creates two pipelines that both include preprocessing and TF-IDF:
      - Multinomial Naive Bayes
      - Logistic Regression
    """
    common_steps = [
        ('preprocess', TextPreprocessor(remove_stopwords=True, do_lemmatize=True)),
        ('tfidf', TfidfVectorizer(
            # TF-IDF turns cleaned text into numeric features
            ngram_range=(1, 2),        # unigrams + bigrams often work well for text classification
            min_df=2,                  # ignore rare terms
            max_features=50000,        # cap vocabulary size (tune as needed)
            sublinear_tf=True,         # use 1 + log(tf) instead of raw tf
            lowercase=False            # we already lowercased during preprocessing
        ))
    ]

    pipe_nb = Pipeline(common_steps + [
        ('clf', MultinomialNB())
    ])

    pipe_lr = Pipeline(common_steps + [
        ('clf', LogisticRegression(max_iter=300, solver='liblinear'))  # simple & effective baseline
    ])

    return {
        "NaiveBayes": pipe_nb,
        "LogisticRegression": pipe_lr
    }

# ---------------------------
# Evaluate helper
# ---------------------------
def evaluate(y_true, y_pred, average='binary'):
    """
    Compute standard metrics and return a dict. Also returns the confusion matrix.
    If labels are not {0,1} binary, sklearn will adapt accordingly (for multi-class use 'macro').
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report
    }

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # 2) Load dataset
    print("[1/8] Loading dataset...")
    df = load_dataset(args.data_path, text_column=args.text_column, label_column=args.label_column, title_column=args.title_column)
    print(f"Loaded {len(df)} rows. Example columns: {list(df.columns)}")

    # 3) Encode labels (works for string or numeric labels)
    print("[2/8] Encoding labels...")
    y, label_encoder = encode_labels(df['label'])
    X = df['text']

    # Decide averaging for metrics: if exactly 2 classes -> 'binary', else 'macro'
    avg = 'binary' if len(label_encoder.classes_) == 2 else 'macro'

    # 5) Train-test split
    print("[3/8] Splitting data (train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # 6) Build and train models
    print("[4/8] Building pipelines...")
    pipelines = build_pipelines()

    all_results = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, pipe in pipelines.items():
        print(f"[5/8] Training model: {name} ...")
        pipe.fit(X_train, y_train)

        print(f"[6/8] Evaluating model: {name} ...")
        y_pred = pipe.predict(X_test)
        results = evaluate(y_test, y_pred, average=avg)
        all_results[name] = results

        print(f"  {name} Accuracy: {results['accuracy']:.4f}")
        print(f"  {name} Precision: {results['precision']:.4f}")
        print(f"  {name} Recall: {results['recall']:.4f}")
        print(f"  {name} F1-score: {results['f1']:.4f}")
        print(f"  {name} Confusion matrix:\n{results['confusion_matrix']}")

        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_name = name
            best_pipeline = pipe

    # 7) Save metrics to a friendly report
    print("[7/8] Writing metrics report...")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(args.reports_dir, "metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Fake News Detection - Metrics Report\nGenerated: {ts}\n\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Classes: {list(map(str, label_encoder.classes_))}\n\n")
        for name, res in all_results.items():
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Recall:    {res['recall']:.4f}\n")
            f.write(f"F1-score:  {res['f1']:.4f}\n")
            f.write(f"Confusion matrix:\n{res['confusion_matrix']}\n")
            f.write("\nClassification report:\n")
            f.write(res['report'])
            f.write("\n" + "-"*60 + "\n\n")
        f.write(f"Best model by F1-score: {best_name} ({best_f1:.4f})\n")

    print(f"[INFO] Metrics written to: {report_path}")

    # 8) Save the best model as a single, ready-to-use artifact
    print("[8/8] Saving the best pipeline (with label encoder) using pickle...")
    artifact = {
        "pipeline": best_pipeline,       # full sklearn pipeline (preprocess + tfidf + classifier)
        "label_encoder": label_encoder,  # to map numeric predictions back to original class names
        "best_model_name": best_name
    }
    model_path = os.path.join(args.models_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[DONE] Saved best model to: {model_path}")
    print("[TIP] Next: run the Streamlit app with:  streamlit run app.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection models (Naive Bayes & Logistic Regression).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the Kaggle CSV dataset (with text + label).")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text/content column.")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label/target column.")
    parser.add_argument("--title_column", type=str, default=None, help="Optional: name of the title column to concatenate.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--models_dir", type=str, default="models", help="Where to save models (default: models/)")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Where to save reports (default: reports/)")
    args = parser.parse_args()
    main(args)
