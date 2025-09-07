#!/usr/bin/env python3
"""
Streamlit App: Fake News Detector

- Loads the trained pipeline (preprocessing + TF-IDF + classifier) saved by train_fake_news.py
- Lets the user enter news text and returns a prediction (Fake / Real) with confidence.
- Beginner-friendly with comments.
"""

import os
import pickle
import streamlit as st
import numpy as np

from preprocessor import TextPreprocessor

# Inference needs the same NLTK resources as training (for lemmatization), so ensure they're present.
import nltk
def ensure_nltk_data():
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

ensure_nltk_data()

MODEL_PATH = os.path.join("models", "best_model.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write("Enter a news article or snippet below. The app will predict whether it's **Fake** or **Real**.")

# Load trained model artifact (pipeline + label encoder)
@st.cache_resource(show_spinner=True)
def load_artifact(path):
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run the training script first to create models/best_model.pkl.")
    st.stop()

artifact = load_artifact(MODEL_PATH)
pipeline = artifact["pipeline"]
label_encoder = artifact["label_encoder"]
best_model_name = artifact.get("best_model_name", "UnknownModel")

with st.expander("ℹ️ Model Info", expanded=False):
    st.write(f"**Classifier:** {best_model_name}")
    st.write("The pipeline includes text preprocessing (lowercase, punctuation removal, stopword removal, lemmatization) and TF‑IDF features.")

# Text input
user_text = st.text_area("Paste news text here:", height=220, placeholder="Type or paste the news content...")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Predict")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

def prettify_class(c):
    # Map original class back to a nice label (Fake/Real) when possible
    s = str(c).strip().lower()
    if s in {"fake", "false", "0"}:
        return "Fake"
    if s in {"real", "true", "1"}:
        return "Real"
    # Fallback to the raw class name
    return str(c)

if run_btn:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Predict class
        pred = pipeline.predict([user_text])[0]                 # numeric encoded label
        pred_class = label_encoder.inverse_transform([pred])[0] # original class name from the dataset
        pred_label = prettify_class(pred_class)

        # Confidence score (if classifier supports predict_proba)
        prob_text = ""
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba([user_text])[0]
                # The order of probabilities corresponds to label_encoder.classes_
                classes = label_encoder.classes_
                # Create a readable string with class -> probability mapping
                lines = [f"- **{prettify_class(c)}**: {float(p)*100:.2f}%" for c, p in zip(classes, proba)]
                prob_text = "\n".join(lines)
            except Exception:
                pass

        # Show result
        if pred_label.lower() == "real":
            st.success(f"Prediction: **{pred_label}** ✅")
        elif pred_label.lower() == "fake":
            st.error(f"Prediction: **{pred_label}** ❌")
        else:
            st.info(f"Prediction: **{pred_label}**")

        if prob_text:
            st.markdown("**Confidence:**")
            st.markdown(prob_text)

st.caption("Tip: To retrain with a new dataset, run `python train_fake_news.py --data_path path/to/your.csv`, then restart this app.")
