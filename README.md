# 📰 Fake News Detection (Machine Learning)

A beginner-friendly, end-to-end project to classify news as **Fake** or **Real** using machine learning.  
It trains two models (Naive Bayes and Logistic Regression), evaluates them, saves the best pipeline with pickle, and includes a simple **Streamlit** app for live predictions.

## 📂 Project Structure
```
fake-news-ml-project/
├── data/                 # Put your Kaggle CSV here (with text + label)
├── models/               # Saved model(s)
├── reports/              # Evaluation reports
├── app.py                # Streamlit app
├── train_fake_news.py    # Training script
└── requirements.txt
```

## ✅ Requirements
- Python 3.8+
- `pip install -r requirements.txt`

## 📥 Dataset
Use any Kaggle Fake News dataset in CSV format that includes a **text** column and a **label** column.  
If your dataset uses different column names (e.g., `content`, `target`, `class`, etc.), you can pass them via command-line flags.

> Example datasets on Kaggle: "Fake and real news dataset", "Fake News", etc.

## 🚀 Train the Models
1. Place your dataset at `data/fake_news.csv` (or any path you like).
2. Run the training script:
   ```bash
   python train_fake_news.py --data_path data/fake_news.csv
   ```

If your dataset has different columns:
```bash
python train_fake_news.py --data_path data/your.csv --text_column content --label_column label
```

If you also have a title column you'd like to include:
```bash
python train_fake_news.py --data_path data/your.csv --text_column text --label_column label --title_column title
```

This will:
- preprocess text (lowercase, punctuation removal, stopwords, lemmatization)
- vectorize with TF‑IDF (unigrams + bigrams)
- train **Naive Bayes** and **Logistic Regression**
- evaluate accuracy, precision, recall, F1-score, confusion matrix
- save the **best** pipeline (including preprocessing + TF‑IDF + classifier + label encoder) to `models/best_model.pkl`
- write a human-readable report to `reports/metrics.txt`

## 🖥️ Run the Streamlit App
After training completes:
```bash
streamlit run app.py
```
Then paste any news text in the app to get a **Fake/Real** prediction with confidence.

## 🧠 How it Works (Short Version)
- **Preprocessing:** lowercase → remove punctuation & non-letters → remove English stopwords → lemmatize with NLTK.
- **Features:** TF‑IDF (1–2 grams, 50k max features).
- **Models:** Multinomial Naive Bayes and Logistic Regression.
- **Selection:** Pick the one with the highest F1-score on the test set.
- **Serving:** The saved pipeline performs the same preprocessing at inference, so the Streamlit app just loads and predicts.

## ✏️ Notes
- If NLTK resources are missing, scripts will download them automatically on first run.
- If your dataset has imbalanced classes, consider experimenting with `class_weight='balanced'` in Logistic Regression or tuning TF‑IDF/thresholds.
- You can extend this project with cross-validation, hyperparameter search, or more models (SVM, LinearSVC, etc.).

Happy learning! 🎉
