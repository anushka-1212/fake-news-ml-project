# 🧠 Advanced Fake News Detection System

Production-style fake news detection pipeline with model selection, hyperparameter optimization, explainability outputs, and deployment-ready artifacts.

## 🚀 What Makes This Project Advanced

- **Multi-model benchmarking** across four strong linear baselines:
  - Complement Naive Bayes
  - Logistic Regression
  - Calibrated Linear SVM (probability-enabled)
  - Passive Aggressive Classifier
- **Automated hyperparameter optimization** with `RandomizedSearchCV`
- **Cross-validation driven model selection** (not single-run only)
- **Probability calibration** for SVM-based predictions
- **Interpretability outputs** with top weighted terms per class
- **Dual reports**:
  - `reports/metrics.txt` (human-readable)
  - `reports/metrics.json` (machine-readable)
- **Reusable serialized artifact** including pipeline, metrics, label encoder, and metadata
- **CLI inference utility** (`predict.py`) for quick testing and integration

---

## 📂 Project Structure

```bash
fake-news-ml-project/
├── app.py
├── train_fake_news.py      # Advanced training + model search
├── predict.py              # CLI inference
├── preprocessor.py         # Sklearn-compatible text cleaning transformer
├── models/
│   └── best_model.pkl
├── reports/
│   ├── metrics.txt
│   └── metrics.json
└── requirements.txt
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Python 3.8+ recommended.

---

## 📥 Dataset Requirements

Use a CSV dataset with text and label fields.

Default expected columns:
- `text`
- `label`

If your dataset uses different names, pass them through CLI arguments.

---

## 🏋️ Train the Advanced System

```bash
python train_fake_news.py --data_path data/fake_news.csv
```

Optional arguments:

```bash
python train_fake_news.py \
  --data_path data/fake_news.csv \
  --text_column content \
  --label_column target \
  --title_column title \
  --cv_folds 5 \
  --search_iter 20 \
  --top_k_terms 30
```

### Training Outputs

- `models/best_model.pkl`
- `reports/metrics.txt`
- `reports/metrics.json`

---

## 🔍 CLI Inference

```bash
python predict.py --model_path models/best_model.pkl --text "Breaking: scientists discover..."
```

Output includes:
- Predicted class
- Raw label
- Confidence distribution (if probability is available)

---

## 🖥️ Streamlit App

```bash
streamlit run app.py
```

The app loads `models/best_model.pkl` and performs live predictions.

---

## 🧪 Suggested Next Enhancements

- Add transformer embeddings (BERT/RoBERTa) and compare with TF-IDF stack
- Add drift detection against incoming production data
- Build REST API (FastAPI) + Docker deployment
- Add experiment tracking (MLflow/W&B)
- Add threshold tuning based on business precision/recall targets

