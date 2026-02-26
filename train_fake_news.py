#!/usr/bin/env python3
"""Advanced fake news detection training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from preprocessor import TextPreprocessor


def load_dataset(path: str, text_column: str = "text", label_column: str = "label", title_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    available_cols = set(df.columns.str.lower())
    col_map = {c.lower(): c for c in df.columns}

    candidates_text = [text_column.lower(), "text", "content", "article", "body"]
    candidates_label = [label_column.lower(), "label", "target", "class"]

    text_col = next((col_map[c] for c in candidates_text if c in available_cols), None)
    label_col = next((col_map[c] for c in candidates_label if c in available_cols), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns. Available columns: {list(df.columns)}")

    if title_column and title_column.lower() in available_cols:
        title_col = col_map[title_column.lower()]
        df["__combined_text__"] = df[title_col].astype(str).fillna("") + ". " + df[text_col].astype(str).fillna("")
        text_col = "__combined_text__"

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df.dropna(subset=["text", "label"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df


def build_model_spaces(random_state: int) -> Dict[str, Tuple[Pipeline, dict]]:
    shared = [
        ("preprocess", TextPreprocessor(remove_stopwords=True, do_lemmatize=True)),
        (
            "tfidf",
            TfidfVectorizer(lowercase=False, strip_accents="unicode", sublinear_tf=True),
        ),
    ]

    spaces = {
        "ComplementNB": (
            Pipeline(shared + [("clf", ComplementNB())]),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 3, 5],
                "tfidf__max_features": [30000, 50000, 80000],
                "clf__alpha": np.logspace(-2, 1, 8),
            },
        ),
        "LogisticRegression": (
            Pipeline(shared + [("clf", LogisticRegression(max_iter=800, solver="liblinear", random_state=random_state))]),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 3, 5],
                "tfidf__max_features": [40000, 70000, 100000],
                "clf__C": np.logspace(-2, 1, 8),
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "LinearSVC-Calibrated": (
            Pipeline(
                shared
                + [
                    (
                        "clf",
                        CalibratedClassifierCV(
                            estimator=LinearSVC(random_state=random_state),
                            cv=3,
                            method="sigmoid",
                        ),
                    )
                ]
            ),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 3, 5],
                "tfidf__max_features": [50000, 90000],
                "clf__estimator__C": np.logspace(-2, 1, 8),
            },
        ),
        "PassiveAggressive": (
            Pipeline(
                shared
                + [
                    (
                        "clf",
                        PassiveAggressiveClassifier(
                            max_iter=1000,
                            random_state=random_state,
                            early_stopping=True,
                            n_iter_no_change=5,
                        ),
                    )
                ]
            ),
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 3, 5],
                "tfidf__max_features": [50000, 90000],
                "clf__C": np.logspace(-2, 1, 8),
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }
    return spaces


def evaluate(y_true, y_pred, average: str):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def extract_top_features(model: Pipeline, label_encoder: LabelEncoder, top_k: int = 25):
    if "tfidf" not in model.named_steps or "clf" not in model.named_steps:
        return {}

    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = np.array(tfidf.get_feature_names_out())

    coef = None
    if hasattr(clf, "coef_"):
        coef = clf.coef_
    elif hasattr(clf, "estimator") and hasattr(clf.estimator, "coef_"):
        coef = clf.estimator.coef_

    if coef is None:
        return {}

    coef = np.atleast_2d(coef)
    insights = {}

    for idx, cls in enumerate(label_encoder.classes_):
        row = coef[0] if coef.shape[0] == 1 else coef[idx]
        pos_idx = np.argsort(row)[-top_k:][::-1]
        neg_idx = np.argsort(row)[:top_k]
        insights[str(cls)] = {
            "top_positive_terms": names[pos_idx].tolist(),
            "top_negative_terms": names[neg_idx].tolist(),
        }
    return insights


def main(args):
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    print("[1/7] Loading data...")
    df = load_dataset(args.data_path, args.text_column, args.label_column, args.title_column)

    print("[2/7] Encoding labels + split...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scoring = "f1" if len(label_encoder.classes_) == 2 else "f1_macro"
    avg = "binary" if len(label_encoder.classes_) == 2 else "macro"

    print("[3/7] Hyperparameter search across candidate models...")
    spaces = build_model_spaces(args.random_state)
    all_results = {}
    best_name = None
    best_model = None
    best_score = -1.0

    for name, (pipe, params) in spaces.items():
        print(f"  -> searching {name}")
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            n_iter=args.search_iter,
            scoring=scoring,
            n_jobs=-1,
            cv=args.cv_folds,
            random_state=args.random_state,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        pred = search.best_estimator_.predict(X_test)
        metrics = evaluate(y_test, pred, average=avg)
        metrics["cv_best_score"] = float(search.best_score_)
        metrics["best_params"] = search.best_params_
        all_results[name] = metrics

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_name = name
            best_model = search.best_estimator_

    print("[4/7] Collecting model explainability summary...")
    top_terms = extract_top_features(best_model, label_encoder, top_k=args.top_k_terms)

    artifact = {
        "pipeline": best_model,
        "label_encoder": label_encoder,
        "best_model_name": best_name,
        "best_f1": best_score,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": all_results,
        "top_terms": top_terms,
    }

    print("[5/7] Saving model artifact...")
    model_path = os.path.join(args.models_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print("[6/7] Writing reports...")
    text_report = os.path.join(args.reports_dir, "metrics.txt")
    json_report = os.path.join(args.reports_dir, "metrics.json")

    with open(text_report, "w", encoding="utf-8") as f:
        f.write("Advanced Fake News Detection Report\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Best model: {best_name} (F1={best_score:.4f})\n\n")
        for name, res in all_results.items():
            f.write(f"== {name} ==\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Recall: {res['recall']:.4f}\n")
            f.write(f"F1: {res['f1']:.4f}\n")
            f.write(f"CV best score: {res['cv_best_score']:.4f}\n")
            f.write(f"Best params: {res['best_params']}\n")
            f.write(f"Confusion matrix: {res['confusion_matrix']}\n")
            f.write(res["report"] + "\n")
            f.write("-" * 60 + "\n")

        if top_terms:
            f.write("\nTop weighted terms from best model:\n")
            for cls, details in top_terms.items():
                f.write(f"Class '{cls}' strong positive terms: {', '.join(details['top_positive_terms'][:15])}\n")

    serializable_results = {
        "dataset": args.data_path,
        "rows": len(df),
        "classes": [str(c) for c in label_encoder.classes_],
        "best_model": best_name,
        "best_f1": best_score,
        "trained_at": artifact["trained_at"],
        "results": all_results,
        "top_terms": top_terms,
    }
    with open(json_report, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    print("[7/7] Done")
    print(f"Model artifact: {model_path}")
    print(f"Reports: {text_report}, {json_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an advanced fake news detection system.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--title_column", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=4)
    parser.add_argument("--search_iter", type=int, default=10)
    parser.add_argument("--top_k_terms", type=int, default=25)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()
    main(args)
