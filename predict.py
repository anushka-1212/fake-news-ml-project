#!/usr/bin/env python3
"""CLI inference for saved fake news model."""

import argparse
import pickle


def prettify(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"fake", "false", "0"}:
        return "Fake"
    if s in {"real", "true", "1"}:
        return "Real"
    return str(label)


def main(args):
    with open(args.model_path, "rb") as f:
        artifact = pickle.load(f)

    pipeline = artifact["pipeline"]
    encoder = artifact["label_encoder"]
    pred = pipeline.predict([args.text])[0]
    pred_label = encoder.inverse_transform([pred])[0]

    print(f"Prediction: {prettify(pred_label)}")
    print(f"Raw label: {pred_label}")

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([args.text])[0]
        print("Confidence:")
        for cls, prob in zip(encoder.classes_, proba):
            print(f"  {prettify(cls)}: {prob * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fake/real class for text.")
    parser.add_argument("--model_path", default="models/best_model.pkl")
    parser.add_argument("--text", required=True)
    main(parser.parse_args())
