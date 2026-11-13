"""Train the incident auto-tagger (TF-IDF + Logistic Regression baseline).

    python train_classifier.py                      # train on data/incidents.csv
    python train_classifier.py --include-feedback   # also fold in data/feedback.csv

Evaluates on a stratified held-out split, prints accuracy, weighted F1 and
the full classification report, and saves the fitted pipeline (vectorizer +
model + label list) to models/tagger.joblib for tag.py and review_queue.py.

The seed data in data/incidents.csv is a small hand-built demonstration set,
so expect optimistic scores: it proves the loop works end-to-end, not that
the model is production-ready.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "incidents.csv"
DEFAULT_FEEDBACK = BASE_DIR / "data" / "feedback.csv"
DEFAULT_MODEL = BASE_DIR / "models" / "tagger.joblib"

LABELS = ["malware", "phishing", "intrusion",
          "policy_violation", "data_leak", "benign"]


def load_training_frame(data_path=DEFAULT_DATA, feedback_path=DEFAULT_FEEDBACK,
                        include_feedback=False):
    """Load the seed set, optionally appending human-corrected feedback rows."""
    frame = pd.read_csv(data_path)[["text", "label"]]
    if include_feedback and Path(feedback_path).exists():
        feedback = pd.read_csv(feedback_path)[["text", "label"]]
        feedback = feedback[feedback["label"].isin(LABELS)]
        if len(feedback):
            frame = pd.concat([frame, feedback], ignore_index=True)
    return frame


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=10.0)),
    ])


def train_and_evaluate(frame, test_size=0.2, seed=42):
    """Fit on a stratified train split; return (fitted_pipeline, metrics)."""
    train, test = train_test_split(
        frame, test_size=test_size, random_state=seed,
        stratify=frame["label"])
    pipeline = build_pipeline()
    pipeline.fit(train["text"], train["label"])
    predicted = pipeline.predict(test["text"])
    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "accuracy": float(accuracy_score(test["label"], predicted)),
        "weighted_f1": float(f1_score(test["label"], predicted,
                                      average="weighted")),
        "report": classification_report(test["label"], predicted,
                                        zero_division=0),
    }
    return pipeline, metrics


def save_model(pipeline, path=DEFAULT_MODEL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "labels": LABELS}, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--feedback-file", default=str(DEFAULT_FEEDBACK))
    parser.add_argument("--include-feedback", action="store_true",
                        help="append data/feedback.csv rows (human "
                             "corrections) to the training data")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    frame = load_training_frame(args.data, args.feedback_file,
                                include_feedback=args.include_feedback)
    pipeline, metrics = train_and_evaluate(frame, test_size=args.test_size,
                                           seed=args.seed)
    save_model(pipeline, args.model_out)

    print(f"Training rows: {metrics['n_train']}  Test rows: {metrics['n_test']}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print()
    print(metrics["report"])
    print(f"Model saved to {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
