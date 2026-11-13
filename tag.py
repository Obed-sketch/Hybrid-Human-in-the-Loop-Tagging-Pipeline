"""Tag incident text(s) with the trained classifier.

    python tag.py "Email asks me to verify my password on a strange site"
    python tag.py --file new_incidents.txt     # one incident per line

Prints the predicted category and the model's confidence (the winning
class probability) for each input. Train first: python train_classifier.py
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "tagger.joblib"


def load_model(path=DEFAULT_MODEL):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"model not found at {path}; run `python train_classifier.py` first")
    bundle = joblib.load(path)
    return bundle["model"]


def predict_with_confidence(model, texts):
    """Return [(label, confidence), ...] for each text."""
    probabilities = model.predict_proba(texts)
    classes = list(model.classes_)
    results = []
    for row in probabilities:
        row = np.asarray(row, dtype=float)
        best = int(row.argmax())
        results.append((classes[best], float(row[best])))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", help="incident text(s) to tag")
    parser.add_argument("--file", help="text file, one incident per line")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args(argv)

    texts = list(args.texts)
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        texts.extend(line.strip() for line in lines if line.strip())
    if not texts:
        parser.error("provide at least one incident text or --file")

    model = load_model(args.model)
    for text, (label, confidence) in zip(
            texts, predict_with_confidence(model, texts)):
        print(f"{label:18s} {confidence:.2f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
