"""Human-in-the-loop review queue for the incident tagger.

Score unlabeled incidents with the trained model. Confident predictions are
written to an auto-tagged file; anything below the confidence threshold is
routed to a review queue for a human analyst:

    python review_queue.py                      # score data/unlabeled_samples.csv
    python review_queue.py --threshold 0.70     # stricter routing

The review queue CSV has an empty `corrected_label` column. The reviewer
fills it in for items they re-tag (leaving it blank means "prediction was
right"), saves the file, and feeds the corrections back:

    python review_queue.py --feedback my_reviewed_queue.csv

That appends the corrected rows to data/feedback.csv. Retrain with
`python train_classifier.py --include-feedback` to close the loop.
"""
import argparse
from pathlib import Path

import pandas as pd

from tag import load_model, predict_with_confidence

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "unlabeled_samples.csv"
DEFAULT_AUTO_OUT = BASE_DIR / "data" / "auto_tagged.csv"
DEFAULT_QUEUE_OUT = BASE_DIR / "data" / "review_queue.csv"
DEFAULT_FEEDBACK = BASE_DIR / "data" / "feedback.csv"
VALID_LABELS = ["malware", "phishing", "intrusion",
                "policy_violation", "data_leak", "benign"]


def process(frame, model, threshold=0.60):
    """Split incidents into (auto_tagged, needs_review) by confidence."""
    results = predict_with_confidence(model, frame["text"].tolist())
    scored = frame.copy()
    scored["predicted_label"] = [label for label, _ in results]
    scored["confidence"] = [round(conf, 4) for _, conf in results]
    auto = scored[scored["confidence"] >= threshold]
    review = scored[scored["confidence"] < threshold].copy()
    review["corrected_label"] = ""
    return (auto.reset_index(drop=True), review.reset_index(drop=True))


def apply_feedback(corrections_path, feedback_path=DEFAULT_FEEDBACK):
    """Append human corrections from a filled-in review queue CSV.

    Rows with a non-empty, valid `corrected_label` become feedback rows
    (text, label). Returns how many rows were appended.
    """
    corrections = pd.read_csv(corrections_path)
    if "corrected_label" not in corrections.columns:
        raise ValueError("corrections file has no 'corrected_label' column")
    filled = corrections[corrections["corrected_label"].notna()]
    filled = filled[filled["corrected_label"].astype(str).str.strip() != ""]
    invalid = set(filled["corrected_label"]) - set(VALID_LABELS)
    if invalid:
        raise ValueError(f"invalid corrected_label value(s): {sorted(invalid)}")
    rows = pd.DataFrame({
        "text": filled["text"].astype(str),
        "label": filled["corrected_label"].astype(str),
    })
    feedback_path = Path(feedback_path)
    if feedback_path.exists():
        rows = pd.concat([pd.read_csv(feedback_path), rows], ignore_index=True)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(feedback_path, index=False)
    return len(filled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="CSV with a 'text' column of unlabeled incidents")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="confidence below this goes to the review queue "
                             "(default: 0.60)")
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "tagger.joblib"))
    parser.add_argument("--auto-out", default=str(DEFAULT_AUTO_OUT))
    parser.add_argument("--queue-out", default=str(DEFAULT_QUEUE_OUT))
    parser.add_argument("--feedback", metavar="CORRECTIONS_CSV",
                        help="append corrections from a filled-in review "
                             "queue CSV to data/feedback.csv, then exit")
    parser.add_argument("--feedback-file", default=str(DEFAULT_FEEDBACK))
    args = parser.parse_args(argv)

    if args.feedback:
        count = apply_feedback(args.feedback, args.feedback_file)
        print(f"Appended {count} correction(s) to {args.feedback_file}")
        print("Retrain with: python train_classifier.py --include-feedback")
        return 0

    frame = pd.read_csv(args.input)
    model = load_model(args.model)
    auto, review = process(frame, model, threshold=args.threshold)
    auto.to_csv(args.auto_out, index=False)
    review.to_csv(args.queue_out, index=False)
    print(f"Scored {len(frame)} incident(s) at threshold {args.threshold:.2f}:")
    print(f"  auto-tagged:   {len(auto)}  -> {args.auto_out}")
    print(f"  needs review:  {len(review)}  -> {args.queue_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
