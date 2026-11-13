"""Tests for the human-in-the-loop tagging pipeline.

Trains a real model on the seed data (300 short rows; trains in ~a second)
and checks the three behaviours the demo promises: training produces a
loadable model file, an obvious phishing text is tagged as phishing, and
the review-queue routing splits by confidence threshold.
"""
import joblib
import pandas as pd
import pytest

from review_queue import apply_feedback, process
from tag import load_model, predict_with_confidence
from train_classifier import load_training_frame, save_model, train_and_evaluate

PHISHING_TEXT = ("URGENT: Your account has been suspended. Click this link "
                 "immediately to verify your password and restore access "
                 "before it is permanently closed.")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    frame = load_training_frame()
    pipeline, metrics = train_and_evaluate(frame)
    model_path = tmp_path_factory.mktemp("models") / "tagger.joblib"
    save_model(pipeline, model_path)
    return pipeline, metrics, model_path


def test_training_produces_loadable_model_file(trained):
    _, _, model_path = trained
    assert model_path.exists()
    bundle = joblib.load(model_path)
    assert set(bundle["labels"]) == {
        "malware", "phishing", "intrusion",
        "policy_violation", "data_leak", "benign"}
    model = load_model(model_path)  # tag.py's loader accepts the file
    assert model is not None


def test_training_metrics_are_sane(trained):
    _, metrics, _ = trained
    # Sanity floor only: a working pipeline must score far above the 1/6
    # chance level. The exact measured score is printed by
    # train_classifier.py and quoted in the README (0.77 / 0.76 on the
    # final seed set); the categories genuinely overlap in vocabulary, so
    # deliberately does not demand a near-perfect score.
    assert metrics["accuracy"] > 0.70
    assert metrics["weighted_f1"] > 0.70


def test_obvious_phishing_is_tagged_phishing(trained):
    pipeline, _, _ = trained
    [(label, confidence)] = predict_with_confidence(pipeline, [PHISHING_TEXT])
    assert label == "phishing"
    assert confidence > 0.60


class UniformStubModel:
    """Stand-in model: spreads probability almost evenly (max 0.30),
    so every prediction is low-confidence and must be routed to review."""

    classes_ = ["malware", "phishing", "intrusion",
                "policy_violation", "data_leak", "benign"]

    def predict_proba(self, texts):
        row = [0.30, 0.14, 0.14, 0.14, 0.14, 0.14]
        return [list(row) for _ in texts]


class ConfidentStubModel:
    """Stand-in model: 0.95 on 'malware' for every text."""

    classes_ = UniformStubModel.classes_

    def predict_proba(self, texts):
        row = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
        return [list(row) for _ in texts]


def test_low_confidence_items_are_routed_to_review():
    frame = pd.DataFrame({"text": ["something odd happened",
                                   "another vague report"]})
    auto, review = process(frame, UniformStubModel(), threshold=0.60)
    assert len(auto) == 0
    assert len(review) == 2
    assert list(review["corrected_label"]) == ["", ""]


def test_high_confidence_items_are_auto_tagged():
    frame = pd.DataFrame({"text": ["something odd happened"]})
    auto, review = process(frame, ConfidentStubModel(), threshold=0.60)
    assert len(auto) == 1
    assert len(review) == 0
    assert auto.iloc[0]["predicted_label"] == "malware"


def test_feedback_appends_corrections(tmp_path):
    corrections = tmp_path / "reviewed.csv"
    pd.DataFrame({
        "text": ["odd thing one", "odd thing two"],
        "predicted_label": ["benign", "malware"],
        "confidence": [0.4, 0.5],
        "corrected_label": ["intrusion", ""],  # blank = prediction was right
    }).to_csv(corrections, index=False)
    feedback_path = tmp_path / "feedback.csv"
    appended = apply_feedback(corrections, feedback_path)
    assert appended == 1
    saved = pd.read_csv(feedback_path)
    assert saved.to_dict("records") == [
        {"text": "odd thing one", "label": "intrusion"}]
