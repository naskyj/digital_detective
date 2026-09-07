"""Artifact classification: regex + keyword + NER voting ensemble (+ optional ONNX model)."""

from __future__ import annotations

from dd_lab.classify.ensemble import CLASSIFIER_VERSION, Classifier
from dd_lab.classify.model import load_model_voter
from dd_lab.classify.pipeline import ClassifyResult, classify_stream, classify_warehouse
from dd_lab.classify.zeroshot import load_zeroshot_voter

__all__ = [
    "CLASSIFIER_VERSION",
    "Classifier",
    "ClassifyResult",
    "classify_stream",
    "classify_warehouse",
    "load_model_voter",
    "load_zeroshot_voter",
]


def build_classifier(model_dir: str | None = None, zeroshot: bool = False) -> Classifier:
    """Assemble a Classifier with whatever 4th voter is configured.

    Priority: a trained ONNX model if `model_dir` has one, else the pretrained
    zero-shot voter if `zeroshot` is set and installed, else rules-only.
    """
    voter = load_model_voter(model_dir) or load_zeroshot_voter(zeroshot)
    return Classifier(model_voter=voter)
