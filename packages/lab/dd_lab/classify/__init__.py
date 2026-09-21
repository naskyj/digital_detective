"""Artifact classification: regex + keyword + NER voting ensemble (+ optional ONNX models)."""

from __future__ import annotations

from dd_lab.classify.ensemble import CLASSIFIER_VERSION, Classifier
from dd_lab.classify.model import load_model_voter, load_relevance_voter
from dd_lab.classify.pipeline import ClassifyResult, classify_stream, classify_warehouse
from dd_lab.classify.zeroshot import load_zeroshot_voter

__all__ = [
    "CLASSIFIER_VERSION",
    "Classifier",
    "ClassifyResult",
    "classify_stream",
    "classify_warehouse",
    "load_model_voter",
    "load_relevance_voter",
    "load_zeroshot_voter",
]


def build_classifier(
    model_dir: str | None = None, relevance_model_dir: str | None = None, zeroshot: bool = False
) -> Classifier:
    """Assemble a Classifier with whatever category/relevance voters are configured.

    `model_dir` (category voter): a trained ONNX model if configured, else
    the pretrained zero-shot voter if `zeroshot` is set and installed, else
    none - category assignment falls back to the rule-based voters either
    way. `relevance_model_dir` (relevance voter, a separate slot/model -
    see ensemble.py's module docstring and model.py's OnnxRelevanceModel):
    a trained binary relevant/not_relevant ONNX model if configured, else
    relevance is computed from rule evidence alone.
    """
    voter = load_model_voter(model_dir) or load_zeroshot_voter(zeroshot)
    relevance_voter = load_relevance_voter(relevance_model_dir)
    return Classifier(model_voter=voter, relevance_voter=relevance_voter)
