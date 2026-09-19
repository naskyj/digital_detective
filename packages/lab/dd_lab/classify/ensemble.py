"""Voting ensemble: regex + keyword + NER (+ optional ONNX model) -> category.

Mirrors the original system's design (regex_vote / ner_vote / keyword_vote
combined by a vote), with an optional fourth model vote once a trained
transformer is available. Each voter proposes a category and a strength in
[0, 1]; strengths for the same category are combined, the top category wins,
ties break by CATEGORY_PRIORITY, and confidence reflects both the winning score
and how much the voters agreed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import ValueType
from dd_core.schema import ClassificationResult
from dd_lab.classify.entropy import entropy_vote
from dd_lab.classify.ner import ner_vote
from dd_lab.classify.rules import CATEGORY_PRIORITY, keyword_hits, regex_hits

CLASSIFIER_VERSION = "ensemble-0.2.0"

_PRIORITY_INDEX = {c: i for i, c in enumerate(CATEGORY_PRIORITY)}

# How much each voter counts toward the combined score.
_W_REGEX = 1.0
_W_KEYWORD = 0.85
_W_NER = 0.7
_W_MODEL = 1.1
# Weakest voter, deliberately - a value-shape heuristic with no key-name or
# keyword context at all (see entropy.py's own docstring). Its job is to
# break ties in favour of/against GENERAL when nothing else has a strong
# opinion, not to override a real signal from another voter.
_W_ENTROPY = 0.5


def _combine(scores: dict[C, float]) -> tuple[C, float]:
    if not scores:
        return C.GENERAL, 0.0
    top = max(scores.items(), key=lambda kv: (kv[1], -_PRIORITY_INDEX.get(kv[0], 99)))
    cat, raw = top
    total = sum(scores.values())
    agreement = raw / total if total else 0.0
    confidence = max(0.0, min(1.0, 0.5 * min(raw, 1.0) + 0.5 * agreement))
    return cat, round(confidence, 3)


class Classifier:
    """Stateless apart from the lazily-loaded NER/ONNX models."""

    def __init__(self, model_voter=None) -> None:
        self._model_voter = model_voter  # callable(key, value) -> (C, float) | None

    def classify(
        self, key: str, value: str, value_type: ValueType | str = ValueType.STRING
    ) -> ClassificationResult:
        scores: dict[C, float] = defaultdict(float)
        spans: list[str] = []

        rx = regex_hits(key, value)
        regex_vote = _best(rx)
        for hit in rx:
            scores[hit.category] += _W_REGEX * hit.weight
            spans.append(hit.span)

        kw = keyword_hits(key, value)
        keyword_vote = _best(kw)
        for hit in kw:
            scores[hit.category] += _W_KEYWORD * hit.weight

        nv = ner_vote(key, value)
        ner_cat = None
        if nv:
            ner_cat, ner_strength = nv
            scores[ner_cat] += _W_NER * ner_strength

        model_cat = None
        if self._model_voter is not None:
            mv = self._model_voter(key, value)
            if mv:
                model_cat, model_strength = mv
                scores[model_cat] += _W_MODEL * model_strength

        ev = entropy_vote(key, value)
        entropy_cat = None
        if ev:
            entropy_cat, entropy_strength = ev
            scores[entropy_cat] += _W_ENTROPY * entropy_strength

        # A bare timestamp / number with no other signal is TIMESTAMP / GENERAL.
        if not scores and str(value_type) == ValueType.TIMESTAMP:
            scores[C.TIMESTAMP] = 0.6

        category, confidence = _combine(scores)
        return ClassificationResult(
            category=category,
            confidence=confidence,
            regex_vote=regex_vote,
            keyword_vote=keyword_vote,
            ner_vote=ner_cat,
            model_vote=model_cat,
            entropy_vote=entropy_cat,
            evidence_spans=spans[:5],
            classifier_version=CLASSIFIER_VERSION,
            classified_ts=datetime.now(UTC),
        )


def _best(hits) -> C | None:
    if not hits:
        return None
    return max(hits, key=lambda h: h.weight).category
