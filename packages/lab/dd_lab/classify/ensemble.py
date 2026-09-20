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
from dd_core.enums import SourceType, ValueType
from dd_core.schema import ClassificationResult
from dd_lab.classify.entropy import entropy_vote
from dd_lab.classify.gates import boosted_relevance, is_structurally_empty
from dd_lab.classify.infra_gazetteer import is_infra_noise
from dd_lab.classify.ner import ner_vote
from dd_lab.classify.rules import CATEGORY_PRIORITY, content_field_hits, keyword_hits, regex_hits

CLASSIFIER_VERSION = "ensemble-0.3.0"

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

# Per-source-type voter routing (classifier_backend_spec.md's "PER-SOURCE
# ROUTING" finding): on the collaborator's own back-test, structured
# key/value sources already reach ~97% recall / ~82% precision on schema-
# key + value-shape (regex/keyword) alone - adding the fuzzy voters there
# (NER, model, entropy) gained 0 recall and only cost precision, since a
# rogue vote on an already-precise structured record has nothing to rescue
# and only adds a chance of a false positive. So for these source types,
# NER/model/entropy are skipped outright, not just down-weighted - see
# Classifier.classify() below. Content-dependent sources (cache/memory/
# app_data) keep the full voter set, since that's where real recall was
# actually missing. MEMORY_PROCESS is structured key=value process
# metadata (PID/name/CreateTime, ...), not free text, so it's grouped with
# the structured tier even though it's a memory-derived source type.
_STRUCTURED_SOURCES = frozenset(
    {
        SourceType.LOCAL_STORAGE,
        SourceType.SESSION_STORAGE,
        SourceType.INDEXEDDB,
        SourceType.COOKIES,
        SourceType.PREFERENCES,
        SourceType.LOCAL_STATE,
        SourceType.MEMORY_PROCESS,
    }
)
_CONTENT_SOURCES = frozenset(
    {
        SourceType.CACHE,
        SourceType.MEMORY_STRING,
        SourceType.MEMORY_ARTIFACT,
        SourceType.MEMORY_NETWORK,
        SourceType.APP_DATA,
    }
)


def _combine(scores: dict[C, float]) -> tuple[C, float, float]:
    """(category, confidence, relevance_score). `relevance_score` is the
    winning category's own raw accumulated score, capped to [0, 1] - the
    same signal `confidence` already partly uses, but as its own,
    unconflated value, and pinned to exactly 0.0 whenever GENERAL wins
    (never a positive "relevance" for the category that means "nothing
    said this mattered"). `category`/`confidence` are unchanged - relevance
    is read from the same combine step, not a second ensemble pass."""
    if not scores:
        return C.GENERAL, 0.0, 0.0
    top = max(scores.items(), key=lambda kv: (kv[1], -_PRIORITY_INDEX.get(kv[0], 99)))
    cat, raw = top
    total = sum(scores.values())
    agreement = raw / total if total else 0.0
    confidence = max(0.0, min(1.0, 0.5 * min(raw, 1.0) + 0.5 * agreement))
    relevance = 0.0 if cat == C.GENERAL else round(min(1.0, raw), 3)
    return cat, round(confidence, 3), relevance


class Classifier:
    """Stateless apart from the lazily-loaded NER/ONNX models."""

    def __init__(self, model_voter=None) -> None:
        self._model_voter = model_voter  # callable(key, value) -> (C, float) | None

    def classify(
        self,
        key: str,
        value: str,
        value_type: ValueType | str = ValueType.STRING,
        *,
        app: str | None = None,
        source_type: SourceType | str | None = None,
        deleted: bool = False,
    ) -> ClassificationResult:
        # Normalize source_type once - classify_warehouse's re-classify path
        # reads it back out of DuckDB as a plain string, not a SourceType
        # instance (see pipeline.py). An unrecognized/absent value falls
        # through to `None`, which routes as "run every voter" below - the
        # same behavior as every pre-phase-4 caller that never passed
        # source_type at all, so an unknown source type never silently
        # loses recall by being mis-routed into the structured tier.
        try:
            st = SourceType(source_type) if source_type is not None else None
        except ValueError:
            st = None
        structured = st in _STRUCTURED_SOURCES
        # Content-field flattening (below) is gated on explicit membership
        # in _CONTENT_SOURCES, unlike the NER/model/entropy routing above -
        # `source_type=None` (every pre-phase-5 caller) does NOT get it. A
        # brand-new, JSON-parsing-on-every-value feature defaults off for
        # "unspecified", rather than on-by-default like the pre-existing
        # voters that predate this phase.
        content_tier = st in _CONTENT_SOURCES

        # Empty-body pre-gate (classifier_backend_spec.md): runs FIRST,
        # before any voter, and overrides everything - a structurally empty
        # value carries no evidence no matter what ID the key/URL contains.
        if is_structurally_empty(value):
            return ClassificationResult(
                category=C.GENERAL,
                confidence=0.0,
                relevant=False,
                relevance_score=0.0,
                classifier_version=CLASSIFIER_VERSION,
                classified_ts=datetime.now(UTC),
            )

        scores: dict[C, float] = defaultdict(float)
        spans: list[str] = []

        rx = regex_hits(key, value, app)
        regex_vote = _best(rx)
        for hit in rx:
            scores[hit.category] += _W_REGEX * hit.weight
            spans.append(hit.span)

        kw = keyword_hits(key, value, app)
        if content_tier:
            # Content-bearing field-name gazetteer, applied to flattened
            # JSON leaf keys (rules.py::content_field_hits) - the spec's
            # single biggest validated cache-recall fix (+7pts, precision
            # held). Mechanically a keyword hit (a lexicon match on a field
            # name), so it's unioned straight into `kw` rather than tracked
            # as a separate signal - reuses _W_KEYWORD below like every
            # other keyword-mechanism hit.
            kw = kw + content_field_hits(value)
        keyword_vote = _best(kw)
        for hit in kw:
            scores[hit.category] += _W_KEYWORD * hit.weight

        # Structured sources: literally don't call these three voters (not
        # just zero-weight them) - see _STRUCTURED_SOURCES' comment above.
        # Skipping outright also means GLiNER/ONNX inference is never
        # invoked for the ~70%+ of rows that are structured, a real perf
        # win alongside the precision one.
        ner_cat = None
        model_cat = None
        entropy_cat = None
        if not structured:
            nv = ner_vote(key, value)
            if nv:
                ner_cat, ner_strength = nv
                scores[ner_cat] += _W_NER * ner_strength

            if self._model_voter is not None:
                mv = self._model_voter(key, value)
                if mv:
                    model_cat, model_strength = mv
                    scores[model_cat] += _W_MODEL * model_strength

            ev = entropy_vote(key, value)
            if ev:
                entropy_cat, entropy_strength = ev
                scores[entropy_cat] += _W_ENTROPY * entropy_strength

        # A bare timestamp / number with no other signal is TIMESTAMP / GENERAL.
        if not scores and str(value_type) == ValueType.TIMESTAMP:
            scores[C.TIMESTAMP] = 0.6

        category, confidence, relevance_score = _combine(scores)

        # ID-in-plumbing carve-out (classifier_backend_spec.md): an id
        # riding along in app plumbing or a telemetry URL isn't evidence -
        # overrides whatever the voters above produced, same override tier
        # as the empty-body gate but necessarily checked after voting since
        # it needs the accumulated result, not just the raw key/value.
        if is_infra_noise(key, value):
            category, confidence, relevance_score = C.GENERAL, 0.0, 0.0

        relevance_score = boosted_relevance(
            relevance_score, deleted=deleted, is_general=category == C.GENERAL
        )
        return ClassificationResult(
            category=category,
            confidence=confidence,
            relevant=category != C.GENERAL,
            relevance_score=relevance_score,
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
