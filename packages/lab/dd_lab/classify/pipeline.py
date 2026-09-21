"""Run the ensemble over artifacts, in-stream during ingest or as a re-run.

Classifying in the parse stream (`classify_stream`) means the warehouse columns
are populated at INSERT time — far cheaper than point-updating a columnar store
afterwards. `classify_warehouse` is the re-run path (e.g. after the agent tunes
the rules) and uses a single set-based UPDATE per batch.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from dd_core.schema import NormalizedArtifact
from dd_lab.classify.ensemble import CLASSIFIER_VERSION, Classifier
from dd_lab.etl.warehouse import Warehouse

log = logging.getLogger("dd_lab.classify")


@dataclass
class ClassifyResult:
    classified: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    classifier_version: str = CLASSIFIER_VERSION


def classify_stream(
    artifacts: Iterator[NormalizedArtifact],
    classifier: Classifier | None = None,
    tally: ClassifyResult | None = None,
) -> Iterator[NormalizedArtifact]:
    """Attach a ClassificationResult to each artifact as it flows past."""
    classifier = classifier or Classifier()
    for a in artifacts:
        a.classification = classifier.classify(
            a.key, a.value, a.value_type, app=a.app, source_type=a.source_type, deleted=a.deleted
        )
        if tally is not None:
            tally.classified += 1
            cat = str(a.classification.category)
            tally.by_category[cat] = tally.by_category.get(cat, 0) + 1
        yield a


def classify_warehouse(
    wh: Warehouse, classifier: Classifier | None = None
) -> ClassifyResult:
    """Re-classify every artifact already in the warehouse."""
    classifier = classifier or Classifier()
    result = ClassifyResult()
    for rows in wh.iter_for_classification():
        updates = []
        for r in rows:
            cr = classifier.classify(
                r["key"],
                r["value"],
                r["value_type"],
                app=r["app"],
                source_type=r["source_type"],
                deleted=r["deleted"],
            )
            cat = str(cr.category)
            result.by_category[cat] = result.by_category.get(cat, 0) + 1
            updates.append(
                (
                    r["artifact_id"],
                    {
                        "category": cat,
                        "confidence": cr.confidence,
                        "relevant": cr.relevant,
                        "relevance_score": cr.relevance_score,
                        "regex_vote": _s(cr.regex_vote),
                        "ner_vote": _s(cr.ner_vote),
                        "keyword_vote": _s(cr.keyword_vote),
                        "model_vote": _s(cr.model_vote),
                        "entropy_vote": _s(cr.entropy_vote),
                        "relevance_model_score": cr.relevance_model_score,
                        "evidence_spans": json.dumps(cr.evidence_spans, ensure_ascii=False),
                        "classifier_version": cr.classifier_version,
                        "classified_ts": cr.classified_ts,
                    },
                )
            )
        result.classified += wh.apply_classifications(updates)
    wh.build_views()
    log.info("re-classified %d artifacts: %s", result.classified, result.by_category)
    return result


def _s(v) -> str | None:
    return str(v) if v is not None else None
