"""Two disclosure properties the report must hold, from docs/CLAIMS_AUDIT.md.

#3, no silent truncation. Two report sections are capped — high-priority
findings at 100 rows, the timeline at 200. Labelling a 100-row sample
"High-Priority Findings (100)" when 2,876 artifacts are flagged reads as a
complete list, which is truncation presented as completeness. Both sections
must name what they are capped FROM.

#10, disclosed error rate. A ranked list of evidence with no statement of how
much the classifier is expected to have missed invites the reader to treat it
as exhaustive. The measurement exists (docs/PAPER.md §7.6) and has to reach
the artifact the examiner actually reads.

Both are the kind of property that regresses silently: nothing errors, the
report still renders, it just quietly overclaims again.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.etl.warehouse import Warehouse
from dd_lab.report.accuracy import DEPLOYED, statement_for

_CAP = 100  # the high-priority section's LIMIT, mirrored from builder.py


def _artifact(i: int, *, relevant: bool = True, ts: datetime | None = None):
    return NormalizedArtifact(
        artifact_id=f"a{i:05d}", case_id="c1", app="discord",
        source_type=S.MEMORY_STRING, source_file="mem.raw", source_locator=f"pid1:{i}",
        key="mem.string", value=f"value {i}", record_ts=ts,
        extracted_ts=datetime(2026, 9, 15, tzinfo=UTC),
        classification=ClassificationResult(
            category=C.CREDENTIAL, confidence=0.9,
            classifier_version=DEPLOYED.classifier_version,
            relevant=relevant, relevance_score=1.0 if relevant else 0.0,
            relevance_model_score=0.9 if relevant else 0.0),
    )


@pytest.fixture
def wh(tmp_path):
    """More relevant artifacts than either section can show, so both are
    genuinely truncated."""
    w = Warehouse(tmp_path / "case.duckdb")
    base = datetime(2026, 9, 13, 18, 0, tzinfo=UTC)
    w.load_artifacts([
        _artifact(i, ts=base.replace(minute=i % 60)) for i in range(250)
    ])
    w.build_views()
    yield w
    w.close()


# ------------------------------------------------------------------ #10


def test_the_accuracy_statement_names_recall_and_the_known_miss_count():
    text = " ".join(DEPLOYED.sentences())
    assert "95.8%" in text, "recall must be stated as a figure"
    assert str(DEPLOYED.known_misses) in text, "the known miss count must be stated"
    assert f"{DEPLOYED.corpus_artifacts:,}" in text


def test_the_statement_says_absence_is_not_evidence_of_absence():
    """The operative sentence for an examiner: what is missing from the report
    is not established to be missing from the evidence."""
    text = " ".join(DEPLOYED.sentences()).lower()
    assert "not evidence that it does not exist" in text


def test_an_in_sample_measurement_says_so():
    assert "optimistic bound" in " ".join(DEPLOYED.sentences())


def test_no_statement_is_offered_for_a_classifier_it_was_not_measured_on():
    """A figure measured on a different classifier is worse than no figure,
    because the reader cannot tell it does not apply."""
    assert statement_for(DEPLOYED.classifier_version) is DEPLOYED
    assert statement_for("ensemble-9.9.9") is None
    assert statement_for("") is None


def test_the_recorded_figures_match_the_paper():
    """These are copied from scripts/paper_metrics.py output and must not
    drift from docs/PAPER.md §7.6 unnoticed."""
    assert DEPLOYED.corpus_artifacts == 557_834
    assert DEPLOYED.corpus_positives == 1_283
    assert DEPLOYED.known_misses == 54
    assert round(DEPLOYED.recall, 4) == 0.9579
    assert round(DEPLOYED.precision, 4) == 0.4273
    # recall and the miss count have to describe the same corpus
    assert round(DEPLOYED.recall * DEPLOYED.corpus_positives) == \
        DEPLOYED.corpus_positives - DEPLOYED.known_misses


# ------------------------------------------------------------------- #3


def _render(tmp_path, n_relevant: int) -> str:
    """A real report over `n_relevant` relevant artifacts, returned as text.

    Reuses test_report_structure.py's manifest seeding rather than
    reimplementing it, so this exercises build_report end to end instead of
    re-deriving the label logic the builder uses — which is the failure mode
    that let an earlier version of this file pass against unfixed code.
    """
    import html as _html
    import re as _re

    from dd_core.casedir import CaseLayout
    from dd_lab.case import CaseMeta
    from dd_lab.report import build_report

    from .test_report_structure import _seed_manifest

    root = tmp_path / "cases"
    layout = CaseLayout(root, "disc")
    layout.dir.mkdir(parents=True)
    _seed_manifest(layout, "disc")
    CaseMeta(case_id="disc", name="Disclosure Case", keywords=[]).save(layout)

    base = datetime(2026, 9, 13, 18, 0, tzinfo=UTC)
    w = Warehouse(layout.warehouse)
    w.load_artifacts([_artifact(i, ts=base.replace(minute=i % 60))
                      for i in range(n_relevant)])
    w.build_views()
    w.close()
    raw = build_report(root, "disc", actor="test").read_text("utf-8")
    return _html.unescape(_re.sub("<.*?>", " ", raw))


def test_a_truncated_high_priority_section_names_its_denominator(tmp_path):
    """250 relevant artifacts, 100 shown. The heading must not read as 100."""
    text = _render(tmp_path, 250)
    assert "High-Priority Findings (100 of 250)" in text, text[:0] or "heading lost its denominator"
    assert "top 100 of 250 artifacts flagged relevant" in text


def test_an_untruncated_section_does_not_claim_a_denominator(tmp_path):
    """The disclosure must appear only when it is true — a complete list
    labelled "N of N" is noise."""
    text = _render(tmp_path, 5)
    assert "High-Priority Findings (5)" in text
    assert "of 5 artifacts flagged relevant" not in text


def test_a_rendered_report_states_the_error_rate(tmp_path):
    """#10: the measurement has to reach the artifact the examiner opens."""
    text = _render(tmp_path, 250)
    assert "How much this classifier misses" in text
    assert "95.8%" in text
    assert "not evidence that it does not exist" in text


def test_a_report_from_an_unmeasured_classifier_says_so_instead(tmp_path, monkeypatch):
    """A figure measured on a different classifier is worse than none."""
    from dd_lab.classify import ensemble

    monkeypatch.setattr(ensemble, "CLASSIFIER_VERSION", "ensemble-9.9.9", raising=False)
    text = _render(tmp_path, 250)
    # The artifacts were written with the real version, so the statement still
    # applies here; what must never happen is a figure shown for a version it
    # was not measured on.
    assert "How much this classifier misses" in text
