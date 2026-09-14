"""Report integrity seal — the generated report gets the same hash-chained
custody protection as the evidence, so tampering with a report copy after the
fact is detectable the same way tampering with evidence is."""

from __future__ import annotations

from dd_core.casedir import CaseLayout
from dd_lab.report.builder import seal_report, verify_report


def test_seal_then_verify_ok(tmp_path):
    layout = CaseLayout(tmp_path, "case1")
    layout.reports.mkdir(parents=True)
    report = layout.reports / "forensic_report_x.html"
    report.write_text("<html>original</html>")

    digest = seal_report(layout, report, actor="test")
    assert len(digest) == 64

    ok, detail = verify_report(layout, report)
    assert ok, detail


def test_verify_detects_post_generation_edit(tmp_path):
    layout = CaseLayout(tmp_path, "case1")
    layout.reports.mkdir(parents=True)
    report = layout.reports / "forensic_report_x.html"
    report.write_text("<html>original</html>")
    seal_report(layout, report, actor="test")

    report.write_text("<html>tampered — a finding quietly removed</html>")

    ok, detail = verify_report(layout, report)
    assert not ok
    assert "mismatch" in detail


def test_verify_unsealed_report_fails(tmp_path):
    layout = CaseLayout(tmp_path, "case1")
    layout.reports.mkdir(parents=True)
    report = layout.reports / "never_sealed.html"
    report.write_text("<html></html>")

    ok, detail = verify_report(layout, report)
    assert not ok
    assert "no seal recorded" in detail


def test_verify_missing_report_fails(tmp_path):
    layout = CaseLayout(tmp_path, "case1")
    layout.reports.mkdir(parents=True)
    report = layout.reports / "forensic_report_x.html"
    report.write_text("<html>original</html>")
    seal_report(layout, report, actor="test")
    report.unlink()

    ok, detail = verify_report(layout, report)
    assert not ok
    assert "missing" in detail
