"""Generate the HTML forensic report from a case warehouse + logs.

Self-contained single file (inline CSS), suitable for disclosure. Content:
case context, evidence inventory, integrity attestation, artifact-category
breakdown, an evidence timeline, high-priority findings, keyword hits, and
chain-of-custody + agent-audit appendices.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

from dd_core.casedir import CaseLayout
from dd_core.custody import CustodyLog
from dd_core.enums import CustodyEvent
from dd_core.hashing import sha256_file
from dd_core.manifest import Manifest, verify_case
from dd_lab.agent.audit import AuditLog
from dd_lab.agent.findings import FindingsStore
from dd_lab.case import CaseMeta
from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate import build_correlation_report

_CSS = """
:root{--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--accent:#8b1a1a;--hi:#fff4f4}
*{box-sizing:border-box}body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);margin:0;background:#fafafa}
.wrap{max-width:960px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 10px;border-bottom:2px solid var(--accent);padding-bottom:4px}
h3{font-size:14px;margin:20px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--muted);margin:0 0 24px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#f0f0f0;font-weight:600}
.kv{display:grid;grid-template-columns:180px 1fr;gap:2px 16px;margin:8px 0}
.kv div:nth-child(odd){color:var(--muted)}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;background:#eee;font-size:12px}
.ok{background:#e6f5e6;color:#1a6b1a}.bad{background:#fbe4e4;color:#9b1a1a}.warn{background:#fff3d6;color:#8a5a00}
.hi{background:var(--hi)}
code{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12px;word-break:break-all}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
footer{margin-top:50px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
"""


def _esc(v: object) -> str:
    return html.escape(str(v)) if v is not None else ""


def _rows(headers: list[str], rows: list[list], highlight: bool = False) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        cls = ' class="hi"' if highlight else ""
        body += "<tr" + cls + ">" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_report(
    cases_root: str | Path, case_id: str, *, actor: str = "dd-lab"
) -> Path:
    layout = CaseLayout(cases_root, case_id)
    meta = CaseMeta.load_or_none(layout) or CaseMeta(case_id=case_id, name=case_id)
    manifest = Manifest.read(layout.dir)
    verification = verify_case(layout.dir)
    findings = FindingsStore(layout.findings_log).list()
    wh = Warehouse(layout.warehouse)

    try:
        wh.build_views()
        total = wh.count()
        by_cat = wh.query("SELECT category, artifacts, round(mean_confidence,2) AS conf FROM v_by_category")
        by_src = wh.query("SELECT app, source_type, artifacts FROM v_source_breakdown")
        timeline = wh.query(
            "SELECT strftime(record_ts,'%Y-%m-%d %H:%M') AS ts, app, category, "
            "substr(key,1,50) AS key, substr(value,1,80) AS value "
            "FROM v_timeline WHERE relevant LIMIT 120"
        )
        high = wh.query(
            "SELECT category, round(confidence,2) AS conf, app, substr(key,1,60) AS key, "
            "substr(value,1,120) AS value FROM v_high_priority LIMIT 100"
        )
        kw_hits = _keyword_hits(wh, meta.keywords)
        correlations = build_correlation_report(wh)
    finally:
        wh.close()

    parts: list[str] = [f"<style>{_CSS}</style><div class='wrap'>"]
    parts.append(f"<h1>Forensic Analysis Report</h1><p class='sub'>{_esc(meta.name)} &middot; case <code>{_esc(case_id)}</code></p>")

    sig = "<span class='pill ok'>verified</span>" if verification.ok else "<span class='pill bad'>FAILED</span>"
    parts.append("<h2>1. Case &amp; Evidence</h2>")
    parts.append("<div class='kv'>")
    parts.append(f"<div>Case name</div><div>{_esc(meta.name)}</div>")
    parts.append(f"<div>Description</div><div>{_esc(meta.description) or '&mdash;'}</div>")
    parts.append(f"<div>Investigation keywords</div><div>{_esc(', '.join(meta.keywords)) or '&mdash;'}</div>")
    parts.append(f"<div>Acquisition host</div><div>{_esc(manifest.host.hostname)} ({_esc(manifest.host.os)} {_esc(manifest.host.os_version)})</div>")
    parts.append(f"<div>Acquired</div><div>{_esc(manifest.started_ts)} &rarr; {_esc(manifest.completed_ts)}</div>")
    parts.append(f"<div>Collector</div><div>{_esc(manifest.collector_version)}, operator {_esc(manifest.operator)}</div>")
    parts.append(f"<div>Evidence files</div><div>{manifest.entry_count} files, {manifest.total_bytes:,} bytes</div>")
    parts.append(f"<div>Integrity</div><div>{sig} &mdash; {verification.checked} files re-hashed against the signed manifest</div>")
    parts.append(f"<div>Applications</div><div>{_esc(', '.join(a.name for a in manifest.target_apps))}</div>")
    parts.append(f"<div>Total artifacts</div><div>{total:,}</div>")
    parts.append("</div>")

    if not verification.ok:
        parts.append("<p class='pill bad'>Evidence integrity could not be confirmed. "
                     f"Signature valid: {verification.signature_valid}. Issues: "
                     f"{_esc(verification.issues[:10])}</p>")

    parts.append("<h2>2. Artifact Categories</h2>")
    parts.append(_rows(["Category", "Artifacts", "Mean confidence"],
                       [[c["category"], f"{c['artifacts']:,}", c["conf"]] for c in by_cat]))
    parts.append("<h3>By source</h3>")
    parts.append(_rows(["App", "Source", "Artifacts"],
                       [[c["app"], c["source_type"], f"{c['artifacts']:,}"] for c in by_src]))

    parts.append(f"<h2>3. High-Priority Findings ({len(high)})</h2>")
    parts.append("<p class='sub'>Credentials, payment data, location data and chat content, ranked by sensitivity and confidence.</p>")
    parts.append(_rows(["Category", "Conf", "App", "Key", "Value"],
                       [[h["category"], h["conf"], h["app"], h["key"], h["value"]] for h in high],
                       highlight=True) if high else "<p>None.</p>")

    parts.append(f"<h2>4. Findings ({len(findings)})</h2>")
    parts.append(
        "<p class='sub'>Every conclusion the analysis produced, with the citations it rests on. "
        "Each citation was independently re-verified against the warehouse and its value/hash "
        "snapshotted at write time; a finding with any failing citation is rejected outright and "
        "never reaches this report as anything but a rejected entry. \"Judge\" is an independent "
        "model's adversarial review, not the same model that produced the finding. No finding here "
        "is itself a legal conclusion — that determination is left to the investigator.</p>"
    )
    if findings:
        for f in findings:
            status_cls = {
                "human_approved": "ok", "judge_accepted": "ok",
                "rejected_unverified": "bad", "judge_rejected": "bad", "human_rejected": "bad",
                "judge_flagged": "warn", "verified": "warn",
            }.get(str(f.status), "")
            parts.append(
                f"<h3><span class='pill {status_cls}'>{_esc(f.status)}</span> {_esc(f.claim)}</h3>"
            )
            meta_bits = [f"by {_esc(f.created_by)}", _esc(f.created_ts)]
            if f.entity_type:
                meta_bits.append(f"entity: {_esc(f.entity_type)}={_esc(f.entity_value)}")
            parts.append(f"<p class='sub'>{' &middot; '.join(meta_bits)}</p>")
            parts.append(_rows(
                ["OK", "App", "File", "Locator", "Quote / reason"],
                [[("✓" if c.ok else "✗"), c.app, c.source_file, c.source_locator,
                  c.quote if c.ok else c.reason] for c in f.citations],
            ))
            if f.judge_verdict:
                parts.append(
                    f"<p><b>Judge</b> ({_esc(f.judge_verdict.model)}): "
                    f"<span class='pill {'ok' if f.judge_verdict.verdict == 'accept' else 'warn' if f.judge_verdict.verdict == 'flag' else 'bad'}'>"
                    f"{_esc(f.judge_verdict.verdict)}</span> &mdash; {_esc(f.judge_verdict.reasoning)}</p>"
                )
            else:
                parts.append("<p class='sub'>Not yet judged (no model reachable, or run without one).</p>")
            if f.human_review:
                parts.append(
                    f"<p><b>Human review</b> by {_esc(f.human_review.reviewer)}: "
                    f"<span class='pill {'ok' if f.human_review.decision == 'approve' else 'bad'}'>"
                    f"{_esc(f.human_review.decision)}</span> &mdash; {_esc(f.human_review.note)}</p>"
                )
    else:
        parts.append("<p>No findings were written for this case.</p>")

    parts.append(f"<h2>5. Cross-Source Correlations ({len(correlations)})</h2>")
    parts.append(
        "<p class='sub'>Identifiers (ids, invite links, emails, asset ids) that appear in more "
        "than one application or storage mechanism — a link a single artifact cannot show on its "
        "own. Each row cites its source file and in-file locator; times are the record's own "
        "timestamp, oldest first, as a reconstructed sequence for that identifier. An entity is "
        "only ever found by scanning independently RELEVANT artifacts (see the Relevant column "
        "below), but once found, every occurrence of it anywhere in the case is cited as "
        "corroborating evidence, relevant or not — a plumbing/telemetry match sharing the same "
        "value is kept, not hidden, just clearly marked apart from a genuinely relevant hit.</p>"
    )
    if correlations:
        for e in correlations:
            parts.append(
                f"<h3>{_esc(e.entity_type)} &mdash; <code>{_esc(e.value)}</code> "
                f"({len(e.artifacts)} artifacts across {_esc(', '.join(e.apps))})</h3>"
            )
            parts.append(_rows(
                ["Relevant", "Timestamp", "App", "Source", "File", "Locator", "Key", "Value"],
                [["✓" if c.relevant else "—", c.record_ts, c.app, c.source_type, c.source_file,
                  c.source_locator, c.key, c.value[:160]] for c in e.timeline()],
                highlight=True,
            ))
    else:
        parts.append("<p>No identifier was found in more than one app or storage mechanism.</p>")

    if meta.keywords:
        parts.append(f"<h2>6. Investigation Keyword Hits ({sum(k['hits'] for k in kw_hits)})</h2>")
        parts.append(_rows(["Keyword", "Hits", "Relevant hits", "Example"],
                           [[k["keyword"], k["hits"], k["relevant_hits"], k["example"]]
                            for k in kw_hits]))

    parts.append(f"<h2>7. Evidence Timeline ({len(timeline)} of the earliest non-generic events)</h2>")
    parts.append(_rows(["Timestamp", "App", "Category", "Key", "Value"],
                       [[t["ts"], t["app"], t["category"], t["key"], t["value"]] for t in timeline])
                 if timeline else "<p>No timestamped artifacts.</p>")

    parts.append("<h2>Appendix A &mdash; Chain of Custody</h2>")
    coc = CustodyLog(layout.dir)
    ok, bad = coc.verify_chain()
    chain_badge = (
        "<span class='pill ok'>intact</span>" if ok
        else f"<span class='pill bad'>broken at #{bad}</span>"
    )
    parts.append(f"<p>Custody chain: {chain_badge}</p>")
    parts.append(_rows(["#", "Time", "Event", "Actor", "Detail"],
                       [[r.seq, r.ts, r.event, r.actor, r.detail] for r in coc.entries()]))

    parts.append("<h2>Appendix B &mdash; Analysis Audit</h2>")
    audit = AuditLog(layout.audit_log).records()
    parts.append(_rows(["#", "Time", "Kind", "Name", "Detail"],
                       [[r.seq, r.ts, r.kind, r.name, r.detail] for r in audit])
                 if audit else "<p>No agent audit log (deterministic run).</p>")

    parts.append(f"<footer>Generated {datetime.now(UTC).isoformat(timespec='seconds')} by Digital Detective. "
                 f"Classifier: ensemble. This report is produced from acquired evidence and does not itself "
                 f"constitute a legal conclusion.</footer></div>")

    layout.reports.mkdir(parents=True, exist_ok=True)
    out = layout.reports / f"forensic_report_{datetime.now(UTC):%Y%m%d_%H%M%S}.html"
    out.write_text("".join(parts), encoding="utf-8")
    seal_report(layout, out, actor=actor)
    return out


def seal_report(layout: CaseLayout, report_path: Path, *, actor: str = "dd-lab") -> str:
    """Hash the generated report and log that hash into the same hash-chained
    custody log that protects the evidence, so the report's own integrity is
    provable later the same way the evidence's is — a copy edited after
    generation no longer matches its sealed hash."""
    digest = sha256_file(report_path)
    CustodyLog(layout.dir).append(
        CustodyEvent.REPORT_GENERATED, actor, report_path.name,
        data={"filename": report_path.name, "sha256": digest, "size_bytes": report_path.stat().st_size},
    )
    return digest


def verify_report(layout: CaseLayout, report_path: str | Path) -> tuple[bool, str]:
    """Re-hash `report_path` and check it against the seal recorded for it in
    the custody log. Returns (ok, detail)."""
    report_path = Path(report_path)
    sealed = [
        r for r in CustodyLog(layout.dir).entries()
        if r.event == CustodyEvent.REPORT_GENERATED and r.data.get("filename") == report_path.name
    ]
    if not sealed:
        return False, f"no seal recorded for {report_path.name}"
    if not report_path.is_file():
        return False, f"{report_path.name} is missing"
    current = sha256_file(report_path)
    sealed_hash = sealed[-1].data.get("sha256")
    if current != sealed_hash:
        return (
            False,
            f"hash mismatch: sealed {sealed_hash[:16]}… vs current {current[:16]}… "
            "— report was modified after generation",
        )
    return True, f"matches the seal recorded {sealed[-1].ts}"


def _keyword_hits(wh: Warehouse, keywords: list[str]) -> list[dict]:
    hits = []
    for kw in keywords:
        # Not filtered to relevant-only - an investigation keyword search
        # is deliberately broader than the primary relevance signal (the
        # investigator chose these terms specifically, independent of what
        # the ensemble scored) - but the relevant-hit count is surfaced
        # alongside the raw one so a reader can see how much of a keyword's
        # hit count is independently-relevant vs. everything else.
        rows = wh.query(
            "SELECT count(*) AS n, "
            "count(*) FILTER (WHERE relevant) AS relevant_n, "
            "min(substr(value,1,100)) AS example FROM artifacts "
            "WHERE value ILIKE ? OR key ILIKE ?",
            [f"%{kw}%", f"%{kw}%"],
        )
        hits.append({
            "keyword": kw, "hits": rows[0]["n"], "relevant_hits": rows[0]["relevant_n"],
            "example": rows[0]["example"] or "",
        })
    return hits
