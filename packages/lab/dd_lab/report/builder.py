"""Generate the HTML forensic report from a case warehouse + logs.

Self-contained single file (inline CSS), suitable for disclosure. Content:
case context, evidence inventory, integrity attestation, artifact-category
breakdown, an evidence timeline, high-priority findings, keyword hits, and
chain-of-custody + agent-audit appendices.
"""

from __future__ import annotations

import html
import re
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
from dd_lab.investigate import build_case_narrative, build_correlation_report

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

/* Print / PDF. The report is a single self-contained file, so "export to
   PDF" is the browser's own print-to-PDF - no renderer dependency, nothing
   to install, and the output is paginated by the same engine that displayed
   it. These rules make that output usable rather than incidental. */
@media print{
  body{background:#fff;font-size:10.5pt}
  .wrap{max-width:none;padding:0}
  /* Keep a section's heading with its content, and never split a finding,
     a citation table row or a key/value block across a page boundary. */
  h2,h3{break-after:avoid-page;page-break-after:avoid}
  h2{break-before:page;page-break-before:always}
  h2:first-of-type{break-before:auto;page-break-before:auto}
  table{break-inside:auto;font-size:9pt}
  tr,.kv{break-inside:avoid;page-break-inside:avoid}
  /* Repeat column headers on every page a long table spans - without this a
     citation table running to page three has unlabelled columns. */
  thead{display:table-header-group}
  th{background:#f0f0f0 !important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .pill,.hi{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  /* A URL or locator that wraps is readable; one that overflows is lost. */
  td,code{word-break:break-word;overflow-wrap:anywhere}
  footer{break-before:avoid-page}
}
@page{margin:16mm 14mm 18mm}
"""


def _esc(v: object) -> str:
    return html.escape(str(v)) if v is not None else ""


class _Sections:
    """Section numbers were hardcoded strings, so a case with no keywords
    produced a report that jumped from 5 straight to 7. Numbering is now
    derived from what is actually emitted."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, title: str) -> str:
        self.n += 1
        return f"<h2>{self.n}. {_esc(title)}</h2>"


def _md_bold(text: str) -> str:
    """The narrative is generated once, in dd_lab.investigate.timeline, and
    rendered in several places. It uses `**bold**`; escape first, then turn
    only that one marker into HTML - never a general markdown pass, which
    would be an injection surface for evidence-derived text."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _esc(text))


def _rows(headers: list[str], rows: list[list], highlight: bool = False) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        cls = ' class="hi"' if highlight else ""
        body += "<tr" + cls + ">" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_report(
    cases_root: str | Path, case_id: str, *, actor: str = "dd-lab", llm_cfg=None
) -> Path:
    """`llm_cfg` is optional throughout. With no model configured the report
    is complete and self-consistent - the computed narrative is the account.
    A model, when there is one, adds a readable retelling of those same facts
    and nothing else (see dd_lab.agent.narrative)."""
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
        # Reads `artifacts`, not `v_timeline`: the view selects neither
        # source_file nor source_locator, which made the timeline the ONLY
        # report section whose rows could not be traced back to evidence.
        # Expiry rows are excluded here for the same reason the narrative
        # excludes them - a cookie's expires_utc is a future validity
        # boundary, not something that happened (see investigate/timeline.py).
        timeline = wh.query(
            "SELECT strftime(record_ts,'%Y-%m-%d %H:%M') AS ts, app, category, deleted, "
            "substr(key,1,50) AS key, substr(value,1,80) AS value, "
            "source_file, source_locator "
            "FROM artifacts WHERE relevant AND record_ts IS NOT NULL "
            "AND NOT regexp_matches(key, '\\.(expires?_utc|expiry|expires_at|expiration)$') "
            "ORDER BY record_ts LIMIT 200"
        )
        high = wh.query(
            "SELECT category, round(confidence,2) AS conf, app, substr(key,1,60) AS key, "
            "substr(value,1,120) AS value FROM v_high_priority LIMIT 100"
        )
        kw_hits = _keyword_hits(wh, meta.keywords)
        correlations = build_correlation_report(wh)
        story = build_case_narrative(wh)
        written_story = None
        if llm_cfg is not None:
            from dd_lab.agent.narrative import write_case_story

            written_story = write_case_story(wh, llm_cfg)
    finally:
        wh.close()

    parts: list[str] = [f"<style>{_CSS}</style><div class='wrap'>"]
    parts.append(f"<h1>Forensic Analysis Report</h1><p class='sub'>{_esc(meta.name)} &middot; case <code>{_esc(case_id)}</code></p>")
    sec = _Sections()

    sig = "<span class='pill ok'>verified</span>" if verification.ok else "<span class='pill bad'>FAILED</span>"
    parts.append(sec("Case & Evidence"))
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

    # Placed second, immediately after provenance: a reader should be able to
    # learn what the evidence shows before being handed 200 rows of it. The
    # prose is computed in dd_lab.investigate.timeline - no model involved, so
    # it is identical on every run and on every machine.
    parts.append(sec(f"What Happened ({story['episode_count']} episodes)"))
    cov = story["coverage"]
    parts.append(
        "<p class='sub'>Reconstructed from artifacts that carry an event timestamp, grouped "
        "into episodes of continuous activity. Cookie expiry dates are excluded: they are "
        "future validity boundaries, not events. This section is generated deterministically "
        "from the evidence and states its own coverage rather than implying it describes the "
        "whole case.</p>"
    )
    if story["episodes"]:
        parts.append(
            f"<p><strong>{cov['events']:,} of {cov['total']:,} relevant artifacts "
            f"({cov['event_pct']}%)</strong> carry an event timestamp and are described below. "
            f"{cov['state']:,} have no intrinsic time and are present as state at acquisition"
            + (f"; {cov['expiry']:,} are cookie expiry dates and are excluded from the sequence"
               if cov["expiry"] else "")
            + ".</p>"
        )
        if written_story is not None and written_story.verified:
            parts.append(
                "<h3>Summary</h3>"
                "<p class='sub'>Written by the configured language model from the computed "
                "episodes below, and from nothing else. Every artifact it cites was "
                "re-verified against the warehouse. The episode reconstruction that follows "
                "is generated deterministically and is the authoritative record.</p>"
            )
            for para in written_story.account.split("\n\n"):
                if para.strip():
                    parts.append(f"<p>{_esc(para.strip())}</p>")
            parts.append("<h3>Episodes</h3>")
        for ep in story["episodes"]:
            parts.append(f"<p>{_md_bold(ep['summary'])}</p>")
            if ep["notable"]:
                parts.append(_rows(
                    ["", "Time", "App", "Category", "Key", "Value", "File", "Locator"],
                    [["DELETED" if n["deleted"] else "", n["ts"], n["app"], n["category"],
                      n["key"], n["value"], n["source_file"], n["source_locator"]]
                     for n in ep["notable"]],
                    highlight=True,
                ))
    else:
        parts.append(f"<p>{_esc(story['narrative'])}</p>")

    # Inferred times that fall outside every recorded episode. Reported, never
    # plotted as activity: an object created in the past being referenced now
    # is a different fact from something happening then.
    outside = story.get("referenced_outside_window") or {}
    if outside.get("count"):
        parts.append(
            f"<h3>Referenced outside the recorded activity window "
            f"({outside['count']:,})</h3>"
            f"<p class='sub'>Artifacts referencing objects created between "
            f"{_esc(str(outside['earliest'])[:10])} and {_esc(str(outside['latest'])[:10])}, "
            f"outside every episode above. These are evidence that older material was present "
            f"on the device, not evidence of activity at those times.</p>"
        )
        parts.append(_rows(
            ["Object created", "App", "Derived from", "Key", "Value"],
            [[str(e["ts"])[:19], e["app"], e["time_source"], e["key"], e["value"]]
             for e in outside["examples"]],
        ))

    parts.append(sec("Artifact Categories"))
    parts.append(_rows(["Category", "Artifacts", "Mean confidence"],
                       [[c["category"], f"{c['artifacts']:,}", c["conf"]] for c in by_cat]))
    parts.append("<h3>By source</h3>")
    parts.append(_rows(["App", "Source", "Artifacts"],
                       [[c["app"], c["source_type"], f"{c['artifacts']:,}"] for c in by_src]))

    parts.append(sec(f"High-Priority Findings ({len(high)})"))
    parts.append("<p class='sub'>Credentials, payment data, location data and chat content, ranked by sensitivity and confidence.</p>")
    parts.append(_rows(["Category", "Conf", "App", "Key", "Value"],
                       [[h["category"], h["conf"], h["app"], h["key"], h["value"]] for h in high],
                       highlight=True) if high else "<p>None.</p>")

    # Superseded findings are shown apart, not mixed in and not hidden. Their
    # citations still verify - the artifact really does contain the text - but
    # the premise does not: the entity they are about is no longer one the
    # extractor recognises. On the real case 202 of 207 were "the email
    # 'wechat@3x.png' appears in N artifacts", written before the email
    # pattern was fixed and carried by every report since.
    superseded = [f for f in findings if str(f.status) == "superseded"]
    findings = [f for f in findings if str(f.status) != "superseded"]

    parts.append(sec(f"Findings ({len(findings)})"))
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

    if superseded:
        parts.append(
            f"<h3>Superseded findings ({len(superseded)})</h3>"
            "<p class='sub'>Written by an earlier analysis run and no longer supported. Their "
            "citations still verify &mdash; the cited artifact really does contain the quoted "
            "text &mdash; but the premise does not: the entity each one is about is not one the "
            "current extractor recognises. They are retained rather than deleted, because a "
            "conclusion that was drawn and later invalidated is part of the case history.</p>"
        )
        parts.append(_rows(
            ["Written", "By", "Entity", "Value", "Claim"],
            [[str(f.created_ts)[:19], f.created_by, f.entity_type or "",
              str(f.entity_value or "")[:60], f.claim[:120]] for f in superseded[:50]],
        ))
        if len(superseded) > 50:
            parts.append(f"<p class='sub'>({len(superseded) - 50} further not listed.)</p>")

    parts.append(sec(f"Cross-Source Correlations ({len(correlations)})"))
    parts.append(
        "<p class='sub'>Identifiers (ids, invite links, emails, asset ids) that appear in more "
        "than one application or storage mechanism — a link a single artifact cannot show on its "
        "own. Each row cites its source file and in-file locator; times are the record's own "
        "timestamp, oldest first, as a reconstructed sequence for that identifier. An entity is "
        "only ever found by scanning independently RELEVANT artifacts (see the Relevant column "
        "below), but once found, every occurrence of it anywhere in the case is cited as "
        "corroborating evidence, relevant or not — a plumbing/telemetry match sharing the same "
        "value is kept, not hidden, just clearly marked apart from a genuinely relevant hit. "
        "Ordered by significance &mdash; spread across applications and storage, independent "
        "corroboration, and recovered deletions &mdash; never by raw occurrence count, which "
        "was measured on real evidence not to distinguish a genuine identifier from cached "
        "boilerplate.</p>"
    )
    if correlations:
        for e in correlations:
            cross = " &middot; <strong>spans applications</strong>" if e.cross_app else ""
            deleted = (f" &middot; <strong>{e.deleted_count} in recovered deleted records</strong>"
                       if e.deleted_count else "")
            parts.append(
                f"<h3>{_esc(e.entity_type)} &mdash; <code>{_esc(e.value)}</code></h3>"
                f"<p class='sub'>significance {e.significance} &middot; "
                f"{e.occurrence_count:,} occurrence(s), {e.relevant_count:,} independently "
                f"relevant &middot; {_esc(', '.join(e.apps))}{cross}{deleted}</p>"
            )
            # Why it is here, in plain English. A bare rank tells an
            # investigator nothing about what they might have missed.
            reasons = e.reasons()
            if reasons:
                parts.append("<ul>" + "".join(f"<li>{_esc(r)}</li>" for r in reasons) + "</ul>")
            parts.append(_rows(
                ["Relevant", "Deleted", "Timestamp", "App", "Source", "File", "Locator", "Key", "Value"],
                [["✓" if c.relevant else "—", "DELETED" if c.deleted else "",
                  c.record_ts, c.app, c.source_type, c.source_file,
                  c.source_locator, c.key, c.value[:160]] for c in e.timeline()],
                highlight=True,
            ))
    else:
        parts.append("<p>No identifier was found in more than one app or storage mechanism.</p>")

    if meta.keywords:
        parts.append(sec(f"Investigation Keyword Hits ({sum(k['hits'] for k in kw_hits)})"))
        parts.append(_rows(["Keyword", "Hits", "Relevant hits", "Example"],
                           [[k["keyword"], k["hits"], k["relevant_hits"], k["example"]]
                            for k in kw_hits]))

    parts.append(sec(f"Evidence Timeline ({len(timeline)} events, oldest first)"))
    parts.append(
        "<p class='sub'>Every row cites the file and in-file locator it came from. Cookie "
        "expiry dates are excluded. See the What Happened section above for the same events "
        "grouped into episodes.</p>"
    )
    parts.append(_rows(
        ["", "Timestamp", "App", "Category", "Key", "Value", "File", "Locator"],
        [["DELETED" if t["deleted"] else "", t["ts"], t["app"], t["category"], t["key"],
          t["value"], t["source_file"], t["source_locator"]] for t in timeline],
        highlight=True,
    ) if timeline else "<p>No timestamped artifacts.</p>")

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
