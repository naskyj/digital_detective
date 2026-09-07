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
from dd_core.manifest import Manifest, verify_case
from dd_lab.agent.audit import AuditLog
from dd_lab.case import CaseMeta
from dd_lab.etl.warehouse import Warehouse

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
.ok{background:#e6f5e6;color:#1a6b1a}.bad{background:#fbe4e4;color:#9b1a1a}
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
    wh = Warehouse(layout.warehouse)

    try:
        wh.build_views()
        total = wh.count()
        by_cat = wh.query("SELECT category, artifacts, round(mean_confidence,2) AS conf FROM v_by_category")
        by_src = wh.query("SELECT app, source_type, artifacts FROM v_source_breakdown")
        timeline = wh.query(
            "SELECT strftime(record_ts,'%Y-%m-%d %H:%M') AS ts, app, category, "
            "substr(key,1,50) AS key, substr(value,1,80) AS value "
            "FROM v_timeline WHERE category != 'GENERAL' LIMIT 120"
        )
        high = wh.query(
            "SELECT category, round(confidence,2) AS conf, app, substr(key,1,60) AS key, "
            "substr(value,1,120) AS value FROM v_high_priority LIMIT 100"
        )
        kw_hits = _keyword_hits(wh, meta.keywords)
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

    if meta.keywords:
        parts.append(f"<h2>4. Investigation Keyword Hits ({sum(k['hits'] for k in kw_hits)})</h2>")
        parts.append(_rows(["Keyword", "Hits", "Example"],
                           [[k["keyword"], k["hits"], k["example"]] for k in kw_hits]))

    parts.append(f"<h2>5. Evidence Timeline ({len(timeline)} of the earliest non-generic events)</h2>")
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
    return out


def _keyword_hits(wh: Warehouse, keywords: list[str]) -> list[dict]:
    hits = []
    for kw in keywords:
        rows = wh.query(
            "SELECT count(*) AS n, min(substr(value,1,100)) AS example FROM artifacts "
            "WHERE value ILIKE ? OR key ILIKE ?",
            [f"%{kw}%", f"%{kw}%"],
        )
        hits.append({"keyword": kw, "hits": rows[0]["n"], "example": rows[0]["example"] or ""})
    return hits
