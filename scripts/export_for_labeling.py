"""Export one case's warehouse artifacts to CSV for manual hand-labelling —
the "gold evaluation set" step in the classifier training track
(docs/MODEL_TRAINING.md, M4: "sample ~1,500-2,000 artifacts, STRATIFIED so
every category and every source_type is represented ... hand-label each row
with the true category").

One CSV per source_type (cache.csv, local_storage.csv, indexeddb.csv,
memory_string.csv, ...) so a labeller can work through one evidence type at a
time. Each row's `text` column is exactly the string the trained model will
see (`[<source_type>] <key> = <value>` — see MODEL_TRAINING.md's "Categories"
section), and `category`/`confidence` show what the CURRENT rule/keyword/NER/
model ensemble already predicted, so filling in `label` with the true
category turns this directly into a scoring set for scripts/score_labels.py.

cache.csv is the one exception, a different shape entirely: one row per
physical cache entry (or one element of a JSON array-of-objects body), not
one row per classified field - a content-matching/reconciliation partner's
own unit of analysis is "this cached file", and the old per-field shape
inflated 5,946 real entries into 41,405 rows. No `category`/`confidence`/
`text` columns (they were being assigned per metadata field, meaningless
once a row is a whole entry) and never downsampled by `--sample` (full
coverage parity against a raw cache dump matters more than a stratified
subset here). See dd_lab/parsers/cache.py::iter_cache_entries_for_labeling.

Read-only: opens the case's warehouse for reading (every source_type) and
its evidence directory for reading (cache only, straight from the on-disk
cache files - see above). Never writes to the case directory.

    uv run python scripts/export_for_labeling.py --cases-root <dir> --case-id <id>
    uv run python scripts/export_for_labeling.py --cases-root <dir> --case-id <id> --sample 2000
    uv run python scripts/export_for_labeling.py --cases-root <dir> --case-id <id> --out-dir /path/to/put/csvs
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for pkg in ("core", "lab"):
    p = str(_REPO_ROOT / "packages" / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

_FIELDS = [
    "artifact_id", "app", "source_type", "category", "confidence",
    "key", "value", "text", "record_ts", "source_file", "deleted", "latest", "label",
]

# cache.csv has an entirely different shape from every other source_type's
# export: ONE ROW PER PHYSICAL CACHE ENTRY (or one row per element of a JSON
# array-of-objects body), not one row per metadata field. Found via a real
# partner's reconciliation against a raw ChromeCacheView dump: the old
# per-field export (url/http_status/content_type/request_time/response_time
# each counted as their own "artifact") inflated 5,946 real entries into
# 41,405 rows. `category`/`confidence` are dropped entirely for cache - they
# were being assigned PER METADATA FIELD (e.g. a bare content-type string
# classified CHAT_CONTENT at 0.713), meaningless once those fields become
# columns of one entry's row instead of independent classified values.
#
# Column set is deliberately cache_files.csv's own (CacheFileEntry, minus
# the live-browser-only `openable`) plus `value` (the decoded body) and the
# columns needed to describe how `value` relates to the whole body
# (body_kind/array_index) - the partner's own framing: "build it from
# cache_files.csv... and just add the body", so the two exports never
# independently drift on what a metadata column means. See
# dd_lab/parsers/cache.py::iter_cache_entries_for_labeling/CacheLabelRow,
# the single source of truth for this row shape.
_CACHE_FIELDS = [
    "entry_id", "app", "source_type", "cache_kind", "cache_dir", "key",
    "physical_file", "offset", "url", "filename", "http_status", "content_type",
    "content_encoding", "cache_control", "server", "size_bytes",
    "request_time", "response_time", "deleted", "body_kind", "array_index", "value", "label",
]


def _load_rows(wh, source_type: str) -> list[dict]:
    from dd_lab.parsers.base import seq_from_locator

    rows = wh.query(
        "SELECT artifact_id, app, source_type, category, confidence, key, value, "
        "record_ts, source_file, deleted, source_locator FROM artifacts WHERE source_type = ? "
        "ORDER BY app, key",
        [source_type],
    )
    # "latest": the row holding the CURRENT value of its (app, key), not a
    # superseded one - the highest sequence number among every row sharing
    # that key. Historical rows are NOT dropped (deliberately - see
    # test_deleted_records.py/this session's `deleted` flag work: a prior
    # value is real evidence, not noise to discard) - they're marked, so a
    # hand-labeller or a downstream query can filter to current-only
    # without the full history ever leaving the export. A row whose locator
    # doesn't carry a sequence number at all (source_types other than
    # local_storage/session_storage/indexeddb - nothing to compare against)
    # defaults to latest=True: there's no version information to say
    # otherwise.
    best_seq: dict[tuple[str, str], int] = {}
    seqs: dict[str, int | None] = {}
    for r in rows:
        seq = seq_from_locator(r["source_locator"])
        seqs[r["artifact_id"]] = seq
        if seq is not None:
            k = (r["app"], r["key"])
            if seq > best_seq.get(k, -1):
                best_seq[k] = seq
    for r in rows:
        seq = seqs[r["artifact_id"]]
        k = (r["app"], r["key"])
        r["latest"] = seq is None or seq >= best_seq.get(k, seq)
        del r["source_locator"]
        r["text"] = f"[{r['source_type']}] {r['key']} = {r['value']}"
        r["label"] = ""  # fill in with the true category before scoring
    return rows


def _load_cache_rows(cases_root: str, case_id: str, apps: list[str]) -> list[dict]:
    """Every cache entry across every app that has cache evidence, ALL
    content types included (JSON, JS, CSS, HTML, images, fonts, wasm, ...) -
    reads fresh from the evidence directory (like list_cache_files, not the
    warehouse) because the warehouse's own JSON flattening already applies a
    different, more aggressive splitting rule (any array, not just a
    top-level array-of-objects) that can't be losslessly un-flattened back
    into "one row per entry". See
    dd_lab/parsers/cache.py::iter_cache_entries_for_labeling."""
    from dd_core.casedir import CaseLayout
    from dd_lab.parsers.base import ParseContext
    from dd_lab.parsers.cache import iter_cache_entries_for_labeling

    layout = CaseLayout(cases_root, case_id)
    out: list[dict] = []
    for app in apps:
        ctx = ParseContext(case_id=case_id, app=app, evidence_root=layout.evidence)
        for row in iter_cache_entries_for_labeling(ctx):
            d = row.model_dump(mode="json")
            d["source_type"] = "cache"
            d["label"] = ""
            out.append(d)
    return out


def _stratified_sample(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Roughly `target` rows out of `rows`, spread across categories present
    - a rare category with only a handful of rows still gets a few, instead
    of being swamped by GENERAL the way a plain random sample would."""
    if target >= len(rows):
        return rows
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"] or "GENERAL", []).append(r)
    n_cats = len(by_cat)
    floor = max(1, min(5, target // max(n_cats, 1)))
    out: list[dict] = []
    remaining_budget = target
    # smallest categories first, so the floor is honoured before the budget runs out
    for _cat, group in sorted(by_cat.items(), key=lambda kv: len(kv[1])):
        take = min(len(group), max(floor, 1), remaining_budget)
        out.extend(rng.sample(group, take))
        remaining_budget -= take
    # spend whatever's left proportionally across the larger categories
    if remaining_budget > 0:
        pool = [r for r in rows if r not in out]
        out.extend(rng.sample(pool, min(remaining_budget, len(pool))))
    return out


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] = _FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases-root", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--out-dir", default=None, help="default: model/eval/<case-id>/")
    p.add_argument(
        "--sample", type=int, default=None,
        help="stratified sample of about this many rows TOTAL, spread across source_type "
             "and category, instead of every row (recommended above a few thousand rows - "
             "see docs/MODEL_TRAINING.md M4)",
    )
    p.add_argument("--seed", type=int, default=13, help="sampling seed, for reproducible re-runs")
    args = p.parse_args(argv)

    from dd_core.casedir import CaseLayout
    from dd_lab.etl.warehouse import Warehouse

    layout = CaseLayout(args.cases_root, args.case_id)
    if not layout.warehouse.is_file():
        print(
            f"no warehouse at {layout.warehouse} - has this case been processed (\"Run analysis\")?",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "model" / "eval" / args.case_id
    rng = random.Random(args.seed)

    wh = Warehouse(layout.warehouse)
    try:
        source_types = [
            r["source_type"]
            for r in wh.query("SELECT DISTINCT source_type FROM artifacts ORDER BY source_type")
        ]
        if not source_types:
            print("this case's warehouse has no artifacts", file=sys.stderr)
            return 1

        raw_counts = {
            r["source_type"]: r["n"]
            for r in wh.query("SELECT source_type, COUNT(*) AS n FROM artifacts GROUP BY source_type")
        }

        apps = sorted(p.name for p in layout.evidence.iterdir()) if layout.evidence.is_dir() else []

        # allocate the total sample budget across source_types proportional to
        # their share of the case, so a source_type with 400k rows doesn't
        # crowd out one with 40. cache is loaded fresh from evidence, not the
        # warehouse (see _load_cache_rows) - its "share" is one row per
        # physical entry, an entirely different unit from every other
        # source_type's one-row-per-classified-field, so it's excluded from
        # that proportional split and always written in full instead (see
        # the --sample handling below).
        all_rows = {
            st: (_load_cache_rows(args.cases_root, args.case_id, apps) if st == "cache" else _load_rows(wh, st))
            for st in source_types
        }
        total_rows = sum(len(v) for st, v in all_rows.items() if st != "cache")

        summary: list[tuple[str, int, int]] = []  # source_type, written, of_total
        grand_total = 0
        for st, rows in all_rows.items():
            if args.sample and st != "cache":
                budget = max(1, round(args.sample * len(rows) / total_rows)) if total_rows else len(rows)
                chosen = _stratified_sample(rows, budget, rng)
            else:
                chosen = rows
            _write_csv(out_dir / f"{st}.csv", chosen, _CACHE_FIELDS if st == "cache" else _FIELDS)
            summary.append((st, len(chosen), len(rows)))
            grand_total += len(chosen)
    finally:
        wh.close()

    raw_total = sum(raw_counts.values())
    print(f"Case {args.case_id}: {raw_total:,} artifacts across {len(source_types)} source type(s)")
    print(f"Wrote {grand_total:,} rows to {out_dir}/")
    for st, written, of_total in sorted(summary, key=lambda x: -x[2]):
        if st == "cache":
            note = f"{written:,} cache entries/elements (always full - not a --sample of the {raw_counts.get('cache', 0):,} warehouse cache rows, a different unit; see cache.py's iter_cache_entries_for_labeling)"
        elif written == of_total:
            note = f"{written:,} (all)"
        else:
            note = f"{written:,} of {of_total:,}"
        print(f"  {st:<18} {note}")
    print()
    print("Fill in the `label` column with the true category for each row, then:")
    print(f"  uv run python scripts/score_labels.py --dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
