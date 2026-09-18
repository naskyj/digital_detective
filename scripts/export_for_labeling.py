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

Read-only: opens the case's warehouse for reading only. Never writes to the
case directory.

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

# cache.csv carries three extra columns _FIELDS doesn't - three genuinely
# different "which file" facts a cache entry has, kept as separate columns
# rather than folded into one:
#   source_file / cache_offset  the PHYSICAL on-disk file (+ byte offset)
#                               Chromium actually stored the bytes in, e.g.
#                               "Cache/Cache_Data/data_3" @ 737280 - meta and
#                               body streams for the SAME entry can resolve
#                               to different physical files (cache.py's
#                               `_location`), so this is per-ROW, not per-entry
#   filename                    a human-readable name guessed from the URL's
#                               last path segment (cache.py's
#                               `url_filename_hint`, the same one the UI's
#                               "Open" download uses) - what the resource
#                               would have been called, unrelated to where
#                               Chromium physically stored it
#   url                         the request URL, same on every row of one
#                               entry's block, not just the cache.url row -
#                               usable for correlation without a lookup
_CACHE_FIELDS = [
    "artifact_id", "app", "source_type", "category", "confidence",
    "key", "value", "text", "record_ts", "source_file", "cache_offset",
    "filename", "url", "deleted", "label",
]


def _seq_from_locator(locator: str) -> int | None:
    """The LevelDB sequence number embedded in a locator ending
    "...\\x1fseq=<N>" - localstorage.py, sessionstorage.py and indexeddb.py
    all use this exact convention (independently, but identically) to keep
    every historical/tombstoned version of a repeatedly-written key as its
    own artifact instead of colliding on one artifact_id (see the indexeddb
    collision bug fixed earlier this session). It's the ONLY recency signal
    available for these three source_types: none of them set `record_ts` -
    LevelDB itself carries no wall-clock timestamp per write, only this
    monotonic per-store counter. None when the locator doesn't use this
    convention (cache doesn't - see _raw_cache_key/_cache_offset instead)
    or the reader didn't expose a sequence number for this record.

    emit_value()/emit_parsed_value() append their own "#<leaf_path>" suffix
    AFTER this (locator=f"{locator}#{leaf_path}", base.py) for every row a
    JSON value flattens into - e.g. "...seq=1233#$" for a value whose path
    is the root ("$", see jsonflat.py) - so that suffix has to be stripped
    BEFORE looking for the number, the same way _raw_cache_key strips it,
    or every such row silently fails int() and falls back to None here."""
    if "#" in locator:
        locator = locator.split("#", 1)[0]
    marker = "\x1fseq="
    if marker not in locator:
        return None
    try:
        return int(locator.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _load_rows(wh, source_type: str) -> list[dict]:
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
        seq = _seq_from_locator(r["source_locator"])
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


def _raw_cache_key(locator: str) -> str:
    """Recover the original Chromium cache key from one of the locator shapes
    cache.py produces (`key`, `key @file:offset`, `key @file:offset::body`,
    or `...::body#leaf.path`) so every row belonging to the same cache entry -
    its url/status/content-type/timestamps AND its (separately-located) body -
    groups back together. Meta and body streams can live at different
    physical offsets for the same key (see cache.py's `_location`), so
    grouping by locator prefix alone doesn't work; this strips exactly the
    suffixes cache.py adds, in the order it adds them."""
    s = locator
    if "#" in s:
        s = s.split("#", 1)[0]
    if s.endswith("::body"):
        s = s[: -len("::body")]
    if " @" in s:
        s = s.split(" @", 1)[0]
    return s


def _cache_offset(locator: str) -> str:
    """The byte offset backing THIS row's own stream (meta or body), i.e.
    the number after the physical filename in ` @file:offset` - empty when
    cache.py couldn't resolve a physical location for this stream (never
    fatal there, so not an error here either). Per-row, not per-entry: a
    row's own locator, not the group's - see the `source_file` comment in
    `_load_cache_rows`."""
    if " @" not in locator:
        return ""
    after = locator.split(" @", 1)[1]  # "file:offset[::body[#leaf.path]]"
    file_and_offset = after.split("::", 1)[0]
    return file_and_offset.rsplit(":", 1)[-1] if ":" in file_and_offset else ""


# Row order within one surviving cache entry: identifying context first
# (url, status, content-type, timestamps), then its body flattened key by
# key - so one JSON response reads as one contiguous block instead of its
# leaves being scattered across the file next to unrelated entries that
# happen to share a leaf name (the export's normal `ORDER BY app, key` groups
# by leaf key across ALL entries, not by entry).
_CACHE_FIELD_ORDER = {
    "cache.url": 0, "cache.http_status": 1, "cache.content_type": 2,
    "cache.request_time": 3, "cache.response_time": 4,
}


def _load_cache_rows(wh) -> list[dict]:
    """Cache entries, JSON API responses only - every other content type
    (JS, HTML, XML, plain text, images, fonts, ...) dropped entirely, since
    those are the overwhelming majority of a Chromium cache and pure noise
    for hand-labelling investigation-relevant data. Rows for a surviving
    entry are kept adjacent and ordered (see _CACHE_FIELD_ORDER) instead of
    interleaved with every other entry's rows."""
    rows = wh.query(
        "SELECT artifact_id, app, source_type, category, confidence, key, value, "
        "record_ts, source_file, deleted, source_locator FROM artifacts WHERE source_type = 'cache'"
    )
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["app"], _raw_cache_key(r["source_locator"])), []).append(r)

    kept: list[tuple[str, str, list[dict]]] = []
    for (app, raw_key), group in groups.items():
        content_type = next((g["value"] for g in group if g["key"] == "cache.content_type"), "")
        if "json" not in (content_type or "").lower():
            continue
        url = next((g["value"] for g in group if g["key"] == "cache.url"), raw_key)
        kept.append((app, url, group))

    from dd_lab.parsers.cache import url_filename_hint

    out: list[dict] = []
    for _app, url, group in sorted(kept, key=lambda t: (t[0], t[1])):
        filename = url_filename_hint(url)
        for r in sorted(group, key=lambda g: (_CACHE_FIELD_ORDER.get(g["key"], 5), g["key"])):
            r["cache_offset"] = _cache_offset(r["source_locator"])
            del r["source_locator"]
            r["filename"] = filename  # human-readable, guessed from the URL - NOT the physical file above
            r["url"] = url  # the entry's URL, on every row - not just the cache.url row
            r["text"] = f"[{r['source_type']}] {r['key']} = {r['value']}"
            r["label"] = ""
            out.append(r)
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

        # cache is filtered (JSON API responses only - see _load_cache_rows)
        # before anything else happens to it, so "of_total" below needs the
        # PRE-filter raw count too, or a filtered source_type would silently
        # look identical to one that was never filtered at all.
        raw_counts = {
            r["source_type"]: r["n"]
            for r in wh.query("SELECT source_type, COUNT(*) AS n FROM artifacts GROUP BY source_type")
        }

        # allocate the total sample budget across source_types proportional to
        # their share of the case, so a source_type with 400k rows doesn't
        # crowd out one with 40
        all_rows = {
            st: (_load_cache_rows(wh) if st == "cache" else _load_rows(wh, st)) for st in source_types
        }
        total_rows = sum(len(v) for v in all_rows.values())

        summary: list[tuple[str, int, int, int]] = []  # source_type, written, of_filtered, of_raw
        grand_total = 0
        for st, rows in all_rows.items():
            if args.sample:
                budget = max(1, round(args.sample * len(rows) / total_rows))
                chosen = _stratified_sample(rows, budget, rng)
            else:
                chosen = rows
            _write_csv(out_dir / f"{st}.csv", chosen, _CACHE_FIELDS if st == "cache" else _FIELDS)
            summary.append((st, len(chosen), len(rows), raw_counts.get(st, len(rows))))
            grand_total += len(chosen)
    finally:
        wh.close()

    raw_total = sum(raw_counts.values())
    print(f"Case {args.case_id}: {raw_total:,} artifacts across {len(source_types)} source type(s)")
    print(f"Wrote {grand_total:,} rows to {out_dir}/")
    for st, written, of_filtered, of_raw in sorted(summary, key=lambda x: -x[3]):
        if written == of_raw:
            note = f"{written:,} (all)"
        elif written == of_filtered:
            note = f"{written:,} of {of_raw:,} (rest filtered out - not JSON, for cache)"
        else:
            note = f"{written:,} of {of_filtered:,} kept, {of_raw:,} raw"
        print(f"  {st:<18} {note}")
    print()
    print("Fill in the `label` column with the true category for each row, then:")
    print(f"  uv run python scripts/score_labels.py --dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
