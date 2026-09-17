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
    "key", "value", "text", "record_ts", "source_file", "label",
]


def _load_rows(wh, source_type: str) -> list[dict]:
    rows = wh.query(
        "SELECT artifact_id, app, source_type, category, confidence, key, value, "
        "record_ts, source_file FROM artifacts WHERE source_type = ? ORDER BY app, key",
        [source_type],
    )
    for r in rows:
        r["text"] = f"[{r['source_type']}] {r['key']} = {r['value']}"
        r["label"] = ""  # fill in with the true category before scoring
    return rows


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
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

        # allocate the total sample budget across source_types proportional to
        # their share of the case, so a source_type with 400k rows doesn't
        # crowd out one with 40
        all_rows = {st: _load_rows(wh, st) for st in source_types}
        total_rows = sum(len(v) for v in all_rows.values())

        summary: list[tuple[str, int, int]] = []  # source_type, written, of_total
        grand_total = 0
        for st, rows in all_rows.items():
            if args.sample:
                budget = max(1, round(args.sample * len(rows) / total_rows))
                chosen = _stratified_sample(rows, budget, rng)
            else:
                chosen = rows
            _write_csv(out_dir / f"{st}.csv", chosen)
            summary.append((st, len(chosen), len(rows)))
            grand_total += len(chosen)
    finally:
        wh.close()

    print(f"Case {args.case_id}: {total_rows:,} artifacts across {len(source_types)} source type(s)")
    print(f"Wrote {grand_total:,} rows to {out_dir}/")
    for st, written, of_total in sorted(summary, key=lambda x: -x[2]):
        note = f"{written:,} of {of_total:,}" if written != of_total else f"{written:,} (all)"
        print(f"  {st:<18} {note}")
    print()
    print("Fill in the `label` column with the true category for each row, then:")
    print(f"  uv run python scripts/score_labels.py --dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
