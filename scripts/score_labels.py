"""Score hand-labelled export_for_labeling.py CSVs against the classifier's
own predictions: overall accuracy, per-category precision/recall/F1,
macro-F1, and a confusion summary. Only rows with a non-empty `label`
column count — leave the rest blank until you get to them and re-run.

    uv run python scripts/score_labels.py --dir model/eval/<case-id>/
    uv run python scripts/score_labels.py --dir model/eval/<case-id>/ --out metrics.json

Stdlib only, no dependency on the classifier itself - this scores whatever
`category` the export already captured against your `label`, so it works
the same whether that came from the rule ensemble or a trained model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "core"))


def _known_categories() -> set[str]:
    try:
        from dd_core.enums import ArtifactCategory

        return {c.value for c in ArtifactCategory}
    except Exception:  # noqa: BLE001 - purely for the typo warning below, never fatal
        return set()


def _load_labeled_rows(dir_: Path) -> list[dict]:
    rows = []
    for f in sorted(dir_.glob("*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r.get("label") or "").strip():
                    rows.append(r)
    return rows


def _score(rows: list[dict], categories: list[str]):
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for r in rows:
        true = r["label"].strip().upper()
        pred = (r.get("category") or "GENERAL").strip().upper()
        confusion[true][pred] += 1
        if true == pred:
            correct += 1
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    n = len(rows)
    accuracy = correct / n if n else 0.0
    per_cat = {}
    f1s = []
    for cat in categories:
        p = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) else 0.0
        rec = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        support = tp[cat] + fn[cat]
        per_cat[cat] = {
            "precision": round(p, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": support,
        }
        if support:
            f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return accuracy, per_cat, macro_f1, confusion


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="directory of CSVs from export_for_labeling.py")
    p.add_argument("--out", default=None, help="optional: also write the metrics as JSON here")
    args = p.parse_args(argv)

    dir_ = Path(args.dir)
    rows = _load_labeled_rows(dir_)
    if not rows:
        print(
            f"no labelled rows found under {dir_} - the `label` column is empty everywhere so far",
            file=sys.stderr,
        )
        return 1

    known = _known_categories()
    seen_labels = {r["label"].strip().upper() for r in rows}
    typos = seen_labels - known if known else set()
    if typos:
        print(
            f"warning: label value(s) not among the {len(known)} known categories - typo? {sorted(typos)}\n",
            file=sys.stderr,
        )

    categories = sorted(seen_labels | {(r.get("category") or "GENERAL").strip().upper() for r in rows})
    accuracy, per_cat, macro_f1, confusion = _score(rows, categories)

    print(f"{len(rows):,} labelled row(s) scored")
    print(f"overall accuracy: {accuracy:.1%}")
    print(f"macro-F1:         {macro_f1:.3f}\n")
    print(f"{'category':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for cat, m in sorted(per_cat.items(), key=lambda kv: -kv[1]["support"]):
        if m["support"] == 0:
            continue
        print(f"{cat:<16}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}")

    print("\nconfusion (true label -> what the classifier predicted instead, mismatches only):")
    any_wrong = False
    for true, preds in sorted(confusion.items()):
        wrong = {c: n for c, n in preds.items() if c != true}
        if wrong:
            any_wrong = True
            wrong_str = ", ".join(f"{c}:{n}" for c, n in sorted(wrong.items(), key=lambda kv: -kv[1]))
            print(f"  {true:<16} -> {wrong_str}")
    if not any_wrong:
        print("  (none - every labelled row matched the classifier's prediction)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n": len(rows), "accuracy": accuracy, "macro_f1": macro_f1, "per_category": per_cat}, indent=2
        ))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
