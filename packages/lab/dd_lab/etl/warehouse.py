"""Per-case analytical warehouse (DuckDB).

One file per case at `<case>/warehouse/case.duckdb`. Holds the normalized
artifact table plus a set of comprehensive views the report and the agent query.
The classifier updates the classification columns in place after load.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from dd_core.schema import NormalizedArtifact

log = logging.getLogger("dd_lab.warehouse")

# A _load_*.jsonl file younger than this is never touched by cleanup - see
# _clean_orphaned_load_files. Generous relative to how long one app's load
# actually takes (the write loop's mtime stays fresh the whole time; only
# the write-closed-but-not-yet-read_json-and-deleted tail end has no writes
# happening, and that's a DuckDB bulk read+insert, seconds even for a very
# large app - not remotely close to this threshold).
_ORPHAN_MIN_AGE_SECONDS = 300


def _clean_orphaned_load_files(warehouse_dir: Path, min_age_seconds: float = _ORPHAN_MIN_AGE_SECONDS) -> None:
    """`load_artifacts` streams to a `_load_<uuid>.jsonl` temp file and always
    deletes it in a `finally` - unless the whole process dies mid-load (killed,
    crashed, or the app was closed/restarted while a run was in progress), in
    which case that cleanup never runs and the file (which can be hundreds of
    MB — one per interrupted load) sits there forever.

    A file's age, not merely its existence, is what actually tells an orphan
    apart from one a DIFFERENT, still-running Warehouse instance for the same
    case is actively streaming into right now - a real failure this caused,
    not a hypothetical one: `run_case`'s ingest job runs in a background
    thread (JobManager.submit) while the server keeps handling other
    requests, so any concurrent request that opens its own Warehouse for the
    same case (e.g. a browser tab reading the case page while "Run analysis"
    is in progress) used to delete the ingest job's own in-progress temp
    file out from under it - confirmed on a real case: `read_json()` then
    failed with "No files found that match the pattern .../_load_<uuid>.jsonl"
    because the file it had just written was gone. An actively-written file's
    mtime is refreshed by every line appended to it (see load_artifacts), so
    it's never more than moments old for as long as the write loop runs; only
    a file whose writer genuinely died stays stale. `min_age_seconds`
    overridable for tests - production callers should never need to."""
    now = time.time()
    for f in warehouse_dir.glob("_load_*.jsonl"):
        try:
            if now - f.stat().st_mtime < min_age_seconds:
                continue  # too recently written to safely assume orphaned
            f.unlink()
            log.info("removed orphaned load file from an interrupted run: %s", f.name)
        except OSError:
            pass  # best-effort - never let cleanup abort opening the warehouse

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id       VARCHAR PRIMARY KEY,
    case_id           VARCHAR NOT NULL,
    app               VARCHAR NOT NULL,
    source_type       VARCHAR NOT NULL,
    source_file       VARCHAR NOT NULL,
    source_locator    VARCHAR NOT NULL,
    key               VARCHAR NOT NULL,
    value             VARCHAR NOT NULL,
    value_type        VARCHAR NOT NULL,
    record_ts         TIMESTAMP,
    extracted_ts      TIMESTAMP NOT NULL,
    size_bytes        BIGINT NOT NULL,
    value_sha256      VARCHAR NOT NULL,
    deleted           BOOLEAN DEFAULT FALSE,  -- recovered from a tombstoned/overwritten record
    -- classification (filled by the NLP ensemble)
    category          VARCHAR,
    confidence        DOUBLE,
    relevant          BOOLEAN,        -- "does this matter to the investigation" - a real field of its
    relevance_score   DOUBLE,         -- own, not every consumer re-deriving category != 'GENERAL'
    regex_vote        VARCHAR,
    ner_vote          VARCHAR,
    keyword_vote      VARCHAR,
    model_vote        VARCHAR,
    entropy_vote      VARCHAR,
    relevance_model_score DOUBLE,  -- the trained relevant/not_relevant model's own raw
                                    -- P(relevant), NULL when no relevance model is configured
    evidence_spans    VARCHAR,     -- JSON array
    classifier_version VARCHAR,
    classified_ts     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id       VARCHAR PRIMARY KEY,
    started_ts   TIMESTAMP NOT NULL,
    finished_ts  TIMESTAMP,
    apps         VARCHAR,
    artifact_count BIGINT,
    notes        VARCHAR
);
"""

VIEWS = """
CREATE OR REPLACE VIEW v_by_category AS
    SELECT category,
           count(*)                          AS artifacts,
           count(DISTINCT app)               AS apps,
           avg(confidence)                   AS mean_confidence,
           min(record_ts)                    AS earliest,
           max(record_ts)                    AS latest
    FROM artifacts
    WHERE category IS NOT NULL
    GROUP BY category
    ORDER BY artifacts DESC;

CREATE OR REPLACE VIEW v_timeline AS
    SELECT record_ts, app, source_type, category, relevant, key, value, artifact_id
    FROM artifacts
    WHERE record_ts IS NOT NULL
    ORDER BY record_ts;

CREATE OR REPLACE VIEW v_high_priority AS
    -- Was a hardcoded category whitelist, predating relevant/relevance_score
    -- existing as real fields - silently ignored them even after they
    -- shipped. Now keys off relevance directly (the actual "does this
    -- matter" signal), category kept only to order ties.
    -- relevance_model_score breaks ties before confidence does. The composed
    -- score saturates: `rule_evidence` is capped at 1.0 and the highest
    -- weighted voter reaches the cap alone, so noisy-OR returns exactly 1.0
    -- for ANY model probability. Measured on the annotated corpus, 2,559
    -- artifacts share the score 1.0 and only 48% of them are relevant, while
    -- the model separates that block almost perfectly (median probability
    -- 1.000 vs 0.000). Ordering by the composed score alone scores average
    -- precision 0.638; adding this tie-break takes it to 0.930, against 0.941
    -- for the model score alone. It changes no relevance decision - only the
    -- order within an existing tie.
    SELECT artifact_id, app, source_type, category, confidence, relevance_score, key, value, record_ts
    FROM artifacts
    WHERE relevant
    ORDER BY relevance_score DESC,
             relevance_model_score DESC NULLS LAST,
             confidence DESC;

CREATE OR REPLACE VIEW v_by_relevance AS
    SELECT relevant,
           count(*)                          AS artifacts,
           avg(relevance_score)              AS mean_relevance_score,
           avg(relevance_model_score)         AS mean_relevance_model_score
    FROM artifacts
    WHERE relevant IS NOT NULL
    GROUP BY relevant
    ORDER BY relevant DESC;

CREATE OR REPLACE VIEW v_app_summary AS
    SELECT app,
           count(*)                                        AS artifacts,
           count(DISTINCT source_type)                     AS source_types,
           count(*) FILTER (WHERE category IS NOT NULL)     AS classified,
           min(record_ts)                                  AS earliest,
           max(record_ts)                                  AS latest
    FROM artifacts
    GROUP BY app
    ORDER BY artifacts DESC;

CREATE OR REPLACE VIEW v_source_breakdown AS
    SELECT app, source_type, count(*) AS artifacts
    FROM artifacts
    GROUP BY app, source_type
    ORDER BY app, artifacts DESC;
"""

_CORE_COLS = (
    "artifact_id", "case_id", "app", "source_type", "source_file", "source_locator",
    "key", "value", "value_type", "record_ts", "extracted_ts", "size_bytes", "value_sha256",
    "deleted",
)
_CLASS_COLS = (
    "category", "confidence", "relevant", "relevance_score",
    "regex_vote", "ner_vote", "keyword_vote", "model_vote", "entropy_vote",
    "relevance_model_score",
    "evidence_spans", "classifier_version", "classified_ts",
)
_INSERT_COLS = _CORE_COLS + _CLASS_COLS

# column -> DuckDB type for the NDJSON bulk reader
_JSON_COLUMNS = (
    ("artifact_id", "VARCHAR"), ("case_id", "VARCHAR"), ("app", "VARCHAR"),
    ("source_type", "VARCHAR"), ("source_file", "VARCHAR"), ("source_locator", "VARCHAR"),
    ("key", "VARCHAR"), ("value", "VARCHAR"), ("value_type", "VARCHAR"),
    ("record_ts", "TIMESTAMP"), ("extracted_ts", "TIMESTAMP"),
    ("size_bytes", "BIGINT"), ("value_sha256", "VARCHAR"), ("deleted", "BOOLEAN"),
    ("category", "VARCHAR"), ("confidence", "DOUBLE"),
    ("relevant", "BOOLEAN"), ("relevance_score", "DOUBLE"),
    ("regex_vote", "VARCHAR"),
    ("ner_vote", "VARCHAR"), ("keyword_vote", "VARCHAR"), ("model_vote", "VARCHAR"),
    ("entropy_vote", "VARCHAR"),
    ("relevance_model_score", "DOUBLE"),
    ("evidence_spans", "VARCHAR"), ("classifier_version", "VARCHAR"), ("classified_ts", "TIMESTAMP"),
)


#: Columns added to `artifacts` after its first release, applied with
#: `ADD COLUMN IF NOT EXISTS` so an already-processed case's warehouse
#: (predating the column) keeps working without a full re-ingest - DuckDB's
#: `CREATE TABLE IF NOT EXISTS` above does NOT retroactively alter an
#: existing table, so a new column needs its own explicit migration here or
#: every real case processed before this line shipped would start throwing
#: "column not found" on any query that names it.
_MIGRATIONS = (
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS deleted BOOLEAN DEFAULT FALSE",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS entropy_vote VARCHAR",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS relevant BOOLEAN",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS relevance_score DOUBLE",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS relevance_model_score DOUBLE",
    # Examiner review of the artifacts the classifier did NOT flag. An
    # approval overwrites `relevant`, so the classifier's own decision is
    # preserved in `relevant_auto` first: the system discards nothing (R3),
    # and every accuracy measurement must still be able to score the
    # classifier rather than the classifier-plus-examiner.
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS relevant_auto BOOLEAN",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS review_decision VARCHAR",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS reviewed_by VARCHAR",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS reviewed_ts TIMESTAMP",
)


class Warehouse:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _clean_orphaned_load_files(self.path.parent)
        self.con = duckdb.connect(str(self.path))
        self.con.execute(SCHEMA)
        for stmt in _MIGRATIONS:
            self.con.execute(stmt)

    # ------------------------------------------------------------------ load

    def reset_app(self, app: str) -> None:
        """Drop any previously-loaded artifacts for an app (idempotent re-ingest)."""
        self.con.execute("DELETE FROM artifacts WHERE app = ?", (app,))

    def load_artifacts(self, artifacts: Iterable[NormalizedArtifact]) -> int:
        """Bulk load via NDJSON + DuckDB's native JSON reader.

        Row-at-a-time ``executemany`` into a columnar store is ~100x too slow at
        100k+ rows; streaming to a temp NDJSON file and letting DuckDB scan it is
        the fast, low-dependency path. Idempotent on artifact_id.
        """
        import json
        import uuid

        tmp = self.path.parent / f"_load_{uuid.uuid4().hex}.jsonl"
        written = 0
        with open(tmp, "w", encoding="utf-8") as fh:
            for a in artifacts:
                fh.write(json.dumps(self._record(a), ensure_ascii=False) + "\n")
                written += 1
        if written == 0:
            tmp.unlink(missing_ok=True)
            return 0
        try:
            col_types = ", ".join(f"'{c}': '{t}'" for c, t in _JSON_COLUMNS)
            self.con.execute(
                f"INSERT INTO artifacts ({', '.join(_INSERT_COLS)}) "
                f"SELECT {', '.join(_INSERT_COLS)} FROM ("
                f"  SELECT *, row_number() OVER (PARTITION BY artifact_id) AS _rn "
                f"  FROM read_json(?, format='newline_delimited', columns={{{col_types}}})"
                f") WHERE _rn = 1 "
                f"ON CONFLICT (artifact_id) DO NOTHING",
                [str(tmp)],
            )
        finally:
            tmp.unlink(missing_ok=True)
        log.info("loaded %d artifact rows into %s", written, self.path.name)
        return written

    @staticmethod
    def _record(a: NormalizedArtifact) -> dict:
        c = a.classification
        return {
            "artifact_id": a.artifact_id, "case_id": a.case_id, "app": a.app,
            "source_type": str(a.source_type), "source_file": a.source_file,
            "source_locator": a.source_locator, "key": a.key, "value": a.value,
            "value_type": str(a.value_type),
            "record_ts": a.record_ts.isoformat() if a.record_ts else None,
            "extracted_ts": a.extracted_ts.isoformat(),
            "size_bytes": a.size_bytes, "value_sha256": a.value_sha256, "deleted": a.deleted,
            "category": str(c.category) if c else None,
            "confidence": c.confidence if c else None,
            "relevant": c.relevant if c else None,
            "relevance_score": c.relevance_score if c else None,
            "regex_vote": _s(c.regex_vote) if c else None,
            "ner_vote": _s(c.ner_vote) if c else None,
            "keyword_vote": _s(c.keyword_vote) if c else None,
            "model_vote": _s(c.model_vote) if c else None,
            "entropy_vote": _s(c.entropy_vote) if c else None,
            "relevance_model_score": c.relevance_model_score if c else None,
            "evidence_spans": _json(c.evidence_spans) if c else None,
            "classifier_version": c.classifier_version if c else None,
            "classified_ts": c.classified_ts.isoformat() if (c and c.classified_ts) else None,
        }

    def record_run(
        self, run_id: str, started_ts, finished_ts, apps: list[str], artifact_count: int, notes: str = ""
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO ingest_runs VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, started_ts, finished_ts, ",".join(apps), artifact_count, notes),
        )

    # --------------------------------------------------------------- views/query

    def build_views(self) -> None:
        self.con.execute(VIEWS)

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        cur = self.con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def count(self) -> int:
        return self.con.execute("SELECT count(*) FROM artifacts").fetchone()[0]

    def iter_for_classification(
        self,
        batch_size: int = 20000,
        *,
        only_unscored: bool = False,
        exclude_source_types: tuple[str, ...] | None = None,
        source_types: tuple[str, ...] | None = None,
    ) -> Iterator[list[dict]]:
        """Yield batches of {artifact_id, key, value, value_type, app,
        source_type, deleted, relevance_model_score} for re-classification.
        app/source_type/deleted are needed by Classifier.classify()'s
        routing/gating (see classifier_backend_spec.md's implementation) -
        without them, re-classifying an already-ingested warehouse would
        skip every source-type-aware rule that the ingest-time
        classify_stream() path gets for free from the NormalizedArtifact
        it's already holding. relevance_model_score is the previously-stored
        model output, for classify_batch(reuse_model_scores=True) - lets a
        re-classify that only changes the COMBINATION logic skip re-running
        the (very expensive) relevance model entirely.

        `only_unscored=True` yields just the rows that have no stored
        relevance_model_score yet - for finishing an interrupted model run
        without redoing the rows it already got through (on the real case
        that was 116,725 rows instead of 675,463).

        `exclude_source_types` skips whole source types. Relevance-model
        inference is far too slow on CPU to run over every artifact of a
        big case (memory_string alone was ~547k of 675k on the real case,
        i.e. days of compute), so this scopes an expensive model pass to
        the sources worth paying for while the excluded ones keep
        rules-only relevance. It does NOT leave them unclassified - every
        artifact still gets a category/relevant/relevance_score from the
        ingest-time or rules-only pass.

        `source_types` is the inverse: process ONLY these. Combined with
        `only_unscored=True` this is how a long model pass over one huge
        source type (e.g. memory_string) runs in the background without
        touching anything already scored - every other row is simply never
        selected, so nothing else is rewritten.

        Pagination is KEYSET (artifact_id > last seen), not LIMIT/OFFSET,
        specifically because of `only_unscored`: the caller writes scores
        into the very column being filtered on, so the result set shrinks
        as it iterates and OFFSET would silently skip rows. Taking "the
        first N unscored" each time instead would be worse still - rows the
        empty-body gate resolves never GET a model score, so they'd stay
        NULL and be re-yielded forever. Keying on artifact_id always moves
        forward regardless of what was or wasn't written.
        """
        clauses, params = [], []
        if only_unscored:
            clauses.append("relevance_model_score IS NULL")
        if exclude_source_types:
            placeholders = ", ".join("?" * len(exclude_source_types))
            clauses.append(f"source_type NOT IN ({placeholders})")
            params.extend(exclude_source_types)
        if source_types:
            placeholders = ", ".join("?" * len(source_types))
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        where = "".join(f"{c} AND " for c in clauses)
        last_id = ""
        while True:
            rows = self.query(
                "SELECT artifact_id, key, value, value_type, app, source_type, deleted, "
                f"relevance_model_score FROM artifacts WHERE {where} artifact_id > ? "
                f"ORDER BY artifact_id LIMIT {batch_size}",
                [*params, last_id],
            )
            if not rows:
                return
            yield rows
            last_id = rows[-1]["artifact_id"]

    def apply_classifications(self, updates: Iterable[tuple[str, dict]]) -> int:
        """Bulk-apply classification results in a single UPDATE ... FROM join per batch.

        Point UPDATEs on a columnar store are pathologically slow; staging + a
        set-based join is orders of magnitude faster.
        """
        self.con.execute(
            "CREATE OR REPLACE TEMP TABLE _cls ("
            "artifact_id VARCHAR, category VARCHAR, confidence DOUBLE, "
            "relevant BOOLEAN, relevance_score DOUBLE, regex_vote VARCHAR, "
            "ner_vote VARCHAR, keyword_vote VARCHAR, model_vote VARCHAR, entropy_vote VARCHAR, "
            "relevance_model_score DOUBLE, "
            "evidence_spans VARCHAR, classifier_version VARCHAR, classified_ts TIMESTAMP)"
        )
        total = 0
        for batch in _batched(updates, 20000):
            self.con.execute("DELETE FROM _cls")
            self.con.executemany(
                "INSERT INTO _cls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        aid, c["category"], c["confidence"],
                        c.get("relevant"), c.get("relevance_score"),
                        c.get("regex_vote"), c.get("ner_vote"),
                        c.get("keyword_vote"), c.get("model_vote"), c.get("entropy_vote"),
                        c.get("relevance_model_score"),
                        c.get("evidence_spans"), c.get("classifier_version"), c.get("classified_ts"),
                    )
                    for aid, c in batch
                ],
            )
            self.con.execute(
                "UPDATE artifacts a SET "
                "category = c.category, confidence = c.confidence, "
                # `relevant_auto` always takes the classifier's new decision,
                # but `relevant` keeps an examiner's review if one exists:
                # re-classifying must not silently revoke a human judgement
                # about evidence. Without this, every /reclassify would undo
                # the whole review queue with no trace.
                "relevant_auto = c.relevant, "
                "relevant = CASE WHEN a.review_decision IS NOT NULL "
                "THEN a.relevant ELSE c.relevant END, "
                "relevance_score = c.relevance_score, "
                "regex_vote = c.regex_vote, "
                "ner_vote = c.ner_vote, keyword_vote = c.keyword_vote, model_vote = c.model_vote, "
                "entropy_vote = c.entropy_vote, relevance_model_score = c.relevance_model_score, "
                "evidence_spans = c.evidence_spans, classifier_version = c.classifier_version, "
                "classified_ts = c.classified_ts "
                "FROM _cls c WHERE a.artifact_id = c.artifact_id"
            )
            total += self.con.execute("SELECT count(*) FROM _cls").fetchone()[0]
        self.con.execute("DROP TABLE IF EXISTS _cls")
        return total

    # ------------------------------------------------------------- review
    #
    # The classifier's recall is 0.958 on the annotated corpus, so roughly one
    # relevant artifact in 24 is never surfaced. Those misses are not sitting
    # in an uncertain middle where a threshold could find them: the composed
    # score is near-bimodal and 45 of the 54 are at exactly 0.0, among 551,809
    # others. Reviewing "everything below the threshold" is 553,727 artifacts
    # - the original problem restated.
    #
    # The model probability does rank them, though, for the same reason it
    # ranks the flagged block (see the v_high_priority view). Ordering the
    # UNFLAGGED artifacts by it and reviewing a fixed budget off the top
    # recovers, measured on the annotated corpus:
    #
    #     50 reviewed  ->  21 of 54 misses (39%), 42% of what is read
    #    100 reviewed  ->  22 of 54 misses (41%)
    #  1,000 reviewed  ->  31 of 54 misses (57%)
    #
    # so fifty artifacts - minutes of work - takes recall from 0.958 to 0.974.
    # The budget is the examiner's to choose; the ordering is what makes any
    # budget worth spending.

    REVIEW_DECISIONS = ("approved", "rejected")

    def review_queue(self, limit: int = 50, offset: int = 0,
                     include_reviewed: bool = False) -> list[dict]:
        """Artifacts the classifier did NOT flag, best candidates first.

        Ordered by the model's own probability rather than the composed
        relevance score, which is 0.0 for almost all of this population and
        therefore cannot order it at all.
        """
        where = "WHERE (relevant IS NOT TRUE)"
        if not include_reviewed:
            where += " AND review_decision IS NULL"
        return self.query(
            "SELECT artifact_id, app, source_type, category, confidence, "
            "relevant, relevance_score, relevance_model_score, "
            "regex_vote, keyword_vote, ner_vote, entropy_vote, evidence_spans, "
            "key, substr(value, 1, 500) AS value, record_ts, deleted, "
            "source_file, source_locator, review_decision, reviewed_by, reviewed_ts "
            f"FROM artifacts {where} "
            "ORDER BY relevance_model_score DESC NULLS LAST, artifact_id "
            "LIMIT ? OFFSET ?",
            [int(limit), int(offset)],
        )

    def review_counts(self) -> dict:
        """Queue depth and review progress, for the UI header."""
        row = self.query(
            "SELECT "
            "  count(*) FILTER (WHERE relevant IS NOT TRUE "
            "                     AND review_decision IS NULL) AS pending, "
            "  count(*) FILTER (WHERE review_decision = 'approved') AS approved, "
            "  count(*) FILTER (WHERE review_decision = 'rejected') AS rejected, "
            "  count(*) FILTER (WHERE relevant) AS relevant "
            "FROM artifacts"
        )
        return row[0] if row else {"pending": 0, "approved": 0, "rejected": 0,
                                   "relevant": 0}

    def record_review(self, artifact_id: str, decision: str,
                      reviewed_by: str = "") -> dict:
        """Record an examiner's decision on one unflagged artifact.

        An approval flips `relevant`, so the artifact reaches the report,
        correlation and timeline like any other relevant artifact. The
        classifier's own decision is copied to `relevant_auto` first and never
        overwritten afterwards, so the override is always visible as an
        override and the classifier remains independently scoreable (R3).

        Returns the updated row. Raises KeyError for an unknown artifact and
        ValueError for an unknown decision.
        """
        if decision not in self.REVIEW_DECISIONS:
            raise ValueError(
                f"decision must be one of {self.REVIEW_DECISIONS}, got {decision!r}")
        existing = self.query(
            "SELECT relevant, relevant_auto, review_decision FROM artifacts "
            "WHERE artifact_id = ?", [artifact_id])
        if not existing:
            raise KeyError(artifact_id)
        self.con.execute(
            "UPDATE artifacts SET "
            # COALESCE so re-reviewing never overwrites the ORIGINAL automatic
            # decision with a previous override.
            "  relevant_auto = COALESCE(relevant_auto, relevant), "
            "  relevant = ?, review_decision = ?, reviewed_by = ?, reviewed_ts = ? "
            "WHERE artifact_id = ?",
            [decision == "approved", decision, reviewed_by,
             datetime.now(UTC), artifact_id],
        )
        return self.query(
            "SELECT artifact_id, app, source_type, category, key, "
            "relevant, relevant_auto, relevance_score, relevance_model_score, "
            "review_decision, reviewed_by, reviewed_ts, source_file, source_locator "
            "FROM artifacts WHERE artifact_id = ?", [artifact_id])[0]

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextmanager
def open_warehouse(db_path: str | Path) -> Iterator[Warehouse]:
    wh = Warehouse(db_path)
    try:
        yield wh
    finally:
        wh.close()


def _s(v) -> str | None:
    return str(v) if v is not None else None


def _json(spans) -> str:
    import json

    return json.dumps(list(spans or []), ensure_ascii=False)


def _dedup(artifacts: Iterable[NormalizedArtifact], seen: set[str]) -> Iterator[NormalizedArtifact]:
    for a in artifacts:
        if a.artifact_id not in seen:
            seen.add(a.artifact_id)
            yield a


def _batched(it: Iterable, n: int) -> Iterator[list]:
    batch: list = []
    for item in it:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
