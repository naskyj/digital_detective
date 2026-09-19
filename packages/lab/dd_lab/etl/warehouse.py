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
    SELECT artifact_id, app, source_type, category, confidence, key, value, record_ts
    FROM artifacts
    WHERE category IN ('CREDENTIAL','PAYMENT_INFO','CHAT_CONTENT','LOCATION_INFO')
    ORDER BY
        array_position(['CREDENTIAL','PAYMENT_INFO','LOCATION_INFO','CHAT_CONTENT'], category),
        confidence DESC;

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

    def iter_for_classification(self, batch_size: int = 20000) -> Iterator[list[dict]]:
        """Yield batches of {artifact_id, key, value, value_type, app,
        source_type, deleted} for re-classification. app/source_type/deleted
        are needed by Classifier.classify()'s routing/gating (see
        classifier_backend_spec.md's implementation) - without them,
        re-classifying an already-ingested warehouse would skip every
        source-type-aware rule that the ingest-time classify_stream() path
        gets for free from the NormalizedArtifact it's already holding."""
        offset = 0
        while True:
            rows = self.query(
                "SELECT artifact_id, key, value, value_type, app, source_type, deleted FROM artifacts "
                f"ORDER BY artifact_id LIMIT {batch_size} OFFSET {offset}"
            )
            if not rows:
                return
            yield rows
            offset += batch_size

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
            "evidence_spans VARCHAR, classifier_version VARCHAR, classified_ts TIMESTAMP)"
        )
        total = 0
        for batch in _batched(updates, 20000):
            self.con.execute("DELETE FROM _cls")
            self.con.executemany(
                "INSERT INTO _cls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        aid, c["category"], c["confidence"],
                        c.get("relevant"), c.get("relevance_score"),
                        c.get("regex_vote"), c.get("ner_vote"),
                        c.get("keyword_vote"), c.get("model_vote"), c.get("entropy_vote"),
                        c.get("evidence_spans"), c.get("classifier_version"), c.get("classified_ts"),
                    )
                    for aid, c in batch
                ],
            )
            self.con.execute(
                "UPDATE artifacts a SET "
                "category = c.category, confidence = c.confidence, "
                "relevant = c.relevant, relevance_score = c.relevance_score, "
                "regex_vote = c.regex_vote, "
                "ner_vote = c.ner_vote, keyword_vote = c.keyword_vote, model_vote = c.model_vote, "
                "entropy_vote = c.entropy_vote, "
                "evidence_spans = c.evidence_spans, classifier_version = c.classifier_version, "
                "classified_ts = c.classified_ts "
                "FROM _cls c WHERE a.artifact_id = c.artifact_id"
            )
            total += self.con.execute("SELECT count(*) FROM _cls").fetchone()[0]
        self.con.execute("DROP TABLE IF EXISTS _cls")
        return total

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
