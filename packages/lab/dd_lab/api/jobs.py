"""In-process background jobs with a durable record.

Long operations (acquisition, ingest, agent runs) execute on a worker thread so
the API stays responsive. Job records — including a rolling log — are kept in a
small SQLite file so progress survives a page reload and a crash leaves a trace.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    case_id    TEXT,
    status     TEXT NOT NULL,          -- queued | running | done | error
    created_ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL,
    log        TEXT NOT NULL DEFAULT '[]',
    result     TEXT,
    error      TEXT
);
"""


class JobManager:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        con = self._con()
        con.executescript(_SCHEMA)
        # Jobs run on worker threads that do not survive a process restart, so
        # anything still 'queued'/'running' at startup is orphaned. Mark it so
        # the case isn't stuck with a phantom active job (which disables "Run").
        con.execute(
            "UPDATE jobs SET status = 'error', "
            "error = 'interrupted — the app was restarted while this job was running', "
            "updated_ts = ? WHERE status IN ('queued', 'running')",
            (_now(),),
        )
        con.commit()

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.db_path, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    # ------------------------------------------------------------------ api
    def submit(
        self, kind: str, fn: Callable[[Callable[[str], None]], dict], case_id: str | None = None
    ) -> str:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._con().execute(
                "INSERT INTO jobs (id, kind, case_id, status, created_ts, updated_ts) "
                "VALUES (?, ?, ?, 'queued', ?, ?)",
                (job_id, kind, case_id, now, now),
            )
            self._con().commit()
        threading.Thread(target=self._run, args=(job_id, fn), daemon=True).start()
        return job_id

    def _run(self, job_id: str, fn: Callable[[Callable[[str], None]], dict]) -> None:
        self._set(job_id, status="running")

        def progress(msg: str) -> None:
            self._append_log(job_id, msg)

        try:
            result = fn(progress) or {}
            self._set(job_id, status="done", result=json.dumps(result, default=str))
        except Exception as e:  # noqa: BLE001
            self._append_log(job_id, f"ERROR: {e}")
            self._set(job_id, status="error", error=f"{e}\n{traceback.format_exc()[-2000:]}")

    def get(self, job_id: str) -> dict | None:
        row = self._con().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["log"] = json.loads(d["log"])
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d

    def list(self, case_id: str | None = None, limit: int = 50) -> list[dict]:
        if case_id:
            rows = self._con().execute(
                "SELECT id, kind, case_id, status, created_ts, updated_ts FROM jobs "
                "WHERE case_id = ? ORDER BY created_ts DESC LIMIT ?",
                (case_id, limit),
            ).fetchall()
        else:
            rows = self._con().execute(
                "SELECT id, kind, case_id, status, created_ts, updated_ts FROM jobs "
                "ORDER BY created_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def active_for(self, case_id: str) -> dict | None:
        row = self._con().execute(
            "SELECT * FROM jobs WHERE case_id = ? AND status IN ('queued','running') "
            "ORDER BY created_ts DESC LIMIT 1",
            (case_id,),
        ).fetchone()
        return self.get(row["id"]) if row else None

    # ------------------------------------------------------------------ internals
    def _set(self, job_id: str, **fields) -> None:
        fields["updated_ts"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._con().execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
            self._con().commit()

    def _append_log(self, job_id: str, msg: str) -> None:
        with self._lock:
            row = self._con().execute("SELECT log FROM jobs WHERE id = ?", (job_id,)).fetchone()
            entries = json.loads(row["log"]) if row else []
            entries.append({"ts": _now(), "msg": msg})
            self._con().execute(
                "UPDATE jobs SET log = ?, updated_ts = ? WHERE id = ?",
                (json.dumps(entries[-500:]), _now(), job_id),
            )
            self._con().commit()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
