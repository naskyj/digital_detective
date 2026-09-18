"""/api/cases/{id}/artifacts and .../export.csv's derived columns
(_enrich_artifact_rows: url/cache_offset/filename for cache, `latest` for
local_storage/session_storage/indexeddb) - synthetic data, no real Electron
profile needed (unlike test_api.py, which is real-evidence-gated for
everything in that file)."""

from __future__ import annotations

import csv
import importlib
import io
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DD_CONFIG_DIR", str(tmp_path / "cfg"))
    from dd_lab.api import settings as settings_mod

    importlib.reload(settings_mod)
    from dd_lab.api import main as main_mod

    importlib.reload(main_mod)
    main_mod._settings.cases_root = str(tmp_path / "cases")
    main_mod._settings.save()

    from dd_core.casedir import CaseLayout
    from dd_core.enums import SourceType, ValueType
    from dd_core.schema import NormalizedArtifact
    from dd_lab.case import CaseMeta
    from dd_lab.etl.warehouse import Warehouse

    case_id = "case_enrich"
    layout = CaseLayout(tmp_path / "cases", case_id)
    layout.create_lab_skeleton()
    CaseMeta(case_id=case_id, name="enrichment test").save(layout)

    now = datetime.now(UTC)
    rows = [
        # a cache entry with a resolved physical location - url/cache_offset/filename all derivable
        NormalizedArtifact(
            artifact_id="c1", case_id=case_id, app="app", source_type=SourceType.CACHE,
            source_file="app/Cache/Cache_Data/data_3",
            source_locator="1/0/https://x.test/a/b.json @data_3:500",
            key="cache.url", value="https://x.test/a/b.json", value_type=ValueType.URL, extracted_ts=now,
        ),
        # three historical writes of the SAME Local Storage key (increasing seq)
        NormalizedArtifact(
            artifact_id="ls1", case_id=case_id, app="app", source_type=SourceType.LOCAL_STORAGE,
            source_file="app/Local Storage/leveldb/000001.log",
            source_locator="origin\x1fk\x1fseq=1#$", key="origin :: k", value="old",
            value_type=ValueType.STRING, extracted_ts=now,
        ),
        NormalizedArtifact(
            artifact_id="ls2", case_id=case_id, app="app", source_type=SourceType.LOCAL_STORAGE,
            source_file="app/Local Storage/leveldb/000002.ldb",
            source_locator="origin\x1fk\x1fseq=9#$", key="origin :: k", value="new",
            value_type=ValueType.STRING, extracted_ts=now,
        ),
        NormalizedArtifact(
            artifact_id="ls3", case_id=case_id, app="app", source_type=SourceType.LOCAL_STORAGE,
            source_file="app/Local Storage/leveldb/000001.log",
            source_locator="origin\x1fk\x1fseq=4#$", key="origin :: k", value="mid",
            value_type=ValueType.STRING, extracted_ts=now,
        ),
    ]
    wh = Warehouse(layout.warehouse)
    try:
        wh.load_artifacts(rows)
        wh.build_views()
    finally:
        wh.close()
    return TestClient(main_mod.app), case_id


def test_cache_row_gets_url_offset_and_filename(client):
    c, case_id = client
    r = c.get(f"/api/cases/{case_id}/artifacts", params={"app": "app", "source_type": "cache"})
    assert r.status_code == 200
    row = r.json()["rows"][0]
    assert row["url"] == "https://x.test/a/b.json"
    assert row["cache_offset"] == 500
    assert row["filename"] == "b.json"
    assert row["latest"] is None  # not a seq-versioned source_type


def test_local_storage_latest_flag_only_the_highest_seq(client):
    c, case_id = client
    r = c.get(f"/api/cases/{case_id}/artifacts", params={"app": "app", "source_type": "local_storage"})
    assert r.status_code == 200
    by_id = {row["artifact_id"]: row for row in r.json()["rows"]}
    assert by_id["ls1"]["latest"] is False
    assert by_id["ls2"]["latest"] is True  # highest seq (9)
    assert by_id["ls3"]["latest"] is False
    assert all(row["url"] is None and row["cache_offset"] is None for row in by_id.values())


def test_latest_flag_correct_even_when_current_version_is_off_page(client):
    """The `latest` flag is computed across EVERY row for this (app,
    source_type), not just the page being rendered - the current version of
    a key might land on a different page than a historical one."""
    c, case_id = client
    # limit=1 puts only one of the three rows on this "page"
    r = c.get(
        f"/api/cases/{case_id}/artifacts",
        params={"app": "app", "source_type": "local_storage", "limit": 1, "offset": 0},
    )
    rows = r.json()["rows"]
    assert len(rows) == 1
    # whichever row this is, its `latest` value must match its true
    # standing among ALL three - never computed as if it were the only one
    assert rows[0]["latest"] == (rows[0]["artifact_id"] == "ls2")


def test_export_csv_includes_the_derived_columns(client):
    c, case_id = client
    r = c.get(f"/api/cases/{case_id}/artifacts/export.csv", params={"app": "app", "source_type": "cache"})
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows[0]["url"] == "https://x.test/a/b.json"
    assert rows[0]["cache_offset"] == "500"
    assert rows[0]["filename"] == "b.json"


def test_single_artifact_detail_has_the_same_derived_columns(client):
    c, case_id = client
    r = c.get(f"/api/cases/{case_id}/artifacts/c1")
    assert r.status_code == 200
    row = r.json()
    assert row["artifact_id"] == "c1"
    assert row["url"] == "https://x.test/a/b.json"
    assert row["cache_offset"] == 500
    assert row["filename"] == "b.json"


def test_single_artifact_detail_404s_cleanly_for_unknown_id(client):
    c, case_id = client
    r = c.get(f"/api/cases/{case_id}/artifacts/does-not-exist")
    assert r.status_code == 404


def test_artifact_id_route_does_not_swallow_export_csv(client):
    """Route-ordering regression guard: /artifacts/{artifact_id} is
    registered after /artifacts/export.csv specifically so a request to the
    literal "export.csv" path isn't matched as artifact_id="export.csv"."""
    c, case_id = client
    r = c.get(f"/api/cases/{case_id}/artifacts/export.csv", params={"app": "app"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
