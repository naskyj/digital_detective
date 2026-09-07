"""Case directory layout.

    <root>/<case_id>/
        manifest.json              signed evidence inventory (collector)
        manifest.json.sig          detached Ed25519 signature (collector)
        chain_of_custody.jsonl     hash-chained custody log
        evidence/                  acquired artifacts, immutable after collection
            <app>/cache/ ...
            <app>/local_storage/ ...
            <app>/session_storage/ ...
            <app>/indexeddb/ ...
            <app>/cookies/ ...
            <app>/memory/ ...
        raw_artifacts/             lab: normalized parser output (jsonl/parquet)
        warehouse/case.duckdb      lab: analytical warehouse
        reports/                   lab: generated reports
        logs/audit.jsonl           lab: agent tool calls + model messages
        case.json                  lab: case metadata (name, keywords, status)
"""

from __future__ import annotations

from pathlib import Path


class CaseLayout:
    def __init__(self, root: str | Path, case_id: str):
        self.case_id = case_id
        self.dir = Path(root) / case_id

    # evidence side (written by the collector, read-only for the lab) ------
    @property
    def evidence(self) -> Path:
        return self.dir / "evidence"

    @property
    def manifest(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def custody_log(self) -> Path:
        return self.dir / "chain_of_custody.jsonl"

    def app_dir(self, app: str) -> Path:
        return self.evidence / app

    # lab side -----------------------------------------------------------
    @property
    def raw_artifacts(self) -> Path:
        return self.dir / "raw_artifacts"

    @property
    def warehouse(self) -> Path:
        return self.dir / "warehouse" / "case.duckdb"

    @property
    def reports(self) -> Path:
        return self.dir / "reports"

    @property
    def audit_log(self) -> Path:
        return self.dir / "logs" / "audit.jsonl"

    @property
    def case_meta(self) -> Path:
        return self.dir / "case.json"

    def create_evidence_skeleton(self) -> None:
        self.evidence.mkdir(parents=True, exist_ok=True)

    def create_lab_skeleton(self) -> None:
        for p in (self.raw_artifacts, self.warehouse.parent, self.reports, self.audit_log.parent):
            p.mkdir(parents=True, exist_ok=True)
