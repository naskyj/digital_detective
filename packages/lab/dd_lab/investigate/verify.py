"""Self-verification — re-check a claim against the raw evidence before it is
written into a finding or a report.

An agent (or a human) proposes: "artifact X shows the suspect's Roblox id was
also used in this Discord message". Before that sentence is allowed into a
report, `verify_claim` re-reads artifact X straight from the warehouse — which
is loaded verbatim from the signed, hash-verified evidence — and confirms the
claimed text is actually there. This cannot stop a bad *inference*, but it
guarantees every citation is real: no claim survives that points at a
non-existent artifact or misquotes what it says.
"""

from __future__ import annotations

from dataclasses import dataclass

from dd_lab.etl.warehouse import Warehouse


@dataclass
class ClaimVerification:
    artifact_id: str
    ok: bool
    reason: str
    app: str | None = None
    source_file: str | None = None
    source_locator: str | None = None
    value: str | None = None
    value_sha256: str | None = None


def verify_claim(wh: Warehouse, artifact_id: str, claim_substring: str | None = None) -> ClaimVerification:
    """Look up `artifact_id` and, if `claim_substring` is given, confirm it
    actually occurs (case-insensitively) in the stored value."""
    rows = wh.query(
        "SELECT artifact_id, app, source_file, source_locator, value, value_sha256 "
        "FROM artifacts WHERE artifact_id = ?",
        [artifact_id],
    )
    if not rows:
        return ClaimVerification(artifact_id, False, "artifact_id not found in warehouse — claim rejected")
    row = rows[0]
    if claim_substring and claim_substring.lower() not in (row["value"] or "").lower():
        return ClaimVerification(
            artifact_id, False,
            "claimed text not found in the artifact's stored value — claim rejected",
            app=row["app"], source_file=row["source_file"], source_locator=row["source_locator"],
            value=row["value"], value_sha256=row["value_sha256"],
        )
    return ClaimVerification(
        artifact_id, True, "verified against the warehouse",
        app=row["app"], source_file=row["source_file"], source_locator=row["source_locator"],
        value=row["value"], value_sha256=row["value_sha256"],
    )
