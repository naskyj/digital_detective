from __future__ import annotations

from datetime import UTC, datetime

from dd_core import (
    CustodyEvent,
    CustodyLog,
    NormalizedArtifact,
    SourceType,
)
from dd_core.hashing import canonical_json, sha256_canonical
from dd_core.manifest import (
    AcquisitionHost,
    Manifest,
    verify_case,
)
from dd_core.schema import TargetApp
from dd_core.signing import generate_keypair, sign, verify

# --------------------------------------------------------------------- hashing


def test_canonical_json_is_order_independent():
    a = {"b": 1, "a": [3, 2, 1], "c": {"y": 2, "x": 1}}
    b = {"c": {"x": 1, "y": 2}, "a": [3, 2, 1], "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert sha256_canonical(a) == sha256_canonical(b)


# --------------------------------------------------------------------- signing


def test_sign_verify_roundtrip():
    priv, pub = generate_keypair()
    sig = sign(priv, b"evidence")
    assert verify(pub, b"evidence", sig)
    assert not verify(pub, b"tampered", sig)


# ------------------------------------------------------------------- artifacts


def test_artifact_autofills_hash_and_size():
    art = NormalizedArtifact(
        artifact_id="x",
        case_id="c1",
        app="slack",
        source_type=SourceType.LOCAL_STORAGE,
        source_file="slack/local_storage/leveldb/000003.log",
        source_locator="_https://app.slack.com\x00\x01token",
        key="token",
        value="xoxc-abc",
        extracted_ts=datetime.now(UTC),
    )
    assert art.size_bytes == len(b"xoxc-abc")
    assert len(art.value_sha256) == 64


def test_artifact_id_is_stable():
    a = NormalizedArtifact.make_id("c1", SourceType.CACHE, "f", "loc", "k")
    b = NormalizedArtifact.make_id("c1", SourceType.CACHE, "f", "loc", "k")
    c = NormalizedArtifact.make_id("c1", SourceType.CACHE, "f", "loc", "k2")
    assert a == b != c


# -------------------------------------------------------------------- manifest


def _fake_case(tmp_path):
    case = tmp_path / "case_001"
    evidence = case / "evidence"
    (evidence / "slack" / "local_storage").mkdir(parents=True)
    f = evidence / "slack" / "local_storage" / "000003.log"
    f.write_bytes(b"hello leveldb")

    now = datetime.now(UTC)
    priv, pub = generate_keypair()
    from dd_core.manifest import build_entry

    manifest = Manifest(
        case_id="case_001",
        collector_version="0.1.0",
        operator="det. tester",
        host=AcquisitionHost(
            hostname="EVID-PC", os="Windows", os_version="10.0.19045",
            username="suspect", timezone="UTC",
        ),
        started_ts=now,
        completed_ts=now,
        target_apps=[TargetApp(name="slack", user_data_dir=r"C:\Users\suspect\AppData\Roaming\Slack")],
        entries=[build_entry(evidence, f, SourceType.LOCAL_STORAGE, now)],
        public_key_pem=pub.decode(),
    )
    manifest.write(case, priv)
    return case, priv, pub


def test_verify_case_clean(tmp_path):
    case, _, _ = _fake_case(tmp_path)
    result = verify_case(case)
    assert result.ok
    assert result.signature_valid
    assert result.checked == 1
    assert result.issues == []


def test_verify_case_detects_tampering(tmp_path):
    case, _, _ = _fake_case(tmp_path)
    target = case / "evidence" / "slack" / "local_storage" / "000003.log"
    target.write_bytes(b"hello leveldb TAMPERED")
    result = verify_case(case)
    assert not result.ok
    assert any(i.kind in {"hash_mismatch", "size_mismatch"} for i in result.issues)


def test_verify_case_detects_extra_file(tmp_path):
    case, _, _ = _fake_case(tmp_path)
    (case / "evidence" / "slack" / "planted.txt").write_bytes(b"planted")
    result = verify_case(case)
    assert not result.ok
    assert any(i.kind == "unexpected" and i.path.endswith("planted.txt") for i in result.issues)


# --------------------------------------------------------------------- custody


def test_custody_chain_valid(tmp_path):
    log = CustodyLog(tmp_path)
    log.append(CustodyEvent.ACQUISITION_START, "collector", "start")
    log.append(CustodyEvent.ARTIFACT_ACQUIRED, "collector", "slack cache", {"files": 12})
    log.append(CustodyEvent.ACQUISITION_COMPLETE, "collector", "done")
    ok, bad = log.verify_chain()
    assert ok and bad is None
    assert [r.seq for r in log.entries()] == [0, 1, 2]


def test_custody_chain_detects_edit(tmp_path):
    log = CustodyLog(tmp_path)
    log.append(CustodyEvent.ACQUISITION_START, "collector", "start")
    log.append(CustodyEvent.NOTE, "analyst", "original note")
    lines = log.path.read_text().splitlines()
    lines[1] = lines[1].replace("original note", "doctored note")
    log.path.write_text("\n".join(lines) + "\n")
    ok, bad = log.verify_chain()
    assert not ok and bad == 1
