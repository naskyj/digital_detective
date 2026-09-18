"""dd_lab.imaging.acquire.source_app_prefix / dd_lab.api.main._source_label_for_app
- the mechanism that recovers which registered evidence source produced a
given `app` string, without reverse-parsing a label back out of an
already-sanitised name (see the real bug this design avoids: an earlier
"<label> / <app>" combined-then-sanitised approach lost the separator
entirely - both halves are sanitised SEPARATELY then joined instead)."""

from __future__ import annotations

from dd_core.evidence import safe_name
from dd_lab.api.main import _source_label_for_app
from dd_lab.case import EvidenceSourceMeta
from dd_lab.imaging.acquire import source_app_prefix


def _source(label: str) -> EvidenceSourceMeta:
    return EvidenceSourceMeta(label=label, kind="folder", path="/x")


def test_source_app_prefix_is_safe_name_plus_underscore():
    assert source_app_prefix("Maria's Laptop") == safe_name("Maria's Laptop") + "_"


def test_source_label_for_app_matches_the_right_source():
    sources = [_source("Suspect Laptop"), _source("Victim Phone")]
    app1 = source_app_prefix("Suspect Laptop") + safe_name("discord")
    app2 = source_app_prefix("Victim Phone") + safe_name("discord")
    assert _source_label_for_app(app1, sources) == "Suspect Laptop"
    assert _source_label_for_app(app2, sources) == "Victim Phone"


def test_source_label_for_app_none_when_unlabelled():
    """Every case before this feature, and any app whose source left
    `source_label` blank, has no " / "-free-text label to find - the UI
    must render these exactly as before (flat, no source-grouping level),
    not guess a source that doesn't exist."""
    assert _source_label_for_app("discord", []) is None
    assert _source_label_for_app("discord", [_source("Suspect Laptop")]) is None


def test_source_label_for_app_does_not_false_match_a_substring():
    """An app name that happens to CONTAIN another source's prefix as a
    substring (not at the start) must not match it."""
    sources = [_source("A")]
    app = "not_" + source_app_prefix("A") + "discord"
    assert _source_label_for_app(app, sources) is None
