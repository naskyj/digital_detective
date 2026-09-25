"""The optional model-written account — scripted FunctionModel, no real LLM.

What matters here is that this layer can only ever ADD polish: it cannot
introduce a fact, it cannot cite an artifact that does not exist, and its
absence costs the report nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType as S
from dd_core.schema import ClassificationResult, NormalizedArtifact
from dd_lab.agent.narrative import _facts, write_case_story
from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate import build_case_narrative

_NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _scripted(account: str, artifact_ids: list[str]) -> FunctionModel:
    def fn(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={"account": account, "artifact_ids": artifact_ids},
        )])

    return FunctionModel(fn)


def _capture() -> tuple[FunctionModel, list[str]]:
    """A model that records the prompt it was given, so the test can assert
    on what the model is actually allowed to see."""
    seen: list[str] = []

    def fn(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        for m in messages:
            for part in getattr(m, "parts", []):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    seen.append(content)
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result", args={"account": "ok", "artifact_ids": []},
        )])

    return FunctionModel(fn), seen


@pytest.fixture
def wh(tmp_path):
    w = Warehouse(tmp_path / "case.duckdb")
    w.load_artifacts([
        NormalizedArtifact(
            artifact_id=f"art-{i}", case_id="c1", app=app, source_type=S.CACHE,
            source_file=f"{app}/Cache/f{i}", source_locator=f"loc{i}",
            key="cache.url", value=f"https://example.test/{i}",
            record_ts=_NOW + timedelta(minutes=m), extracted_ts=_NOW,
            classification=ClassificationResult(
                category=C.CHAT_CONTENT, confidence=0.9, classifier_version="test",
                relevant=True, relevance_score=0.9),
        )
        for i, (app, m) in enumerate([("discord", 0), ("Slack", 10), ("discord", 900)])
    ])
    yield w
    w.close()


def test_no_model_configured_returns_none_not_an_error(wh):
    """The tool has to work with no API key and no internet. The computed
    narrative is the floor; this layer is polish."""
    assert write_case_story(wh, None) is None


def test_provider_failure_degrades_silently(wh):
    def boom(messages, info):
        raise RuntimeError("no network")

    assert write_case_story(wh, None, model=FunctionModel(boom)) is None


def test_story_with_real_citations_verifies(wh):
    story = write_case_story(
        wh, None, model=_scripted("Activity begins in discord.", ["art-0", "art-1"]))

    assert story is not None
    assert story.verified is True
    assert story.account == "Activity begins in discord."


def test_a_fabricated_citation_voids_the_story(wh):
    """A plausible account with an invented citation is worse than none: the
    whole value is that a reader can follow it back to evidence."""
    story = write_case_story(
        wh, None, model=_scripted("Something happened.", ["art-0", "does-not-exist"]))

    assert story is not None
    assert story.verified is False


def test_the_model_only_ever_sees_computed_facts(wh):
    """It has no tools and no warehouse access - its entire input is the
    episode structure, so it cannot introduce a fact of its own."""
    model, seen = _capture()
    write_case_story(wh, None, model=model)
    prompt = "\n".join(seen)

    assert "COVERAGE:" in prompt
    assert "EPISODES:" in prompt
    assert "Begins in discord" in prompt
    assert "artifact art-0" in prompt


def test_facts_block_carries_coverage_and_deleted_flags(wh):
    facts = _facts(build_case_narrative(wh))

    assert "carry an event timestamp" in facts
    assert "no intrinsic time" in facts
    assert facts.count("artifact art-") >= 2


def test_a_case_with_no_events_is_not_narrated(tmp_path):
    """Nothing to narrate, and the deterministic text already says exactly
    that - so there is no prompt to send and nothing to verify."""
    w = Warehouse(tmp_path / "empty.duckdb")
    try:
        assert write_case_story(w, None, model=_scripted("invented", [])) is None
    finally:
        w.close()
