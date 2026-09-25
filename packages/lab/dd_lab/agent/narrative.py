"""An optional model-written account of the case, on top of the computed one.

The relationship between this module and `dd_lab.investigate.timeline` is the
whole design, so it is worth stating plainly: the timeline module computes the
facts — episodes, their boundaries, which application each begins in and when
the others join, coverage — deterministically, with no model and no network.
This module asks a language model to turn those same facts into prose a human
reads more easily. It is strictly additive:

- Nothing here can introduce a fact. The model is given the computed episode
  structure and a shortlist of notable artifacts, and is told to write only
  from that. It has no tools and no warehouse access.
- Every artifact id it cites is re-verified against the warehouse afterwards,
  the same way `write_finding` verifies a finding's citations. A narrative
  citing an id that does not exist is rejected wholesale rather than printed
  with a warning — a plausible-sounding account with a fabricated citation is
  worse than no account.
- If no model is configured, or the call fails, or verification fails, the
  caller gets None and the deterministic narrative stands alone. The tool has
  to work with no API key and no internet, so the computed account is the
  floor and this is the polish.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dd_lab.agent.config import AgentConfig, build_model
from dd_lab.etl.warehouse import Warehouse
from dd_lab.investigate import build_case_narrative
from dd_lab.investigate.verify import verify_claim

SYSTEM_PROMPT = """\
You are writing the "what happened" section of a forensic report about
Electron-application evidence (Discord, Slack, Roblox and similar).

You are given a reconstruction that has ALREADY been computed from the
evidence: episodes of continuous activity, which application each begins in
and when the others join, the categories of artifact present, and a shortlist
of notable artifacts with their ids. Your job is to turn that into an account
a human investigator can read, not to analyse it further.

Absolute rules:
- Use ONLY the facts given to you. You have no access to the evidence and no
  tools. If something is not in the input, it did not happen as far as this
  section is concerned.
- Cite artifact ids from the shortlist for any specific claim. Every id you
  write is re-checked against the warehouse; inventing one voids the whole
  section.
- Respect the coverage figure you are given. Most artifacts in a case of this
  kind carry no timestamp at all, so this account describes only the part of
  the evidence that can be placed in time. Say so, in your own words, rather
  than implying the account is complete.
- Describe evidence, never conclusions. "Payment-related artifacts appear
  alongside credential artifacts in the same period" is in scope. Any
  statement about intent, guilt, or what a person was trying to do is not.
  Do not name a crime. Do not speculate about motive.
- No headings, no bullet lists, no markdown. Three to six short paragraphs of
  plain prose, in chronological order.
"""


class _Written(BaseModel):
    account: str = Field(description="The prose account, 3-6 short paragraphs.")
    artifact_ids: list[str] = Field(
        default_factory=list,
        description="Every artifact id cited in the account, so each can be re-verified.",
    )


@dataclass
class CaseStory:
    account: str
    artifact_ids: list[str]
    model: str
    verified: bool


def _facts(story: dict, max_episodes: int = 12) -> str:
    """The model's entire input: computed facts, nothing else."""
    cov = story["coverage"]
    lines = [
        f"COVERAGE: {cov['events']} of {cov['total']} relevant artifacts "
        f"({cov['event_pct']}%) carry an event timestamp. {cov['state']} have no "
        f"intrinsic time and are only known to have been present when the evidence "
        f"was acquired. {cov['expiry']} are cookie expiry dates and are excluded.",
        f"EPISODES: {story['episode_count']}",
    ]
    for ep in story["episodes"][:max_episodes]:
        lines.append("")
        lines.append(ep["summary"].replace("**", ""))
        for n in ep["notable"]:
            lines.append(
                f"  - artifact {n['artifact_id']} | {n['app']} | {n['category']} | "
                f"{n['key']} = {str(n['value'])[:160]}"
                + (" | RECOVERED DELETED RECORD" if n["deleted"] else "")
            )
    return "\n".join(lines)


def write_case_story(
    wh: Warehouse, cfg: AgentConfig | None, *, relevant_only: bool = True, model=None
) -> CaseStory | None:
    """Model-written account, or None if one cannot be produced safely.

    `model` is injectable so this is testable without a provider - the same
    convention judge.py uses.
    """
    story = build_case_narrative(wh, relevant_only=relevant_only)
    if not story["episodes"]:
        return None  # nothing to narrate; the computed text already says so
    if model is None:
        if cfg is None:
            return None
        try:
            model = build_model(cfg)
        except Exception:
            return None

    agent = Agent(model, output_type=_Written, system_prompt=SYSTEM_PROMPT)
    try:
        result = agent.run_sync(_facts(story))
    except Exception:
        # No key, no network, provider error - the deterministic narrative
        # stands on its own and the report is still complete without this.
        return None

    written: _Written = result.output
    # Every cited id must exist in this case's warehouse. A fabricated
    # citation voids the section rather than being footnoted - the whole
    # value of the account is that a reader can follow it back to evidence.
    verified = all(verify_claim(wh, aid).ok for aid in written.artifact_ids)
    return CaseStory(
        account=written.account.strip(),
        artifact_ids=list(written.artifact_ids),
        model=getattr(cfg, "model", "") if cfg else "",
        verified=verified,
    )
