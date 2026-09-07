"""Agent orchestration layer: deterministic tools + an optional LLM driver."""

from __future__ import annotations

from dd_lab.agent.audit import AuditLog
from dd_lab.agent.config import AgentConfig, build_model
from dd_lab.agent.runner import RunOutcome, run_agent, run_pipeline
from dd_lab.agent.state import PipelineState
from dd_lab.agent.tools import InvestigationDeps

__all__ = [
    "AgentConfig",
    "AuditLog",
    "InvestigationDeps",
    "PipelineState",
    "RunOutcome",
    "build_model",
    "run_agent",
    "run_pipeline",
]
