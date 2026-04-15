"""Core data models for agent-compose."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Framework(Enum):
    DIRECT = "direct"
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    OPENAI_SDK = "openai_sdk"


@dataclass
class AgentSpec:
    """Parsed agent definition from YAML."""

    name: str
    model: str = "gpt-4o-mini"
    provider: str = ""
    framework: Framework = Framework.DIRECT
    system_prompt: str = ""
    prompt: str = ""
    temperature: float = 0.0
    max_tokens: int = 2000
    connects_to: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    mcp_servers: list[dict] = field(default_factory=list)
    max_iterations: int = 10
    timeout_seconds: int = 120
    retry: bool = False
    retry_delay: int = 2
    output: str = ""
    output_format: str = ""
    condition: str = ""
    # CrewAI-specific
    role: str = ""
    goal: str = ""
    backstory: str = ""

    def detect_provider(self) -> str:
        """Auto-detect provider from model name."""
        if self.provider:
            return self.provider
        m = self.model.lower()
        if any(m.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
            return "openai"
        if m.startswith("claude-"):
            return "anthropic"
        if m.startswith("gemini-"):
            return "google"
        if m.startswith("deepseek"):
            return "deepseek"
        if m.startswith("ollama/"):
            return "ollama"
        if m.startswith("openrouter/"):
            return "openrouter"
        return "openai"


@dataclass
class AgentResult:
    """Result from executing a single agent."""

    name: str
    status: AgentStatus
    framework: str = "direct"
    output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None
    trace: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class PipelineSpec:
    """Parsed pipeline from YAML."""

    name: str = "unnamed"
    description: str = ""
    agents: list[AgentSpec] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Aggregate result from the full pipeline."""

    pipeline_name: str
    description: str = ""
    agent_results: list[AgentResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.agent_results)

    @property
    def total_latency(self) -> float:
        return sum(r.latency_seconds for r in self.agent_results)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.agent_results)

    @property
    def all_succeeded(self) -> bool:
        return all(r.status == AgentStatus.COMPLETED for r in self.agent_results)

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.agent_results if r.status == AgentStatus.COMPLETED)

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_name,
            "description": self.description,
            "summary": {
                "agents": len(self.agent_results),
                "completed": self.completed_count,
                "total_tokens": self.total_tokens,
                "total_latency_s": round(self.total_latency, 2),
                "total_cost_usd": round(self.total_cost, 4),
                "success": self.all_succeeded,
            },
            "agents": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "framework": r.framework,
                    "tokens": r.total_tokens,
                    "latency_s": round(r.latency_seconds, 2),
                    "cost_usd": round(r.cost_usd, 4),
                    "error": r.error,
                    "output_preview": r.output[:200] if r.output else None,
                }
                for r in self.agent_results
            ],
        }
