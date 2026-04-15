"""agent-compose: Docker Compose for multi-agent AI systems."""

from .models import (
    AgentResult,
    AgentSpec,
    AgentStatus,
    Framework,
    PipelineResult,
    PipelineSpec,
)
from .runner import Runner, run_pipeline

__version__ = "0.1.0"
__all__ = [
    "AgentResult",
    "AgentSpec",
    "AgentStatus",
    "Framework",
    "PipelineResult",
    "PipelineSpec",
    "Runner",
    "run_pipeline",
]
