"""Load, validate, and parse agent-compose YAML specs."""

from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

from ..models import AgentSpec, Framework, PipelineSpec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_spec(path: str | Path = "agent-compose.yaml") -> PipelineSpec:
    """Load and validate a pipeline spec from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spec not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML format in {path}")
    if "agents" not in raw:
        raise ValueError(f"Spec must contain 'agents' in {path}")
    if not isinstance(raw["agents"], dict):
        raise ValueError(f"'agents' must be a mapping in {path}")

    settings = raw.get("settings", {})
    default_model = settings.get("default_model", "gpt-4o-mini")
    default_framework = settings.get("default_framework", "direct")

    # Resolve env vars
    env: dict[str, str] = {}
    for key, val in raw.get("env", {}).items():
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            env[key] = os.environ.get(val[2:-1], "")
        else:
            env[key] = str(val)

    # Parse agents
    agents: list[AgentSpec] = []
    for name, spec in raw["agents"].items():
        if spec is None:
            spec = {}
        agents.append(_parse_agent(name, spec, default_model, default_framework))

    # Validate references
    agent_names = {a.name for a in agents}
    for agent in agents:
        for target in agent.connects_to:
            if target not in agent_names:
                raise ValueError(
                    f"Agent '{agent.name}' connects_to '{target}' which doesn't exist"
                )

    # Check for cycles
    _check_cycles(agents)

    return PipelineSpec(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        agents=agents,
        settings=settings,
        env=env,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


_FRAMEWORK_MAP = {
    "direct": Framework.DIRECT,
    "langgraph": Framework.LANGGRAPH,
    "crewai": Framework.CREWAI,
    "openai_sdk": Framework.OPENAI_SDK,
}


def _parse_agent(
    name: str, spec: dict, default_model: str, default_framework: str
) -> AgentSpec:
    fw_str = spec.get("framework", default_framework).lower()
    framework = _FRAMEWORK_MAP.get(fw_str, Framework.DIRECT)

    return AgentSpec(
        name=name,
        model=spec.get("model", default_model),
        provider=spec.get("provider", ""),
        framework=framework,
        system_prompt=spec.get("system_prompt", ""),
        prompt=spec.get("prompt", ""),
        temperature=spec.get("temperature", 0.0),
        max_tokens=spec.get("max_tokens", 2000),
        connects_to=spec.get("connects_to", []),
        tools=spec.get("tools", []),
        mcp_servers=spec.get("mcp_servers", []),
        max_iterations=spec.get("max_iterations", 10),
        timeout_seconds=spec.get("timeout_seconds", 120),
        retry=spec.get("retry", False),
        retry_delay=spec.get("retry_delay", 2),
        output=spec.get("output", ""),
        output_format=spec.get("output_format", ""),
        condition=spec.get("condition", ""),
        role=spec.get("role", ""),
        goal=spec.get("goal", ""),
        backstory=spec.get("backstory", ""),
    )


def _check_cycles(agents: list[AgentSpec]) -> None:
    """Detect cycles in the agent dependency graph via DFS."""
    graph: dict[str, set[str]] = {a.name: set(a.connects_to) for a in agents}
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def _dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if _dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.discard(node)
        return False

    for name in graph:
        if name not in visited:
            if _dfs(name):
                raise ValueError(f"Circular dependency detected involving '{name}'")


# ---------------------------------------------------------------------------
# DAG utilities
# ---------------------------------------------------------------------------


def build_execution_order(spec: PipelineSpec) -> list[list[str]]:
    """Return execution levels (each level can run in parallel).

    Uses reverse-edge topological sort: agents at level 0 have no
    upstream producers; level 1 agents depend only on level-0 agents, etc.
    """
    # Build reverse adjacency: agent → set of agents that produce input for it
    # connects_to means "output flows to", so we invert to get "depends on"
    incoming: dict[str, set[str]] = defaultdict(set)
    all_names: set[str] = set()
    for agent in spec.agents:
        all_names.add(agent.name)
        for target in agent.connects_to:
            incoming[target].add(agent.name)

    # Kahn's algorithm
    in_degree = {name: len(incoming.get(name, set())) for name in all_names}
    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    levels: list[list[str]] = []

    while queue:
        level = list(queue)
        levels.append(level)
        next_queue: deque[str] = deque()
        for node in level:
            for agent in spec.agents:
                if agent.name == node:
                    for target in agent.connects_to:
                        in_degree[target] -= 1
                        if in_degree[target] == 0:
                            next_queue.append(target)
        queue = next_queue

    return levels


def resolve_template(
    template: str,
    user_input: str,
    outputs: dict[str, str],
    env: dict[str, str],
) -> str:
    """Replace {{input}}, {{agent.output}}, {{env.VAR}}, {{file:path}}."""
    if not template:
        return user_input

    result = template
    result = result.replace("{{input}}", user_input)

    for agent_name, output in outputs.items():
        result = result.replace(f"{{{{{agent_name}.output}}}}", output)

    for key, val in env.items():
        result = result.replace(f"{{{{env.{key}}}}}", val)

    for match in re.finditer(r"\{\{file:(.+?)\}\}", result):
        fpath = match.group(1)
        try:
            content = Path(fpath).read_text()
        except FileNotFoundError:
            content = f"[file not found: {fpath}]"
        result = result.replace(match.group(0), content)

    return result
