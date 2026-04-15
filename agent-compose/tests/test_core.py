"""Comprehensive tests for agent-compose."""

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from agent_compose.loaders.spec import (
    build_execution_order,
    load_spec,
    resolve_template,
)
from agent_compose.models import (
    AgentResult,
    AgentSpec,
    AgentStatus,
    Framework,
    PipelineResult,
    PipelineSpec,
)


# ───────────────────────────────────────────────────────────
# Spec Loading
# ───────────────────────────────────────────────────────────


class TestSpecLoading:
    """Test YAML spec parsing and validation."""

    def _write_spec(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "agent-compose.yaml"
        p.write_text(yaml.dump(data))
        return p

    def test_load_minimal_spec(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "name": "test",
            "agents": {
                "hello": {"model": "gpt-4o-mini"},
            },
        })
        spec = load_spec(p)
        assert spec.name == "test"
        assert len(spec.agents) == 1
        assert spec.agents[0].name == "hello"
        assert spec.agents[0].model == "gpt-4o-mini"

    def test_load_multi_agent_spec(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "name": "pipeline",
            "agents": {
                "researcher": {
                    "model": "gpt-4o",
                    "framework": "langgraph",
                    "connects_to": ["writer"],
                },
                "writer": {
                    "model": "gpt-4o-mini",
                    "output": "report.md",
                },
            },
        })
        spec = load_spec(p)
        assert len(spec.agents) == 2
        assert spec.agents[0].connects_to == ["writer"]
        assert spec.agents[0].framework == Framework.LANGGRAPH
        assert spec.agents[1].output == "report.md"

    def test_crewai_fields_parsed(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "agents": {
                "analyst": {
                    "framework": "crewai",
                    "role": "Senior Analyst",
                    "goal": "Find trends",
                    "backstory": "20 years in finance",
                    "model": "gpt-4o",
                },
            },
        })
        spec = load_spec(p)
        agent = spec.agents[0]
        assert agent.framework == Framework.CREWAI
        assert agent.role == "Senior Analyst"
        assert agent.goal == "Find trends"
        assert agent.backstory == "20 years in finance"

    def test_default_model_applied(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "settings": {"default_model": "claude-sonnet-4-20250514"},
            "agents": {
                "a": {},
            },
        })
        spec = load_spec(p)
        assert spec.agents[0].model == "claude-sonnet-4-20250514"

    def test_missing_agents_raises(self, tmp_path):
        p = self._write_spec(tmp_path, {"name": "bad"})
        with pytest.raises(ValueError, match="agents"):
            load_spec(p)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_spec("/nonexistent/path.yaml")

    def test_invalid_connects_to_raises(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "agents": {
                "a": {"connects_to": ["nonexistent"]},
            },
        })
        with pytest.raises(ValueError, match="nonexistent"):
            load_spec(p)

    def test_circular_dependency_raises(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "agents": {
                "a": {"connects_to": ["b"]},
                "b": {"connects_to": ["a"]},
            },
        })
        with pytest.raises(ValueError, match="[Cc]ircular"):
            load_spec(p)

    def test_self_loop_raises(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "agents": {
                "a": {"connects_to": ["a"]},
            },
        })
        with pytest.raises(ValueError, match="[Cc]ircular"):
            load_spec(p)

    def test_three_node_cycle_raises(self, tmp_path):
        p = self._write_spec(tmp_path, {
            "agents": {
                "a": {"connects_to": ["b"]},
                "b": {"connects_to": ["c"]},
                "c": {"connects_to": ["a"]},
            },
        })
        with pytest.raises(ValueError, match="[Cc]ircular"):
            load_spec(p)


# ───────────────────────────────────────────────────────────
# DAG / Execution Order
# ───────────────────────────────────────────────────────────


class TestExecutionOrder:
    """Test topological sort and parallel level detection."""

    def _make_spec(self, agents_dict: dict) -> PipelineSpec:
        agents = []
        for name, cfg in agents_dict.items():
            agents.append(AgentSpec(
                name=name,
                connects_to=cfg.get("connects_to", []),
            ))
        return PipelineSpec(agents=agents)

    def test_linear_pipeline(self):
        spec = self._make_spec({
            "a": {"connects_to": ["b"]},
            "b": {"connects_to": ["c"]},
            "c": {},
        })
        levels = build_execution_order(spec)
        assert levels == [["a"], ["b"], ["c"]]

    def test_parallel_agents(self):
        spec = self._make_spec({
            "a": {},
            "b": {},
            "c": {},
        })
        levels = build_execution_order(spec)
        assert len(levels) == 1
        assert set(levels[0]) == {"a", "b", "c"}

    def test_diamond_topology(self):
        spec = self._make_spec({
            "a": {"connects_to": ["b", "c"]},
            "b": {"connects_to": ["d"]},
            "c": {"connects_to": ["d"]},
            "d": {},
        })
        levels = build_execution_order(spec)
        assert levels[0] == ["a"]
        assert set(levels[1]) == {"b", "c"}
        assert levels[2] == ["d"]

    def test_fan_out(self):
        spec = self._make_spec({
            "root": {"connects_to": ["b", "c", "d"]},
            "b": {},
            "c": {},
            "d": {},
        })
        levels = build_execution_order(spec)
        assert levels[0] == ["root"]
        assert set(levels[1]) == {"b", "c", "d"}

    def test_single_agent(self):
        spec = self._make_spec({"solo": {}})
        levels = build_execution_order(spec)
        assert levels == [["solo"]]


# ───────────────────────────────────────────────────────────
# Template Resolution
# ───────────────────────────────────────────────────────────


class TestTemplateResolution:
    def test_input_variable(self):
        result = resolve_template("Query: {{input}}", "hello world", {}, {})
        assert result == "Query: hello world"

    def test_agent_output_variable(self):
        result = resolve_template(
            "Analysis: {{researcher.output}}",
            "",
            {"researcher": "some findings"},
            {},
        )
        assert result == "Analysis: some findings"

    def test_env_variable(self):
        result = resolve_template(
            "Key: {{env.API_KEY}}",
            "",
            {},
            {"API_KEY": "sk-123"},
        )
        assert result == "Key: sk-123"

    def test_file_variable(self, tmp_path):
        f = tmp_path / "context.txt"
        f.write_text("file content here")
        result = resolve_template(
            f"Data: {{{{file:{f}}}}}",
            "",
            {},
            {},
        )
        assert "file content here" in result

    def test_missing_file(self):
        result = resolve_template(
            "Data: {{file:/nonexistent/file.txt}}",
            "",
            {},
            {},
        )
        assert "file not found" in result.lower()

    def test_multiple_variables(self):
        result = resolve_template(
            "Input: {{input}}, Research: {{r.output}}, Key: {{env.K}}",
            "hello",
            {"r": "findings"},
            {"K": "val"},
        )
        assert "hello" in result
        assert "findings" in result
        assert "val" in result

    def test_empty_template_returns_input(self):
        result = resolve_template("", "fallback input", {}, {})
        assert result == "fallback input"


# ───────────────────────────────────────────────────────────
# Model Classes
# ───────────────────────────────────────────────────────────


class TestModels:
    def test_agent_spec_detect_provider_openai(self):
        spec = AgentSpec(name="t", model="gpt-4o")
        assert spec.detect_provider() == "openai"

    def test_agent_spec_detect_provider_anthropic(self):
        spec = AgentSpec(name="t", model="claude-sonnet-4-20250514")
        assert spec.detect_provider() == "anthropic"

    def test_agent_spec_detect_provider_google(self):
        spec = AgentSpec(name="t", model="gemini-2.5-pro")
        assert spec.detect_provider() == "google"

    def test_agent_spec_detect_provider_ollama(self):
        spec = AgentSpec(name="t", model="ollama/llama3.2")
        assert spec.detect_provider() == "ollama"

    def test_agent_spec_explicit_provider(self):
        spec = AgentSpec(name="t", model="custom-model", provider="custom")
        assert spec.detect_provider() == "custom"

    def test_pipeline_result_metrics(self):
        result = PipelineResult(
            pipeline_name="test",
            agent_results=[
                AgentResult(
                    name="a",
                    status=AgentStatus.COMPLETED,
                    input_tokens=100,
                    output_tokens=200,
                    latency_seconds=1.5,
                    cost_usd=0.01,
                ),
                AgentResult(
                    name="b",
                    status=AgentStatus.COMPLETED,
                    input_tokens=300,
                    output_tokens=400,
                    latency_seconds=2.5,
                    cost_usd=0.02,
                ),
            ],
        )
        assert result.total_tokens == 1000
        assert result.total_latency == 4.0
        assert result.total_cost == 0.03
        assert result.all_succeeded is True
        assert result.completed_count == 2

    def test_pipeline_result_with_failure(self):
        result = PipelineResult(
            pipeline_name="test",
            agent_results=[
                AgentResult(name="a", status=AgentStatus.COMPLETED),
                AgentResult(name="b", status=AgentStatus.FAILED, error="boom"),
            ],
        )
        assert result.all_succeeded is False
        assert result.completed_count == 1

    def test_pipeline_result_to_dict(self):
        result = PipelineResult(
            pipeline_name="test",
            agent_results=[
                AgentResult(
                    name="a",
                    status=AgentStatus.COMPLETED,
                    framework="direct",
                    input_tokens=100,
                    output_tokens=200,
                ),
            ],
        )
        d = result.to_dict()
        assert d["pipeline"] == "test"
        assert d["summary"]["agents"] == 1
        assert d["summary"]["completed"] == 1
        assert d["summary"]["success"] is True
        # Must be JSON serializable
        json.dumps(d)

    def test_agent_result_total_tokens(self):
        r = AgentResult(
            name="x",
            status=AgentStatus.COMPLETED,
            input_tokens=500,
            output_tokens=300,
        )
        assert r.total_tokens == 800
