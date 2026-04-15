"""Execution engines for each supported framework."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..models import AgentResult, AgentSpec, AgentStatus

logger = logging.getLogger("agent-compose")


# ---------------------------------------------------------------------------
# Cost estimation (per million tokens, March 2026 pricing)
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_M, output_per_M)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3-mini": (1.10, 4.40),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}

_DEFAULT_PRICING = (3.00, 15.00)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts."""
    inp_rate, out_rate = _DEFAULT_PRICING
    for prefix, rates in _PRICING.items():
        if model.startswith(prefix):
            inp_rate, out_rate = rates
            break
    return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Base engine
# ---------------------------------------------------------------------------


class BaseEngine(ABC):
    """Abstract base class for framework-specific agent engines."""

    @abstractmethod
    async def execute(self, spec: AgentSpec, prompt: str) -> AgentResult:
        """Execute the agent and return a result."""
        ...


# ---------------------------------------------------------------------------
# Direct engine — raw LLM API calls
# ---------------------------------------------------------------------------


class DirectEngine(BaseEngine):
    """Execute agents as direct LLM API calls (no framework)."""

    async def execute(self, spec: AgentSpec, prompt: str) -> AgentResult:
        provider = spec.detect_provider()
        start = time.monotonic()

        try:
            if provider in ("openai", "deepseek", "openrouter"):
                content, in_tok, out_tok = await self._call_openai(spec, prompt)
            elif provider == "anthropic":
                content, in_tok, out_tok = await self._call_anthropic(spec, prompt)
            else:
                content, in_tok, out_tok = await self._call_openai(spec, prompt)

            latency = time.monotonic() - start
            return AgentResult(
                name=spec.name,
                status=AgentStatus.COMPLETED,
                framework="direct",
                output=content,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_seconds=latency,
                cost_usd=estimate_cost(spec.model, in_tok, out_tok),
            )

        except Exception as e:
            latency = time.monotonic() - start
            if spec.retry:
                await asyncio.sleep(spec.retry_delay)
                try:
                    if provider in ("openai", "deepseek", "openrouter"):
                        content, in_tok, out_tok = await self._call_openai(spec, prompt)
                    elif provider == "anthropic":
                        content, in_tok, out_tok = await self._call_anthropic(spec, prompt)
                    else:
                        content, in_tok, out_tok = await self._call_openai(spec, prompt)

                    latency = time.monotonic() - start
                    return AgentResult(
                        name=spec.name,
                        status=AgentStatus.COMPLETED,
                        framework="direct",
                        output=content,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        latency_seconds=latency,
                        cost_usd=estimate_cost(spec.model, in_tok, out_tok),
                    )
                except Exception as retry_err:
                    return AgentResult(
                        name=spec.name,
                        status=AgentStatus.FAILED,
                        framework="direct",
                        latency_seconds=time.monotonic() - start,
                        error=f"Retry failed: {retry_err}",
                    )

            return AgentResult(
                name=spec.name,
                status=AgentStatus.FAILED,
                framework="direct",
                latency_seconds=latency,
                error=str(e),
            )

    async def _call_openai(
        self, spec: AgentSpec, prompt: str
    ) -> tuple[str, int, int]:
        import openai
        import os

        provider = spec.detect_provider()
        base_url = None
        api_key = None

        if provider == "deepseek":
            base_url = "https://api.deepseek.com"
            api_key = os.getenv("DEEPSEEK_API_KEY")
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.getenv("OPENROUTER_API_KEY")

        logger.info(f"DEBUG: Provider={provider}, BaseURL={base_url}")
        client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        messages: list[dict] = []
        if spec.system_prompt:
            messages.append({"role": "system", "content": spec.system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await client.chat.completions.create(
            model=spec.model,
            messages=messages,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    async def _call_anthropic(
        self, spec: AgentSpec, prompt: str
    ) -> tuple[str, int, int]:
        import anthropic

        client = anthropic.AsyncAnthropic()
        kwargs: dict = {
            "model": spec.model,
            "max_tokens": spec.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if spec.system_prompt:
            kwargs["system"] = spec.system_prompt

        resp = await client.messages.create(**kwargs)
        content = "".join(
            b.text for b in resp.content if hasattr(b, "text")
        )
        return content, resp.usage.input_tokens, resp.usage.output_tokens


# ---------------------------------------------------------------------------
# Stub engines for LangGraph, CrewAI, OpenAI SDK
# (full implementations in v0.2)
# ---------------------------------------------------------------------------


class LangGraphEngine(BaseEngine):
    """Execute agents via LangGraph (ReAct agent with tools)."""

    async def execute(self, spec: AgentSpec, prompt: str) -> AgentResult:
        try:
            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            provider = spec.detect_provider()
            base_url = "https://api.deepseek.com" if provider == "deepseek" else None
            api_key = os.getenv("DEEPSEEK_API_KEY") if provider == "deepseek" else None

            logger.info(f"DEBUG: [LangGraph] Provider={provider}, BaseURL={base_url}")
            model = ChatOpenAI(
                model=spec.model,
                temperature=spec.temperature,
                openai_api_key=api_key,
                base_url=base_url,
            )
            agent = create_react_agent(model, tools=[])

            start = time.monotonic()
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            latency = time.monotonic() - start

            output = result["messages"][-1].content if result["messages"] else ""
            return AgentResult(
                name=spec.name,
                status=AgentStatus.COMPLETED,
                framework="langgraph",
                output=output,
                latency_seconds=latency,
            )

        except ImportError:
            # Fallback to direct engine
            engine = DirectEngine()
            result = await engine.execute(spec, prompt)
            result.framework = "langgraph (fallback: direct)"
            return result

        except Exception as e:
            return AgentResult(
                name=spec.name,
                status=AgentStatus.FAILED,
                framework="langgraph",
                error=str(e),
            )


class CrewAIEngine(BaseEngine):
    """Execute agents via CrewAI (role-based crews)."""

    async def execute(self, spec: AgentSpec, prompt: str) -> AgentResult:
        try:
            from crewai import Agent, Crew, Task, Process

            start = time.monotonic()
            provider = spec.detect_provider()
            llm_config = {
                "model": spec.model,
                "temperature": spec.temperature,
            }
            if provider == "deepseek":
                # For CrewAI/LiteLLM, use 'deepseek/model-name' or set base_url
                llm_config["base_url"] = "https://api.deepseek.com"
                llm_config["api_key"] = os.getenv("DEEPSEEK_API_KEY")
                # Sometimes LiteLLM needs the provider prefix
                if not spec.model.startswith("deepseek/"):
                    llm_config["model"] = f"deepseek/{spec.model}"

            llm = None
            if provider == "deepseek":
                from langchain_openai import ChatOpenAI
                logger.info(f"DEBUG: [CrewAI] Using DeepSeek ChatOpenAI with base_url={llm_config['base_url']}")
                llm = ChatOpenAI(
                    model=spec.model,
                    openai_api_key=llm_config["api_key"],
                    base_url=llm_config["base_url"],
                    temperature=spec.temperature
                )
            else:
                llm = spec.model

            agent = Agent(
                role=spec.role or spec.name,
                goal=spec.goal or f"Complete the task: {spec.name}",
                backstory=spec.backstory or "You are a helpful AI assistant.",
                llm=llm,
                verbose=False,
            )

            task = Task(
                description=prompt,
                expected_output="A thorough, well-structured response.",
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
            latency = time.monotonic() - start

            return AgentResult(
                name=spec.name,
                status=AgentStatus.COMPLETED,
                framework="crewai",
                output=str(result),
                latency_seconds=latency,
            )

        except ImportError:
            engine = DirectEngine()
            result = await engine.execute(spec, prompt)
            result.framework = "crewai (fallback: direct)"
            return result

        except Exception as e:
            return AgentResult(
                name=spec.name,
                status=AgentStatus.FAILED,
                framework="crewai",
                error=str(e),
            )


class OpenAISDKEngine(BaseEngine):
    """Execute agents via OpenAI Agents SDK."""

    async def execute(self, spec: AgentSpec, prompt: str) -> AgentResult:
        try:
            from agents import Agent, Runner as OAIRunner

            start = time.monotonic()
            agent = Agent(
                name=spec.name,
                instructions=spec.system_prompt or "You are a helpful assistant.",
                model=spec.model,
            )
            result = await OAIRunner.run(agent, prompt)
            latency = time.monotonic() - start

            return AgentResult(
                name=spec.name,
                status=AgentStatus.COMPLETED,
                framework="openai_sdk",
                output=result.final_output,
                latency_seconds=latency,
            )

        except ImportError:
            engine = DirectEngine()
            result = await engine.execute(spec, prompt)
            result.framework = "openai_sdk (fallback: direct)"
            return result

        except Exception as e:
            return AgentResult(
                name=spec.name,
                status=AgentStatus.FAILED,
                framework="openai_sdk",
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------

from ..models import Framework  # noqa: E402

ENGINE_REGISTRY: dict[Framework, type[BaseEngine]] = {
    Framework.DIRECT: DirectEngine,
    Framework.LANGGRAPH: LangGraphEngine,
    Framework.CREWAI: CrewAIEngine,
    Framework.OPENAI_SDK: OpenAISDKEngine,
}


def get_engine(framework: Framework) -> BaseEngine:
    """Return the engine instance for a given framework."""
    engine_cls = ENGINE_REGISTRY.get(framework, DirectEngine)
    return engine_cls()
