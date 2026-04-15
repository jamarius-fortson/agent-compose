"""Pipeline execution engine — the core orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from .engines import get_engine
from .loaders.spec import build_execution_order, load_spec, resolve_template
from .models import AgentResult, AgentStatus, PipelineResult, PipelineSpec

logger = logging.getLogger("agent-compose")


class Runner:
    """Execute a pipeline spec by walking the agent DAG."""

    def __init__(self, spec: PipelineSpec, user_input: str = ""):
        self.spec = spec
        self.user_input = user_input
        self.outputs: dict[str, str] = {}
        self.results: dict[str, AgentResult] = {}

    async def run(self) -> PipelineResult:
        """Execute the full pipeline, respecting dependencies and parallelism."""
        levels = build_execution_order(self.spec)
        agent_map = {a.name: a for a in self.spec.agents}

        pipeline_result = PipelineResult(
            pipeline_name=self.spec.name,
            description=self.spec.description,
        )

        for level in levels:
            # Execute all agents at this level concurrently
            tasks = []
            for agent_name in level:
                agent_spec = agent_map[agent_name]
                tasks.append(self._execute_agent(agent_spec))

            level_results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent_name, result in zip(level, level_results):
                if isinstance(result, Exception):
                    agent_result = AgentResult(
                        name=agent_name,
                        status=AgentStatus.FAILED,
                        error=str(result),
                    )
                else:
                    agent_result = result

                self.results[agent_name] = agent_result
                if agent_result.status == AgentStatus.COMPLETED:
                    self.outputs[agent_name] = agent_result.output
                pipeline_result.agent_results.append(agent_result)

        # Save output files
        for agent_spec in self.spec.agents:
            if agent_spec.output and agent_spec.name in self.outputs:
                output_dir = self.spec.settings.get("output_dir", ".")
                output_path = Path(output_dir) / agent_spec.output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(self.outputs[agent_spec.name])
                logger.info(f"Output saved to {output_path}")

        return pipeline_result

    async def _execute_agent(self, spec) -> AgentResult:
        """Execute a single agent through its framework engine."""
        # Resolve the prompt template
        prompt_template = spec.prompt or spec.system_prompt or ""
        if not prompt_template:
            # No explicit prompt — use user input + upstream outputs
            parts = [self.user_input]
            for agent in self.spec.agents:
                for target in agent.connects_to:
                    if target == spec.name and agent.name in self.outputs:
                        parts.append(
                            f"\n--- Output from {agent.name} ---\n"
                            f"{self.outputs[agent.name]}"
                        )
            prompt = "\n".join(parts)
        else:
            prompt = resolve_template(
                prompt_template,
                self.user_input,
                self.outputs,
                self.spec.env,
            )

        # If no user input in prompt, prepend it
        if self.user_input and self.user_input not in prompt:
            prompt = f"{self.user_input}\n\n{prompt}"

        # Check condition
        if spec.condition:
            if not self._evaluate_condition(spec.condition):
                return AgentResult(
                    name=spec.name,
                    status=AgentStatus.SKIPPED,
                    framework=spec.framework.value,
                )

        # Get the right engine and execute
        engine = get_engine(spec.framework)
        logger.info(
            f"[{spec.name}] Starting ({spec.model}, {spec.framework.value} engine)"
        )

        try:
            result = await asyncio.wait_for(
                engine.execute(spec, prompt),
                timeout=spec.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = AgentResult(
                name=spec.name,
                status=AgentStatus.FAILED,
                framework=spec.framework.value,
                error=f"Timeout after {spec.timeout_seconds}s",
            )

        if result.status == AgentStatus.COMPLETED:
            logger.info(
                f"[{spec.name}] Completed in {result.latency_seconds:.1f}s "
                f"({result.total_tokens:,} tokens, ${result.cost_usd:.3f})"
            )
        else:
            logger.error(f"[{spec.name}] Failed: {result.error}")

        return result

    def _evaluate_condition(self, condition: str) -> bool:
        """Simple condition evaluator for {{agent.output}} contains X."""
        resolved = resolve_template(
            condition, self.user_input, self.outputs, self.spec.env
        )
        if " contains " in resolved:
            parts = resolved.split(" contains ", 1)
            return parts[1].strip().lower() in parts[0].strip().lower()
        return bool(resolved.strip())


async def run_pipeline(
    spec_path: str = "agent-compose.yaml",
    user_input: str = "",
    dry_run: bool = False,
) -> PipelineResult:
    """High-level entry point to load and execute a pipeline."""
    spec = load_spec(spec_path)

    if dry_run:
        levels = build_execution_order(spec)
        result = PipelineResult(pipeline_name=spec.name, description=spec.description)
        for level in levels:
            for name in level:
                agent = next(a for a in spec.agents if a.name == name)
                result.agent_results.append(
                    AgentResult(
                        name=name,
                        status=AgentStatus.PENDING,
                        framework=agent.framework.value,
                    )
                )
        return result

    runner = Runner(spec, user_input)
    return await runner.run()
