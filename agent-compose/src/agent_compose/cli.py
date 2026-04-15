"""CLI for agent-compose."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from .loaders.spec import build_execution_order, load_spec
from .models import AgentStatus, PipelineResult
from .runner import run_pipeline


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )


def _print_result(result: PipelineResult) -> None:
    """Rich terminal output for pipeline results."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.print()

        table = Table(
            title=f"agent-compose — {result.pipeline_name}",
            box=box.ROUNDED,
            header_style="bold cyan",
            title_style="bold white",
        )
        table.add_column("Agent", style="bold", min_width=14)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Framework", width=12)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Time", justify="right", width=7)
        table.add_column("Cost", justify="right", width=8)

        for r in result.agent_results:
            if r.status == AgentStatus.COMPLETED:
                status = "[green]✅ Done[/]"
            elif r.status == AgentStatus.FAILED:
                status = "[red]❌ Fail[/]"
            elif r.status == AgentStatus.SKIPPED:
                status = "[dim]⏭ Skip[/]"
            else:
                status = "[yellow]⏳ Pend[/]"

            table.add_row(
                r.name,
                status,
                r.framework,
                f"{r.total_tokens:,}" if r.total_tokens else "—",
                f"{r.latency_seconds:.1f}s" if r.latency_seconds else "—",
                f"${r.cost_usd:.3f}" if r.cost_usd else "—",
            )

        table.add_section()
        total = len(result.agent_results)
        done = result.completed_count
        style = "green bold" if result.all_succeeded else "red bold"
        table.add_row(
            "[bold]TOTAL[/]",
            f"[{style}]{done}/{total}[/{style}]",
            "",
            f"[bold]{result.total_tokens:,}[/bold]",
            f"[bold]{result.total_latency:.1f}s[/bold]",
            f"[bold]${result.total_cost:.3f}[/bold]",
        )

        console.print(table)

        # Show output files
        for r in result.agent_results:
            # Find the agent spec with output file
            pass

        # Show errors
        for r in result.agent_results:
            if r.error:
                console.print(f"  [red]⚠ {r.name}:[/] {r.error}")

        console.print()

    except ImportError:
        _print_result_plain(result)


def _print_result_plain(result: PipelineResult) -> None:
    print(f"\nagent-compose — {result.pipeline_name}")
    print("=" * 60)
    for r in result.agent_results:
        status = "DONE" if r.status == AgentStatus.COMPLETED else r.status.value.upper()
        print(
            f"  {r.name:<16} {status:<8} {r.framework:<12} "
            f"{r.total_tokens:>6} tok  {r.latency_seconds:>5.1f}s  "
            f"${r.cost_usd:.3f}"
        )
    print("=" * 60)
    print(
        f"  Total: {result.completed_count}/{len(result.agent_results)} "
        f"| {result.total_tokens:,} tokens | ${result.total_cost:.3f}"
    )
    print()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="agent-compose")
def cli():
    """Docker Compose for multi-agent AI systems."""


@cli.command()
@click.option("-f", "--file", "spec_file", default="agent-compose.yaml",
              help="Path to spec file")
@click.option("--input", "user_input", default="", help="Input text for the pipeline")
@click.option("--input-file", type=click.Path(exists=True), help="Read input from file")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--dry-run", is_flag=True, help="Show execution plan without running")
@click.option("--log-level", default="info", help="Log level")
@click.option("--max-cost", type=float, help="Max total cost in USD")
@click.option("--timeout", type=int, help="Global timeout in seconds")
@click.option("--exit-code", is_flag=True, help="Non-zero exit on failure")
@click.option("-o", "--output", type=click.Path(), help="Save JSON results to file")
def up(spec_file, user_input, input_file, fmt, dry_run, log_level,
       max_cost, timeout, exit_code, output):
    """Spin up the agent pipeline."""
    _setup_logging(log_level)

    if input_file:
        user_input = Path(input_file).read_text().strip()

    result = asyncio.run(run_pipeline(spec_file, user_input, dry_run))

    if fmt == "json":
        data = result.to_dict()
        json_str = json.dumps(data, indent=2)
        if output:
            Path(output).write_text(json_str)
        else:
            click.echo(json_str)
    else:
        _print_result(result)
        if output:
            Path(output).write_text(json.dumps(result.to_dict(), indent=2))

    # CI checks
    failed = False
    if not result.all_succeeded:
        failed = True
    if max_cost and result.total_cost > max_cost:
        click.echo(f"⚠ Cost ${result.total_cost:.3f} exceeds limit ${max_cost:.3f}")
        failed = True

    if exit_code and failed:
        sys.exit(1)


@cli.command()
@click.option("-f", "--file", "spec_file", default="agent-compose.yaml")
def validate(spec_file):
    """Validate the YAML spec without running."""
    try:
        spec = load_spec(spec_file)
        levels = build_execution_order(spec)

        try:
            from rich.console import Console
            console = Console()
            console.print(f"\n[green]✅ Valid:[/] {spec.name}")
            console.print(f"   Agents: {len(spec.agents)}")
            console.print(f"   Execution levels: {len(levels)}")
            for i, level in enumerate(levels):
                console.print(f"   Level {i}: {', '.join(level)}")
            console.print()
        except ImportError:
            print(f"\n✅ Valid: {spec.name}")
            print(f"   Agents: {len(spec.agents)}")
            for i, level in enumerate(levels):
                print(f"   Level {i}: {', '.join(level)}")
            print()

    except Exception as e:
        click.echo(f"❌ Invalid: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("-f", "--file", "spec_file", default="agent-compose.yaml")
@click.option("--format", "fmt", type=click.Choice(["ascii", "mermaid"]), default="ascii")
def graph(spec_file, fmt):
    """Show the agent topology as a graph."""
    spec = load_spec(spec_file)

    if fmt == "mermaid":
        click.echo("graph TD")
        for agent in spec.agents:
            label = f'{agent.name}["{agent.name}<br/>{agent.model}"]'
            click.echo(f"    {label}")
            for target in agent.connects_to:
                click.echo(f"    {agent.name} --> {target}")
    else:
        # ASCII graph
        levels = build_execution_order(spec)
        agent_map = {a.name: a for a in spec.agents}
        for i, level in enumerate(levels):
            names = "  ".join(
                f"{n} ({agent_map[n].framework.value})" for n in level
            )
            prefix = "→ " if i > 0 else "  "
            click.echo(f"{prefix}Level {i}: {names}")


@cli.command()
@click.option("-f", "--file", "spec_file", default="agent-compose.yaml")
@click.argument("agent_name")
@click.option("--input", "user_input", default="", help="Input text")
def run(spec_file, agent_name, user_input):
    """Run a single agent in isolation (for debugging)."""
    from .engines import get_engine

    spec = load_spec(spec_file)
    agent = next((a for a in spec.agents if a.name == agent_name), None)
    if not agent:
        click.echo(f"Agent '{agent_name}' not found in spec")
        sys.exit(1)

    async def _run():
        engine = get_engine(agent.framework)
        prompt = agent.prompt or agent.system_prompt or user_input
        if user_input and user_input not in prompt:
            prompt = f"{user_input}\n\n{prompt}"
        return await engine.execute(agent, prompt)

    result = asyncio.run(_run())
    click.echo(f"\nAgent: {result.name} ({result.framework})")
    click.echo(f"Status: {result.status.value}")
    click.echo(f"Tokens: {result.total_tokens:,}")
    click.echo(f"Latency: {result.latency_seconds:.1f}s")
    if result.error:
        click.echo(f"Error: {result.error}")
    else:
        click.echo(f"\nOutput:\n{result.output}")


def main():
    cli()


if __name__ == "__main__":
    main()
