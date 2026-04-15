# Contributing to agent-compose

## Setup
```bash
git clone https://github.com/daniellopez882/agent-compose.git
cd agent-compose
pip install -e ".[dev]"
pytest tests/ -v
```

## High-Impact Contributions

### Framework Engines (most wanted)
- **LangGraph engine** — full ReAct agent with tool support
- **CrewAI engine** — role-based crews from YAML
- **OpenAI Agents SDK engine** — handoff patterns

### Built-in Tools
- `web_search` — Tavily/SerpAPI integration
- `code_exec` — sandboxed Python execution
- `file_read` / `file_write` — local file operations

### Features
- **MCP integration** — connect to MCP servers from YAML
- **Graph visualization** — `agent-compose graph` as Mermaid
- **Streaming** — stream agent outputs in real-time
- **Shared memory** — cross-agent state

### Examples
Share your `agent-compose.yaml` pipelines!

## Code Style
- Python 3.10+ with type hints
- Lint: `ruff check src/ tests/`
- Tests required for new features
- Conventional commits: `feat:`, `fix:`, `docs:`

## Pull Requests
1. Fork → feature branch → tests → PR
2. One feature per PR
3. Include a YAML example if adding a new capability
