# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reference project for the Agent Development Workshop 2026. A LangGraph agent starter template using Python, LangChain, and OpenAI.

## Commands

```bash
# Install dependencies (including dev)
pip install -e ".[dev]"
# or with uv (recommended — pyproject.toml uses dependency-groups which is uv-native)
uv sync --dev

# Run the LangGraph development server (serves graphs defined in langgraph.json)
langgraph dev

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy .

# Run tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name
```

## Architecture

### Graph Definitions (`langgraph.json`)

The project exposes two LangGraph graphs, each defined as a Python variable in its module:

| Graph name | Module | Export |
|---|---|---|
| `agent` | `src/agent/start.py` | `agent` |
| `models` | `src/module_1/1.1_models.py` | `models` |

`langgraph dev` reads `langgraph.json` and serves all listed graphs via the LangGraph API.

### Package Layout

- `src/agent/` — main agent logic; installed as both `agent` and `langgraph.templates.agent`
- `src/module_1/` — workshop module files (numbered, e.g. `1.1_models.py`)

### Environment Variables

Copy `.env.example` to `.env`. The LangGraph server loads `.env` automatically. Key variables:
- `LANGSMITH_PROJECT` — isolates traces in LangSmith (default: `new-agent`)
- Add LLM provider API keys here (e.g. `OPENAI_API_KEY`)

### Linting Rules

Ruff is configured with `E`, `F`, `I`, `D` (Google docstring convention), `T201`, `UP` rules. `D417` (param docs), `E501` (line length), and a few `UP` rules are ignored. Tests skip all `D` and `UP` rules.
