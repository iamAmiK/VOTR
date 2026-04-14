<div align="center">

# MCP-Router: Dynamic Tool Routing for MCP Ecosystems

**Production-oriented hybrid retrieval and routing for large Model Context Protocol tool catalogs**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-service-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Paper](https://img.shields.io/badge/Paper-MCP--Router-blue?style=for-the-badge)](../paper/main.tex)
[![MCP](https://img.shields.io/badge/MCP-compatible-6f42c1?style=for-the-badge)](https://modelcontextprotocol.io/)

</div>

MCP-Router extends proactive tool retrieval ideas from MCP-Zero into a deployable service for live MCP stacks.  
It combines dense retrieval, BM25, SPLADE-lite features, field-aware reranking, and confidence-gated top-k selection to reduce prompt bloat while preserving routing accuracy.

---

## Why MCP-Router

Modern agent deployments face three practical constraints:

- **Prompt budget pressure:** Tool schemas can dominate context windows when thousands of tools are available.
- **Live ecosystem churn:** Servers and tools are added/removed frequently, requiring dynamic registration.
- **Routing ambiguity:** Similar tools across providers need robust ranking and uncertainty-aware handoff.

MCP-Router addresses these with:

- **Hybrid retrieval:** dense + BM25 + sparse expansion fused with weighted RRF.
- **Adaptive candidate set:** confidence policy returns 1/3/5 tools instead of fixed-k.
- **Hot registration:** discover and index new servers at runtime.
- **Schema compression:** compact tool lines for lower token overhead.
- **Session memory:** avoids redundant reinjection in multi-turn workflows.

---

## Project Layout

```text
MCP-Router/
├── src/mcp_router/                # Core router package
│   ├── router.py                  # FastAPI entrypoint
│   ├── retrieval/                 # Hybrid retrieval + reranking
│   ├── registry/                  # Tool/server registration and discovery
│   ├── session/                   # Session memory
│   ├── schema_compress/           # Compact schema formatting
│   └── stores/qdrant_store.py     # Optional vector-store adapter
├── scripts/                       # Index build and migration helpers
├── benchmarks/                    # Functional, efficiency, ablation runs
├── evaluation/                    # Evaluation helpers + result summaries
├── config.yaml                    # Default runtime config
├── RUN_EVALS.md                   # Benchmark runbook
└── README.md
```

---

## Installation

### Requirements

- Python 3.10+
- `OPENAI_API_KEY` for embedding-time query routing

### Setup

```bash
cd MCP-Router
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dev,eval]"
python -m pip install -e ".[qdrant]"
```

---

## Quickstart

### 1) Build an index

From MCP-Zero style export:

```bash
python scripts/build_index.py \
  --input "../MCP-Zero/MCP-tools/mcp_tools_with_embedding.json" \
  --output data/index
```

Small local index for fast iteration:

```bash
python scripts/build_index.py \
  --input "../MCP-Zero/MCP-tools/mcp_tools_with_embedding.json" \
  --output data/index \
  --max-servers 20
```

### 2) Configure API key

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

### 3) Start the API

```bash
python -m uvicorn mcp_router.router:app --host 0.0.0.0 --port 8765
```

---

## API Examples

### `POST /route`

```json
{
  "server_intent": "GitHub repositories and API",
  "tool_intent": "search for repositories matching a query string",
  "session_id": "user-123"
}
```

### `POST /register`

```json
{
  "server": {
    "name": "MyServer",
    "description": "Does X",
    "summary": "Short summary for routing",
    "tools": [
      {"name": "do_x", "description": "Runs X", "parameter": {}}
    ],
    "source": "user"
  }
}
```

### `POST /register/discover`

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\repo"],
  "server_name": "Filesystem",
  "server_description": "Read and write files in a local workspace",
  "timeout_seconds": 30
}
```

---

## Evaluation

Use the benchmark runbook in `RUN_EVALS.md`. Common entry points:

```bash
python benchmarks/functional_correctness/_run_all_suites.py
python benchmarks/efficiency/run_latency.py
python benchmarks/baselines_ablations/run_profiles.py
```

For paper tables and summary scripts:

```bash
python benchmarks/efficiency/build_paper_comparison.py
python scripts/generate_results_tables.py
```

---

## Optional Orchestrator Integration

For end-to-end integration testing with multi-step tool-calling loops, use the companion repository:

- [`VOTR-Orchestrator`](https://github.com/<your-username>/VOTR-Orchestrator)

This `VOTR` repository is the core router implementation. The orchestrator is maintained separately so users can adopt the router without extra orchestration dependencies.

---

## Reproducibility Notes

- Main config is in `config.yaml`; local overrides should go in `config.local.yaml`.
- Generated index artifacts live under `data/index*` and are intentionally ignored from source control.
- Generated benchmark outputs live under `benchmarks/results` and `evaluation/results`.

---

## Citation

If you use MCP-Router in research, cite the paper in `paper/main.tex` and your final bibliographic record.
