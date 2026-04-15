<div align="center">

# VOTR: Vector Orchestrated Tool Retrieval

**Production-grade MCP tool retrieval and routing for large, dynamic tool ecosystems**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-service-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-compatible-6f42c1?style=for-the-badge)](https://modelcontextprotocol.io/)
[![Paper](https://img.shields.io/badge/Paper-VOTR-blue?style=for-the-badge)](https://github.com/iamAmiK/VOTR/blob/main/docs/(Draft)%20VOTR%20-%20Vector%20Orchestrated%20Tool%20Retrieval%20for%20Scalable%20Multi-Agent%20Systems.pdf)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-assets-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/a13awd/VOTR)
[![Orchestrator](https://img.shields.io/badge/Orchestrator-VOTR--Orchestrator-8a2be2?style=for-the-badge)](https://github.com/iamAmiK/VOTR-Orchestrator)

</div>

VOTR is the system described in the paper `[VOTR: Vector Orchestrated Tool Retrieval for Scalable Multi-Agent Systems]`([paper](https://github.com/iamAmiK/VOTR/blob/main/docs/VOTR%20-%20Vector%20Orchestrated%20Tool%20Retrieval%20for%20Scalable%20Multi-Agent%20Systems.pdf))`: a FastAPI service that retrieves and ranks MCP tools before model invocation, then returns a compact candidate set for schema injection. It is designed to preserve retrieval quality while reducing prompt overhead and supporting live MCP registry updates.

---

## Highlights

- **Tool retrieval accuracy:** Successfully pulls tools from a dataset of 309 servers, 2806 tools at 96.4% recall with smaller sets providing 100% recall.
- **Paper-aligned system:** Implements the VOTR retrieval stack and evaluation workflow from the manuscript.
- **Hybrid retrieval core:** Dense similarity + BM25 + SPLADE-lite fused with weighted Reciprocal Rank Fusion.
- **Field-aware reranking:** Structured overlap scoring across server/tool name, description, and parameter signals.
- **Confidence-gated handoff:** Dynamic \(k \in \{1,3,5\}\) selection calibrated from non-conformity style thresholds + optional abstention protocol.
- **Registry built for live MCP:** Runtime discovery and hot registration through stdio and HTTP/SSE pathways.
- **Robustness features:** Overlap-aware disambiguation, abstention/null-route guards, and regression-oriented suites.
- **Token efficiency:** Compact schema lines (paper reports 26.2 tokens/tool vs 93.5 MCP-Zero-style wrapper on the indexed corpus).

---

## Problem Context

In large MCP deployments, injecting every tool schema is not viable. The paper motivates this with a 309-server / 2,806-tool setting, where full schema injection can exceed practical prompt budgets with high accuracy and latency. VOTR treats tool selection as a retrieval and ranking problem with uncertainty-aware candidate sizing, instead of static top-k injection.

---

## Repository Scope

This repository is the **core VOTR router implementation**.

- `src/mcp_router/`: retrieval engine, reranking, confidence policy, registry, API
- `benchmarks/`: functional correctness, ablations, efficiency, confidence, robustness
- `evaluation/`: reporting and external benchmark adapters (including LiveMCPBench tooling)
- `scripts/`: index/data preparation and result table generation
- `docs/`: implementation notes and policy documentation

Companion integration loop (optional, separate repo):
- [`VOTR-Orchestrator`](https://github.com/iamAmiK/VOTR-Orchestrator)

`VOTR-Orchestrator` is used for E2E integration testing around the router. In practice, it acts as the execution harness that sends routed tool candidates into a production-style multi-step agent loop, then validates tool-calling behavior across full conversations and chained tasks.

---

## Hugging Face Assets (Embeddings / Index Artifacts)

Prebuilt embeddings derived and expanded from MCP-Zero embeddings and index artifacts are published here:

- **Dataset/artifacts page:** [https://huggingface.co/datasets/a13awd/VOTR](https://huggingface.co/datasets/a13awd/VOTR)

Current published payload includes approximately **623 MB** of data uploaded from the `MCP-Router/data` tree, covering precomputed routing artifacts for reproducibility.

Hosted artifact types include:

- Precomputed tool/server embedding shards (`.npy`)
- Index metadata (`meta.json`, registry export, schema docs)
- Benchmark-ready subset indexes (small/medium/full, LiveMCPBench variant)
- Versioned checksum manifest for reproducibility
- 3072 dimension embedding of 309 MCP servers + 2806 tools made from text-embedding-3-large (OpenAI) ≈ 340MB

---

## Installation

### Requirements

- Python 3.10+
- `OPENAI_API_KEY` (for query-time embedding in default configuration)

### Setup

```bash
python -m pip install -e .
```

Optional extras (for future development and integration):

```bash
python -m pip install -e ".[dev,eval]"
python -m pip install -e ".[qdrant]"
```

---

## Quickstart

### 1) Build index locally (if not using Hugging Face artifacts - requires OpenAI API Key)

```bash
python scripts/build_index.py \
  --input "../MCP-Zero/MCP-tools/mcp_tools_with_embedding.json" \
  --output data/index
```

Small dev build:

```bash
python scripts/build_index.py \
  --input "../MCP-Zero/MCP-tools/mcp_tools_with_embedding.json" \
  --output data/index \
  --max-servers 20
```

### 2) Set API key

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

### 3) Run the router

```bash
python -m uvicorn mcp_router.router:app --host 0.0.0.0 --port 8765
```

---

## API Surface

Common endpoints:

- `POST /route` - retrieve and rank candidate tools for a request
- `POST /register` - register a server/tools payload directly
- `POST /register/discover` - discover/register from stdio MCP server
- `POST /register/discover/sse` - discover/register from HTTP endpoint

Minimal `POST /route` body:

```json
{
  "server_intent": "GitHub repositories and API",
  "tool_intent": "search repositories by query",
  "session_id": "ss2013"
}
```

---

## Evaluation at a Glance

Paper evaluation covers:

- **Single-tool routing** across small / medium / large suites
- **Multi-hop** and **multi-tool** scaled suites for small / medium / large suites (including long-hop stress runs)
- **Ablations** (dense-only, BM25-only, dense+BM25, fullstack)
- **Confidence calibration and handoff behavior**
- **Latency and token-efficiency measurements**
- **Out-of-distribution stress test on LiveMCPBench**

Runbook:

Common commands:

```bash
python benchmarks/functional_correctness/_run_all_suites.py
python benchmarks/efficiency/run_latency.py
python benchmarks/baselines_ablations/run_profiles.py
python benchmarks/efficiency/build_paper_comparison.py
python scripts/generate_results_tables.py
```

---

## Reproducibility Notes

- Use `config.yaml` for baseline settings; put machine-specific overrides in `config.local.yaml`.
- Generated runtime artifacts and benchmark outputs are intentionally ignored by `.gitignore`.
- For paper-consistent runs, keep index build source/version and benchmark suite versions fixed.

---

## Citation

- TODO : create citation
