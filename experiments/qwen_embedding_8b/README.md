# Qwen3 Embedding 8B Pipeline (Isolated)

This folder provides an isolated pipeline for running VOTR with
`qwen/qwen3-embedding-8b` via OpenRouter, without changing the existing
OpenAI, Gemma, BGE, or Gemini code paths.

## What stays untouched

- Existing router entrypoint: `mcp_router.router:app`
- Existing OpenAI index: `data/index`
- Existing Gemma index: `data/index_gemma`
- Existing BGE index: `data/index_bge_large_en`
- Existing Gemini index: `data/index_gemini_embedding_2`

## What this folder adds

- Qwen embedder: `experiments/qwen_embedding_8b/embedder_qwen.py`
- Qwen router app: `experiments/qwen_embedding_8b/router_qwen.py`
- Meta -> embedding JSON builder: `experiments/qwen_embedding_8b/build_embedding_json_from_meta.py`
- Main-suite benchmark runner: `experiments/qwen_embedding_8b/run_main_suites_qwen.py`
- Separate config: `experiments/qwen_embedding_8b/config.qwen.yaml`

## 1) Install dependencies

```powershell
pip install requests
```

## 2) Set API key in environment

```powershell
$env:OPENROUTER_API_KEY = "YOUR_KEY_HERE"
```

## 3) Build Qwen embedding JSON from existing metadata

Input source: `data/index/meta.json` (text fields only)

Output: `data/qwen_embedding_8b/mcp_tools_with_embedding.qwen3-embedding-8b.json`

```powershell
python experiments/qwen_embedding_8b/build_embedding_json_from_meta.py `
  --meta data/index/meta.json `
  --model qwen/qwen3-embedding-8b `
  --out data/qwen_embedding_8b/mcp_tools_with_embedding.qwen3-embedding-8b.json
```

The default output dimension is `4096`. If OpenRouter returns a different
dimension for this model, rebuild with the matching value using `--dimensions`.

## 4) Build a separate Qwen index

```powershell
python scripts/build_index.py `
  --input data/qwen_embedding_8b/mcp_tools_with_embedding.qwen3-embedding-8b.json `
  --output data/index_qwen_embedding_8b
```

## 5) Run Qwen router only

```powershell
python -m uvicorn experiments.qwen_embedding_8b.router_qwen:app --host 127.0.0.1 --port 8778
```

Health check:

```powershell
curl http://127.0.0.1:8778/health
```

## 6) Run main functional suites against Qwen router

This writes to `benchmarks/results/functional_correctness_qwen_embedding_8b`.

```powershell
python experiments/qwen_embedding_8b/run_main_suites_qwen.py
```

Main suites:

- `single_tool.clean`
- `multi_hop.large.cross_app`
- `multi_hop.medium_250.cross_app`
- `multi_tool.large.single_turn`
- `multi_tool.medium_250.single_turn`
- `ambiguity_collision.priority`
- `robustness_safety.priority`
