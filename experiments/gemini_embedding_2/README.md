# Gemini Embedding Pipeline (Isolated)

This folder provides an **isolated** pipeline for running VOTR with the Gemini
Embedding API, without changing existing OpenAI, Gemma, or BGE code paths.

## What stays untouched

- Existing router entrypoint: `mcp_router.router:app`
- Existing OpenAI index: `data/index`
- Existing Gemma index: `data/index_gemma`
- Existing BGE index: `data/index_bge_large_en`

## What this folder adds

- Gemini embedder: `experiments/gemini_embedding_2/embedder_gemini.py`
- Gemini router app: `experiments/gemini_embedding_2/router_gemini.py`
- Meta -> embedding JSON builder: `experiments/gemini_embedding_2/build_embedding_json_from_meta.py`
- One-shot benchmark runner: `experiments/gemini_embedding_2/run_all_suites_gemini.py`
- Separate config: `experiments/gemini_embedding_2/config.gemini.yaml`

## 1) Install dependencies

```powershell
pip install requests
```

## 2) Set API key in environment (do not hardcode in files)

```powershell
$env:OPENROUTER_API_KEY = "YOUR_KEY_HERE"
```

## 3) Build Gemini embedding JSON from existing metadata

Input source: `data/index/meta.json` (text fields only)

Output: `data/gemini_embedding_2/mcp_tools_with_embedding.gemini-embedding-2-preview.json`

```powershell
python experiments/gemini_embedding_2/build_embedding_json_from_meta.py `
  --meta data/index/meta.json `
  --model google/gemini-embedding-2-preview `
  --out data/gemini_embedding_2/mcp_tools_with_embedding.gemini-embedding-2-preview.json
```

Progress is saved after **each server** to a checkpoint file (default:
`data/gemini_embedding_2/mcp_tools_with_embedding.gemini-embedding-2-preview.checkpoint.json`).
If the build stops on quota (429), set a new `OPENROUTER_API_KEY` and run the **same
command** again; it resumes automatically. Use `--no-resume` to start from
scratch (and remove the checkpoint file if you want a clean run).

## 4) Build a separate Gemini index

```powershell
python scripts/build_index.py `
  --input data/gemini_embedding_2/mcp_tools_with_embedding.gemini-embedding-2-preview.json `
  --output data/index_gemini_embedding_2
```

## 5) Run Gemini router only

```powershell
python -m uvicorn experiments.gemini_embedding_2.router_gemini:app --host 127.0.0.1 --port 8777
```

Health check:

```powershell
curl http://127.0.0.1:8777/health
```

## 6) Run full functional suite against Gemini router

This writes to `benchmarks/results/functional_correctness_gemini_embedding_2`.

```powershell
python experiments/gemini_embedding_2/run_all_suites_gemini.py
```
