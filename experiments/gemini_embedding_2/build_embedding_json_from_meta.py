#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

CHECKPOINT_VERSION = 1


def _tool_text(tool: dict[str, Any]) -> str:
    name = str(tool.get("name", "") or "")
    desc = str(tool.get("description", "") or "")
    return (desc or name or "unknown_tool").strip()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _embed(api_key: str, model: str, texts: list[str], dimensions: int) -> list[list[float]]:
    if not hasattr(_embed, "_last_request_ts"):
        _embed._last_request_ts = 0.0  # type: ignore[attr-defined]
    min_interval_s = 1.1
    elapsed = time.monotonic() - _embed._last_request_ts  # type: ignore[attr-defined]
    if elapsed < min_interval_s:
        time.sleep(min_interval_s - elapsed)

    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
        "dimensions": dimensions,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response_data = None
    for attempt in range(8):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code >= 400:
                msg = f"{resp.status_code} {resp.text}"
                if "429" not in msg:
                    raise RuntimeError(f"OpenRouter embed failed: {msg}")
                if attempt == 7:
                    raise RuntimeError(f"OpenRouter embed failed after retries: {msg}")
                time.sleep(50)
                continue
            response_data = resp.json()
            data = response_data.get("data") or []
            if data:
                _embed._last_request_ts = time.monotonic()  # type: ignore[attr-defined]
                break
            # Some transient/provider responses may omit `data`; retry with backoff.
            if attempt == 7:
                raise RuntimeError(f"OpenRouter embedding response had no data: {response_data}")
            _embed._last_request_ts = time.monotonic()  # type: ignore[attr-defined]
            time.sleep(6)
            continue
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(50)

    if response_data is None:
        raise ValueError("OpenRouter embedding response was empty")
    data = response_data.get("data") or []
    if not data:
        raise ValueError(f"OpenRouter embedding response had no data: {response_data}")
    return [list(item["embedding"]) for item in data]


def _embed_one_server(
    api_key: str,
    model: str,
    dimensions: int,
    batch_size: int,
    server: dict[str, Any],
) -> dict[str, Any]:
    s_name = str(server.get("name", "") or "")
    s_desc = str(server.get("description", "") or "")
    s_sum = str(server.get("summary", "") or s_desc)

    s_vecs = _embed(api_key, model, [s_desc or s_name, s_sum or s_desc or s_name], dimensions)
    s_desc_vec = s_vecs[0]
    s_sum_vec = s_vecs[1]

    tools = server.get("tools", []) or []
    tool_texts = [_tool_text(t) for t in tools]

    out_tools: list[dict[str, Any]] = []
    for j in range(0, len(tool_texts), batch_size):
        chunk_texts = tool_texts[j : j + batch_size]
        chunk_vecs = _embed(api_key, model, chunk_texts, dimensions)
        for k, vec in enumerate(chunk_vecs):
            tool = tools[j + k]
            out_tools.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameter": tool.get("parameter", {}) or {},
                    "description_embedding": vec,
                }
            )

    return {
        "name": s_name,
        "description": s_desc,
        "summary": s_sum,
        "description_embedding": s_desc_vec,
        "summary_embedding": s_sum_vec,
        "tools": out_tools,
    }


def _try_load_checkpoint(
    checkpoint_path: Path,
    meta_resolved: Path,
    model: str,
    total_servers: int,
) -> list[dict[str, Any]] | None:
    if not checkpoint_path.is_file():
        return None
    try:
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if raw.get("version") != CHECKPOINT_VERSION:
        return None
    if raw.get("model") != model:
        return None
    if Path(raw.get("meta", "")).resolve() != meta_resolved:
        return None
    completed = raw.get("completed")
    if not isinstance(completed, list):
        return None
    if len(completed) > total_servers:
        return None
    return completed


def _save_checkpoint(
    checkpoint_path: Path,
    meta_resolved: Path,
    model: str,
    completed: list[dict[str, Any]],
) -> None:
    _atomic_write_json(
        checkpoint_path,
        {
            "version": CHECKPOINT_VERSION,
            "meta": str(meta_resolved),
            "model": model,
            "completed": completed,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create MCP-Zero style embedding JSON from index meta.json using Gemini embeddings."
    )
    ap.add_argument("--meta", type=Path, default=Path("data/index/meta.json"))
    ap.add_argument(
        "--model",
        type=str,
        default="google/gemini-embedding-2-preview",
        help="OpenRouter embedding model id",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/gemini_embedding_2/mcp_tools_with_embedding.gemini-embedding-2-preview.json"),
    )
    ap.add_argument("--dimensions", type=int, default=3072)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Resume file (default: <out>.checkpoint.json)",
    )
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint and start fresh")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    args = ap.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    meta_resolved = args.meta.resolve()
    raw = json.loads(args.meta.read_text(encoding="utf-8"))
    servers: list[dict[str, Any]] = raw.get("servers", [])
    if not servers:
        raise SystemExit(f"No servers found in {args.meta}")

    checkpoint_path = args.checkpoint or args.out.with_name(args.out.stem + ".checkpoint.json")

    out_servers: list[dict[str, Any]] = []
    start_idx = 0

    if not args.no_resume:
        loaded = _try_load_checkpoint(checkpoint_path, meta_resolved, args.model, len(servers))
        if loaded is not None:
            out_servers = loaded
            start_idx = len(out_servers)
            for i in range(start_idx):
                if str(servers[i].get("name", "")) != str(out_servers[i].get("name", "")):
                    raise SystemExit(
                        f"Checkpoint server name mismatch at index {i}: "
                        f"meta={servers[i].get('name')!r} checkpoint={out_servers[i].get('name')!r}. "
                        f"Delete {checkpoint_path} or use --no-resume."
                    )
            if start_idx >= len(servers):
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(out_servers, ensure_ascii=False), encoding="utf-8")
                if checkpoint_path.is_file():
                    checkpoint_path.unlink(missing_ok=True)
                print(f"Checkpoint already complete ({len(out_servers)} servers). Wrote {args.out}", flush=True)
                return
            print(
                f"Resuming from server {start_idx + 1}/{len(servers)} (checkpoint: {checkpoint_path})",
                flush=True,
            )

    for i in range(start_idx, len(servers)):
        done = _embed_one_server(api_key, args.model, args.dimensions, args.batch_size, servers[i])
        out_servers.append(done)
        _save_checkpoint(checkpoint_path, meta_resolved, args.model, out_servers)

        if (i + 1) % 25 == 0 or (i + 1) == len(servers):
            print(f"Embedded {i + 1}/{len(servers)} servers", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_servers, ensure_ascii=False), encoding="utf-8")
    if checkpoint_path.is_file():
        checkpoint_path.unlink(missing_ok=True)
    print(f"Wrote {len(out_servers)} servers to {args.out}", flush=True)


if __name__ == "__main__":
    main()
