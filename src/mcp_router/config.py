from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class RouterConfig(BaseModel):
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    index_dir: Path = Field(default=Path("./data/index"))
    session_ttl_seconds: int = 86400
    default_top_servers: int = 8
    bm25_candidate_multiplier: int = 4
    rrf_k: int = 60
    dense_retrieval_enabled: bool = True
    bm25_retrieval_enabled: bool = True
    dense_rrf_weight: float = 1.0
    bm25_rrf_weight: float = 1.0
    openai_api_key_env: str = "OPENAI_API_KEY"
    adaptive_gap_confident: float = 0.12
    adaptive_min_tools: int = 1
    adaptive_max_tools: int = 8
    bulk_penalty_singular: float = 0.82
    bulk_boost_plural: float = 1.08
    splade_enabled: bool = True
    splade_candidate_multiplier: int = 4
    splade_rrf_weight: float = 0.35
    handoff_enabled: bool = True
    server_score_max_weight: float = 1.0
    server_score_mean_weight: float = 0.0
    handoff_gap_high: float = 0.0010426450507904708
    handoff_gap_medium: float = 0.000836960501159334
    handoff_k_high: int = 1
    handoff_k_medium: int = 3
    handoff_k_low: int = 5
    #: When handoff confidence is ``low``, ensure at least this many tools are
    #: returned in ``RouteResponse.tools`` (capped by ``adaptive_max_tools`` and
    #: available candidates).  Does not change ranking or the confidence label;
    #: it only widens the candidate list so downstream agents can disambiguate.
    low_confidence_retrieval_floor_k: int = 5
    field_aware_enabled: bool = True
    field_aware_rerank_head: int = 24
    field_aware_score_window: float = 0.003
    field_bonus_scale: float = 0.0015
    field_weight_server_name: float = 0.18
    field_weight_server_summary: float = 0.08
    field_weight_tool_name: float = 0.35
    field_weight_tool_description: float = 0.18
    field_weight_parameters: float = 0.12
    explicit_server_boost: float = 0.22
    abstention_guard_enabled: bool = True
    abstention_query_support_threshold: float = 0.18
    abstention_server_support_threshold: float = 0.12
    null_route_enabled: bool = True
    null_route_query_support_threshold: float = 0.213
    null_route_server_support_threshold: float = 0.08
    null_route_tool_support_threshold: float = 0.4
    overlap_aware_enabled: bool = True
    overlap_score_window: float = 0.0025
    overlap_max_group_tools: int = 3
    # Conformal handoff policy — populated by scripts/calibrate_conformal.py
    # and stored in config.local.yaml.  When conformal_enabled=False the engine
    # falls back to the legacy gap-threshold policy.
    conformal_enabled: bool = False
    nc_threshold_k1: float = -0.30   # nc ≤ this → top-1 (high)
    nc_threshold_k3: float = -0.10   # nc ≤ this → top-3 (medium)
    nc_threshold_k5: float =  0.30   # nc ≤ this → top-5 (low)
    nc_threshold_null: float = 0.60  # nc > this  → null-route (abstain)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    base_path: Optional[Path] = None,
    local_path: Optional[Path] = None,
) -> RouterConfig:
    root = Path(__file__).resolve().parents[2]
    base = base_path or (root / "config.yaml")
    data: dict[str, Any] = {}
    if base.is_file():
        with open(base, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    local = local_path or (root / "config.local.yaml")
    if local.is_file():
        with open(local, encoding="utf-8") as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)
    # Path coercion (relative paths = relative to project root, not CWD)
    for key in ("index_dir",):
        if key in data and data[key] is not None:
            p = Path(data[key])
            data[key] = p if p.is_absolute() else (root / p)
    return RouterConfig.model_validate(data)


def openai_api_key(cfg: RouterConfig) -> Optional[str]:
    return os.environ.get(cfg.openai_api_key_env)
