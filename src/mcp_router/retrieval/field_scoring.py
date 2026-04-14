from __future__ import annotations

from dataclasses import dataclass

from mcp_router.config import RouterConfig
from mcp_router.retrieval.query_fields import QueryFields


@dataclass(frozen=True)
class FieldScoringWeights:
    server_name: float
    server_summary: float
    tool_name: float
    tool_description: float
    parameters: float
    explicit_server_boost: float


def scoring_weights(cfg: RouterConfig, query_fields: QueryFields) -> FieldScoringWeights:
    parameter_weight = cfg.field_weight_parameters
    if not query_fields.constraint_query:
        parameter_weight *= 0.35

    return FieldScoringWeights(
        server_name=cfg.field_weight_server_name,
        server_summary=cfg.field_weight_server_summary,
        tool_name=cfg.field_weight_tool_name,
        tool_description=cfg.field_weight_tool_description,
        parameters=parameter_weight,
        explicit_server_boost=cfg.explicit_server_boost,
    )
