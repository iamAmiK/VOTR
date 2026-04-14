from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from mcp_router.config import RouterConfig
from mcp_router.retrieval.adaptive_k import adaptive_top_k
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.intent_rerank import (
    is_plural_intent,
    is_singular_intent,
    looks_bulk_tool,
    looks_destructive,
)
from mcp_router.retrieval.query_fields import decompose_query
from mcp_router.retrieval.field_scoring import scoring_weights
from mcp_router.retrieval.field_rerank import (
    field_aware_bonus,
    field_match_components,
    normalized_query_support,
)
from mcp_router.retrieval.tool_index import ToolIndex
from mcp_router.retrieval.hybrid import HybridRetriever
from mcp_router.schema_compress.compressor import compress_tool_line
from mcp_router.session.memory import SessionMemory


class RoutedTool(BaseModel):
    tool_key: str
    server_name: str
    tool_name: str
    score: float
    compressed: str
    description: str
    parameter: Dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    tools: List[RoutedTool]
    adaptive_k: int
    top1_score: float = 0.0
    top2_score: float = 0.0
    score_gap: float = 0.0
    confidence: str = "low"
    recommended_handoff_k: int = 5
    null_route: bool = False
    overlap_ambiguous: bool = False
    overlap_tool_keys: List[str] = Field(default_factory=list)
    overlap_servers: List[str] = Field(default_factory=list)


class RouterEngine:
    def __init__(
        self,
        cfg: RouterConfig,
        index: ToolIndex,
        embedder: OpenAIEmbedder,
        sessions: SessionMemory,
    ):
        self.cfg = cfg
        self.index = index
        self.embedder = embedder
        self.sessions = sessions
        self.hybrid = HybridRetriever(index.servers, cfg)

    @staticmethod
    def _nonconformity_score(
        top1: float,
        top2: float,
        top1_server: str,
        explicit_server: Optional[str],
    ) -> float:
        """
        Non-conformity score. Lower = more confident = smaller prediction set.
        Derived from calibration analysis of 500-case benchmark:
          - RRF score absolute values compress (not useful standalone)
          - Gap is the primary discriminative axis; log-transformed to spread
            the distribution across the three k-buckets
          - Ratio provides a secondary signal
          - Explicit server match gives a structural confidence bonus
        """
        import math
        gap = max(top1 - top2, 1e-9)
        ratio = (top2 / top1) if top1 > 1e-9 else 1.0
        server_match = 1.0 if (explicit_server and top1_server == explicit_server) else 0.0
        nc_gap   = -math.log10(gap) - 3.0   # gap=0.001 → 0.0, gap=0.0001 → 1.0
        nc_ratio = ratio - 0.975
        return nc_gap + (nc_ratio * 0.5) - (server_match * 0.3)

    def _handoff_policy(
        self,
        top_scores: List[float],
        available: int,
        top1_server: str = "",
        explicit_server: Optional[str] = None,
    ) -> tuple[float, float, float, str, int]:
        top1 = float(top_scores[0]) if len(top_scores) >= 1 else 0.0
        top2 = float(top_scores[1]) if len(top_scores) >= 2 else 0.0
        gap = top1 - top2

        if self.cfg.conformal_enabled:
            nc = self._nonconformity_score(top1, top2, top1_server, explicit_server)
            if nc <= self.cfg.nc_threshold_k1:
                conf, k = "high", self.cfg.handoff_k_high
            elif nc <= self.cfg.nc_threshold_k3:
                conf, k = "medium", self.cfg.handoff_k_medium
            else:
                conf, k = "low", self.cfg.handoff_k_low
        else:
            if gap >= self.cfg.handoff_gap_high:
                conf, k = "high", self.cfg.handoff_k_high
            elif gap >= self.cfg.handoff_gap_medium:
                conf, k = "medium", self.cfg.handoff_k_medium
            else:
                conf, k = "low", self.cfg.handoff_k_low

        k = max(1, min(k, available))
        return top1, top2, gap, conf, k

    @staticmethod
    def _downgrade_confidence(confidence: str) -> str:
        if confidence == "high":
            return "medium"
        if confidence == "medium":
            return "low"
        return confidence

    def _should_null_route(
        self,
        *,
        query_fields: Any,
        field_weights: Any,
        boosted: list[tuple[int, str, float, float]],
    ) -> bool:
        if (
            not self.cfg.null_route_enabled
            or not boosted
            or query_fields is None
            or field_weights is None
            or query_fields.explicit_server_name is not None
        ):
            return False
        top_server, top_tool = self.index.get_tool_record(int(boosted[0][0]))
        support_parts = field_match_components(top_server, top_tool, query_fields)
        query_support = normalized_query_support(top_server, top_tool, query_fields, field_weights)
        tool_support = max(support_parts["tool_name"], support_parts["tool_description"])
        return (
            query_support < self.cfg.null_route_query_support_threshold
            and tool_support <= self.cfg.null_route_tool_support_threshold
        )

    def route(
        self,
        server_intent: str,
        tool_intent: str,
        session_id: Optional[str] = None,
        record_session: bool = True,
    ) -> RouteResponse:
        s_emb = self.embedder.embed(server_intent)
        t_emb = self.embedder.embed(tool_intent)
        q = f"{server_intent} {tool_intent}".strip()
        cap = max(
            self.cfg.adaptive_max_tools * self.cfg.bm25_candidate_multiplier,
            40,
        )

        ranked_lists: List[List[int]] = []
        weights: List[float] = []

        if self.cfg.dense_retrieval_enabled:
            _, srv_ix = self.index.search_servers(
                s_emb,
                self.cfg.default_top_servers,
                max_weight=self.cfg.server_score_max_weight,
                mean_weight=self.cfg.server_score_mean_weight,
            )
            srv_ix = srv_ix[srv_ix >= 0]
            cand = self.index.tools_for_servers(srv_ix)
            if cand.size > 0:
                hier = self.index.score_tools_hierarchical(
                    cand,
                    t_emb,
                    s_emb,
                    server_score_max_weight=self.cfg.server_score_max_weight,
                    server_score_mean_weight=self.cfg.server_score_mean_weight,
                )
                order = np.argsort(-hier)
                embed_list = cand[order].tolist()[:cap]
                if embed_list:
                    ranked_lists.append(embed_list)
                    weights.append(self.cfg.dense_rrf_weight)

        if self.cfg.bm25_retrieval_enabled:
            bm25_pairs = self.hybrid.bm25_rank(q, top_n=cap)
            bm25_list = [tr for tr, _ in bm25_pairs]
            if bm25_list:
                ranked_lists.append(bm25_list)
                weights.append(self.cfg.bm25_rrf_weight)

        if self.cfg.splade_enabled:
            splade_cap = max(cap, self.cfg.adaptive_max_tools * self.cfg.splade_candidate_multiplier)
            splade_pairs = self.hybrid.splade_rank(q, top_n=splade_cap)
            splade_list = [tr for tr, _ in splade_pairs]
            if splade_list:
                ranked_lists.append(splade_list)
                weights.append(self.cfg.splade_rrf_weight)

        if not ranked_lists:
            return RouteResponse(tools=[], adaptive_k=0)

        rrf_scores = self.hybrid.rrf_fusion(ranked_lists, k=self.cfg.rrf_k, weights=weights)

        merged_rows: List[int] = sorted(rrf_scores.keys(), key=lambda tr: -rrf_scores[tr])

        candidates: List[tuple[int, str, float]] = []
        for tr in merged_rows:
            server, tool = self.index.get_tool_record(int(tr))
            key = server.tool_key(tool.name)
            candidates.append((int(tr), key, float(rrf_scores[tr])))

        if not candidates:
            return RouteResponse(tools=[], adaptive_k=0)

        sorted_by_rrf = sorted(candidates, key=lambda x: -x[2])
        head_n = max(15, self.cfg.adaptive_max_tools * 2)
        rrf_head = [c[2] for c in sorted_by_rrf[:head_n]]
        k = adaptive_top_k(rrf_head, self.cfg)
        k = min(max(k, self.cfg.adaptive_min_tools), len(candidates), self.cfg.adaptive_max_tools)

        singular = is_singular_intent(tool_intent)
        plural = is_plural_intent(tool_intent)
        destructive = looks_destructive(tool_intent)
        query_fields = None
        field_weights = None
        field_head = 0
        if self.cfg.field_aware_enabled:
            query_fields = decompose_query(
                server_intent,
                tool_intent,
                [server.name for server in self.index.servers],
            )
            field_weights = scoring_weights(self.cfg, query_fields)
            field_head = max(1, int(self.cfg.field_aware_rerank_head))

        boosted = []
        top_rrf_score = float(candidates[0][2]) if candidates else 0.0
        top_server_name = self.index.get_tool_record(candidates[0][0])[0].name if candidates else ""
        for idx, (tr, key, sc) in enumerate(candidates):
            server, tool = self.index.get_tool_record(tr)
            adj = sc
            handoff_sc = sc

            if (
                query_fields is not None
                and field_weights is not None
                and idx < field_head
                and (top_rrf_score - sc) <= self.cfg.field_aware_score_window
                and (
                    (
                        query_fields.explicit_server_name is not None
                        and query_fields.explicit_server_name == top_server_name
                        and server.name == query_fields.explicit_server_name
                    )
                    or (
                        query_fields.explicit_server_name is None
                        and server.name == top_server_name
                    )
                )
            ):
                adj += self.cfg.field_bonus_scale * field_aware_bonus(server, tool, query_fields, field_weights)

            # Intent-aware disambiguation:
            # - If user likely wants a singular destructive action, downrank bulk tools.
            # - If user likely asks for plural/batch, mildly boost bulk tools.
            if looks_bulk_tool(tool.name, tool.description) and destructive:
                if singular:
                    adj *= self.cfg.bulk_penalty_singular
                    handoff_sc *= self.cfg.bulk_penalty_singular
                elif plural:
                    adj *= self.cfg.bulk_boost_plural
                    handoff_sc *= self.cfg.bulk_boost_plural
            boosted.append((tr, key, adj, handoff_sc))
        boosted.sort(key=lambda x: -x[2])
        boosted_scores = [x[3] for x in boosted]
        boosted_top1_server = self.index.get_tool_record(int(boosted[0][0]))[0].name if boosted else ""
        boosted_explicit_server = query_fields.explicit_server_name if query_fields is not None else None
        if self.cfg.handoff_enabled:
            top1, top2, gap, confidence, handoff_k = self._handoff_policy(
                boosted_scores,
                len(boosted),
                top1_server=boosted_top1_server,
                explicit_server=boosted_explicit_server,
            )
        else:
            top1 = float(boosted_scores[0]) if boosted_scores else 0.0
            top2 = float(boosted_scores[1]) if len(boosted_scores) >= 2 else 0.0
            gap = top1 - top2
            confidence = "disabled"
            handoff_k = k

        overlap_ambiguous = False
        overlap_tool_keys: list[str] = []
        overlap_servers: list[str] = []
        if self.cfg.overlap_aware_enabled and boosted:
            top_tr = int(boosted[0][0])
            top_score = float(boosted[0][2])
            sig = self.index.capability_signature(top_tr)
            if sig:
                cluster: list[tuple[int, str, float]] = []
                seen_servers: set[str] = set()
                for tr, key, adj, _ in boosted[: max(k, self.cfg.overlap_max_group_tools * 2)]:
                    if self.index.capability_signature(int(tr)) != sig:
                        continue
                    server, _tool = self.index.get_tool_record(int(tr))
                    if server.name in seen_servers:
                        continue
                    if (top_score - float(adj)) > self.cfg.overlap_score_window:
                        continue
                    seen_servers.add(server.name)
                    cluster.append((int(tr), key, float(adj)))
                if len(cluster) >= 2:
                    overlap_ambiguous = True
                    overlap_tool_keys = [key for _, key, _ in cluster[: self.cfg.overlap_max_group_tools]]
                    overlap_servers = [
                        self.index.get_tool_record(tr)[0].name
                        for tr, _, _ in cluster[: self.cfg.overlap_max_group_tools]
                    ]
                    handoff_k = max(handoff_k, min(len(overlap_tool_keys), len(boosted), self.cfg.adaptive_max_tools))
                    k = max(k, handoff_k)
                    if confidence != "disabled":
                        confidence = self._downgrade_confidence(confidence)

        if (
            self.cfg.abstention_guard_enabled
            and boosted
            and query_fields is not None
            and field_weights is not None
            and query_fields.explicit_server_name is None
        ):
            top_server, top_tool = self.index.get_tool_record(int(boosted[0][0]))
            support_parts = field_match_components(top_server, top_tool, query_fields)
            query_support = normalized_query_support(top_server, top_tool, query_fields, field_weights)
            server_support = max(support_parts["server_name"], support_parts["server_summary"])
            if (
                query_support < self.cfg.abstention_query_support_threshold
                and server_support < self.cfg.abstention_server_support_threshold
            ):
                confidence = "low" if confidence != "disabled" else confidence
                handoff_k = max(handoff_k, min(self.cfg.handoff_k_low, len(boosted)))
                k = max(k, handoff_k)

        # Low-confidence: return at least N candidates (when not abstaining via null_route)
        # so downstream LLMs can choose; ranking and confidence label are unchanged.
        if (
            self.cfg.low_confidence_retrieval_floor_k > 0
            and boosted
            and confidence == "low"
        ):
            floor = min(
                self.cfg.low_confidence_retrieval_floor_k,
                len(boosted),
                self.cfg.adaptive_max_tools,
            )
            k = max(k, floor)
            handoff_k = max(handoff_k, floor)

        null_route = self._should_null_route(
            query_fields=query_fields,
            field_weights=field_weights,
            boosted=boosted,
        )
        if null_route:
            k = 0
            handoff_k = 0
            overlap_ambiguous = False
            overlap_tool_keys = []
            overlap_servers = []
            confidence = "low" if confidence != "disabled" else confidence

        out: List[RoutedTool] = []
        keys_added: List[str] = []
        for tr, key, _, _ in boosted[:k]:
            server, tool = self.index.get_tool_record(tr)
            sc_rrf = float(rrf_scores[tr])
            out.append(
                RoutedTool(
                    tool_key=key,
                    server_name=server.name,
                    tool_name=tool.name,
                    score=sc_rrf,
                    compressed=compress_tool_line(server.name, tool),
                    description=tool.description,
                    parameter=dict(tool.parameter),
                )
            )
            keys_added.append(key)

        if record_session and session_id and keys_added:
            self.sessions.add_tools(session_id, keys_added)

        return RouteResponse(
            tools=out,
            adaptive_k=k,
            top1_score=top1,
            top2_score=top2,
            score_gap=gap,
            confidence=confidence,
            recommended_handoff_k=handoff_k,
            null_route=null_route,
            overlap_ambiguous=overlap_ambiguous,
            overlap_tool_keys=overlap_tool_keys,
            overlap_servers=overlap_servers,
        )
